import AIProviderSpec

/// What a tool returns to the model.
public typealias ToolOutput = ToolResultOutput

/// Information about the call a tool is answering.
public struct ToolContext: Sendable {
    /// The provider-assigned identifier of the call.
    public var toolCallID: String

    /// The conversation as it stood when the model asked for the call.
    ///
    /// Useful for tools whose behaviour depends on what came before — a search tool that wants
    /// the original question, say, rather than only the query the model distilled from it.
    public var messages: [ModelMessage]

    /// Which step of the loop this is, counting from zero.
    public var stepNumber: Int

    public init(toolCallID: String, messages: [ModelMessage], stepNumber: Int) {
        self.toolCallID = toolCallID
        self.messages = messages
        self.stepNumber = stepNumber
    }
}

/// How a tool call is carried out.
public enum ToolExecution: Sendable, Hashable {
    /// The SDK calls ``Tool/call(_:context:)`` and feeds the result back to the model.
    case local

    /// The application resolves the call.
    ///
    /// The loop stops as soon as the model asks for a client-executed tool, and the result
    /// reports ``FinishReason/toolCalls``. Append a tool message with the result and call again to
    /// continue. This is how a tool that needs the user — a confirmation, a file picker, a
    /// payment — is modeled.
    case client

    /// The provider runs the tool itself and returns the result inline.
    ///
    /// Web search and code interpreters work this way. The SDK never invokes these.
    case provider(ProviderDefinedTool)
}

/// Something a model can be offered as a callable tool.
///
/// The shape mirrors what a model needs to decide: a name, a description of when to use it, and a
/// schema for its arguments. Conform a type to it when the tool has state or dependencies:
///
/// ```swift
/// struct WeatherTool: Tool {
///     @Structured("Where and when to look up the weather.")
///     struct Arguments {
///         @Guidance("The city, as the user wrote it.")
///         var city: String
///         @Guidance("How many days ahead to forecast.", .range(1...7))
///         var days: Int
///     }
///
///     let client: WeatherClient
///     var description: String { "Look up the weather forecast for a city." }
///
///     func call(_ arguments: Arguments, context: ToolContext) async throws -> ToolOutput {
///         .json(try await client.forecast(city: arguments.city, days: arguments.days))
///     }
/// }
/// ```
///
/// For a tool that is just a closure, ``tool(_:description:arguments:providerOptions:execute:)``
/// avoids declaring a type at all.
///
/// Tools are held as `[any Tool]`. Dispatch is inherently dynamic — the model picks a tool by
/// name at runtime — so nothing is gained by tracking the set in the type system, and a great
/// deal of complexity is avoided.
public protocol Tool: Sendable {
    /// The arguments the tool takes.
    ///
    /// Usually a `@Structured` type. Defaults to ``AIProviderSpec/JSONValue`` for tools whose
    /// shape is only known at runtime.
    associatedtype Arguments: StructuredValue = JSONValue

    /// The name the model uses to call the tool.
    ///
    /// Must be unique within a request. Defaults to the conforming type's name.
    var name: String { get }

    /// What the tool does and when to use it.
    ///
    /// This is the model's main signal for choosing between tools, so it deserves more care than
    /// a passing comment. Say what the tool is for, not how it works.
    var description: String { get }

    /// The schema for ``Arguments``. Defaults to the argument type's own schema.
    var inputSchema: JSONSchema { get }

    /// Who runs the tool. Defaults to ``ToolExecution/local``.
    var execution: ToolExecution { get }

    /// Provider-specific settings scoped to this tool, such as cache control.
    var providerOptions: ProviderOptions? { get }

    /// Runs the tool.
    ///
    /// Throwing is a supported outcome: the loop turns a thrown error into an error result that
    /// the model sees and can react to, rather than failing the whole generation. Reserve
    /// throwing for genuine failures, and return ``ToolOutput/errorText(_:)`` for expected ones
    /// such as "no such city", where the message itself is useful to the model.
    ///
    /// - Parameters:
    ///   - arguments: The decoded arguments.
    ///   - context: Which call this answers, and the conversation so far.
    /// - Returns: What to show the model.
    func call(_ arguments: Arguments, context: ToolContext) async throws -> ToolOutput
}

extension Tool {
    public var name: String { String(describing: Self.self) }
    public var inputSchema: JSONSchema { Arguments.jsonSchema }
    public var execution: ToolExecution { .local }
    public var providerOptions: ProviderOptions? { nil }
}

// MARK: - Closure tools

/// A tool defined by a closure rather than a type.
///
/// Created by ``tool(_:description:arguments:providerOptions:execute:)``; there is rarely a
/// reason to name this type directly.
public struct ClosureTool<Arguments: StructuredValue>: Tool {
    public let name: String
    public let description: String
    public let inputSchema: JSONSchema
    public let execution: ToolExecution
    public let providerOptions: ProviderOptions?

    private let body: (@Sendable (Arguments, ToolContext) async throws -> ToolOutput)?

    init(
        name: String,
        description: String,
        inputSchema: JSONSchema,
        execution: ToolExecution = .local,
        providerOptions: ProviderOptions? = nil,
        body: (@Sendable (Arguments, ToolContext) async throws -> ToolOutput)?
    ) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.execution = execution
        self.providerOptions = providerOptions
        self.body = body
    }

    public func call(_ arguments: Arguments, context: ToolContext) async throws -> ToolOutput {
        guard let body else {
            // Unreachable through the loop, which checks `execution` first. Reaching it means a
            // caller invoked a client- or provider-executed tool by hand.
            throw InvalidArgumentError(
                argument: name,
                message: "The tool '\(name)' has no local implementation and cannot be called directly."
            )
        }
        return try await body(arguments, context)
    }
}

/// Defines a tool from a closure.
///
/// ```swift
/// @Structured("A city to look up.")
/// struct CityQuery {
///     @Guidance("The city name.") var city: String
/// }
///
/// let weather = tool("weather", description: "Look up the current weather in a city.") {
///     (query: CityQuery, _) in
///     .text(try await forecast(for: query.city))
/// }
/// ```
///
/// - Parameters:
///   - name: The name the model calls. Must be unique within a request.
///   - description: What the tool does and when to use it.
///   - arguments: The argument type. Usually inferred from `execute`.
///   - providerOptions: Provider-specific settings scoped to this tool.
///   - execute: The tool body.
/// - Returns: A tool ready to pass to a generation call.
public func tool<Arguments: StructuredValue>(
    _ name: String,
    description: String,
    arguments: Arguments.Type = Arguments.self,
    providerOptions: ProviderOptions? = nil,
    execute: @escaping @Sendable (Arguments, ToolContext) async throws -> ToolOutput
) -> ClosureTool<Arguments> {
    ClosureTool(
        name: name,
        description: description,
        inputSchema: Arguments.jsonSchema,
        providerOptions: providerOptions,
        body: execute
    )
}

/// Defines a tool whose argument shape is only known at runtime.
///
/// Use this for tools loaded from configuration or discovered from a remote server, where there
/// is no Swift type to describe them. Arguments arrive as raw ``AIProviderSpec/JSONValue``.
///
/// ```swift
/// let search = dynamicTool(
///     "search",
///     description: "Search the knowledge base.",
///     inputSchema: .object(properties: ["query": .string()], required: ["query"])
/// ) { input, _ in
///     .text(try await index.search(input["query"]?.stringValue ?? ""))
/// }
/// ```
public func dynamicTool(
    _ name: String,
    description: String,
    inputSchema: JSONSchema,
    providerOptions: ProviderOptions? = nil,
    execute: @escaping @Sendable (JSONValue, ToolContext) async throws -> ToolOutput
) -> ClosureTool<JSONValue> {
    ClosureTool(
        name: name,
        description: description,
        inputSchema: inputSchema,
        providerOptions: providerOptions,
        body: execute
    )
}

/// Declares a tool the application resolves rather than the SDK.
///
/// The loop stops when the model calls one, returning ``FinishReason/toolCalls`` along with the
/// call. Perform the work, append a tool message with the result, and call again:
///
/// ```swift
/// let confirm = clientTool(
///     "confirmPurchase",
///     description: "Ask the user to confirm a purchase before it is made.",
///     inputSchema: .object(properties: ["amount": .number()], required: ["amount"])
/// )
///
/// let result = try await generateText(model: model, prompt: prompt, tools: [confirm])
/// if let call = result.toolCalls.first {
///     let approved = await askTheUser()
///     history += result.responseMessages
///     history.append(.tool([ToolResultPart(
///         toolCallID: call.toolCallID,
///         toolName: call.toolName,
///         output: .json(["approved": .bool(approved)])
///     )]))
/// }
/// ```
public func clientTool(
    _ name: String,
    description: String,
    inputSchema: JSONSchema,
    providerOptions: ProviderOptions? = nil
) -> ClosureTool<JSONValue> {
    ClosureTool(
        name: name,
        description: description,
        inputSchema: inputSchema,
        execution: .client,
        providerOptions: providerOptions,
        body: nil
    )
}

/// Wraps a tool the provider implements and runs itself.
///
/// Providers expose these through their own namespaces — for example a web search tool — and
/// this is the general form behind those conveniences.
public func providerTool(
    _ definition: ProviderDefinedTool,
    description: String = "",
    inputSchema: JSONSchema = .any
) -> ClosureTool<JSONValue> {
    ClosureTool(
        name: definition.name,
        description: description,
        inputSchema: inputSchema,
        execution: .provider(definition),
        body: nil
    )
}

// MARK: - Output conveniences

extension ToolResultOutput {
    /// Encodes any `Encodable` value as a JSON tool result.
    ///
    /// - Throws: ``InvalidArgumentError`` if the value cannot be represented as JSON.
    public static func encoding(_ value: some Encodable) throws -> ToolResultOutput {
        .json(try JSONValue(encoding: value))
    }
}

// MARK: - Erasure

extension Tool {
    /// Decodes raw arguments and runs the tool.
    ///
    /// Called through implicit existential opening, which is what lets `[any Tool]` hold tools
    /// with different argument types while each still decodes into its own.
    func invoke(rawInput: JSONValue, context: ToolContext) async throws -> ToolOutput {
        let arguments: Arguments
        do {
            arguments = try Arguments(structuredJSON: rawInput)
        } catch let error as TypeValidationError {
            throw InvalidToolInputError(toolName: name, rawInput: rawInput, cause: error)
        }
        return try await call(arguments, context: context)
    }

    /// The provider-facing description of this tool.
    func languageModelTool() -> LanguageModelTool {
        if case .provider(let definition) = execution {
            return .providerDefined(definition)
        }
        return .function(
            FunctionTool(
                name: name,
                description: description,
                inputSchema: inputSchema,
                providerOptions: providerOptions
            )
        )
    }
}

import Foundation

/// The contract every language model provider implements.
///
/// This protocol is deliberately small and free of associated types, so models are always usable
/// as `any LanguageModel`. It sits at the bottom of the dependency graph: the core generation
/// functions call it, providers implement it, and neither knows about the other. That inversion
/// is what lets a provider ship independently of the core, and what lets middleware wrap any
/// model uniformly.
///
/// Implementations receive a fully normalized request — messages already flattened, tools already
/// reduced to JSON Schema — and are responsible only for translating to and from their wire
/// format. Cross-cutting concerns are handled above this layer and must **not** be reimplemented
/// by providers:
///
/// - **Retries** are applied by the core, driven by ``APICallError/isRetryable``.
/// - **The tool loop** is run by the core; a provider handles exactly one round trip.
/// - **Cancellation** flows through Swift's cooperative task cancellation. Implementations should
///   propagate it to their HTTP client rather than accepting a cancellation token.
///
/// A provider degrades gracefully: when a requested setting has no equivalent, return a
/// ``CallWarning`` rather than throwing, so the call still produces a result.
///
/// ## Versioning
///
/// The protocol name carries the specification version it implements. When a future revision
/// makes a source-breaking change, it arrives as a new `LanguageModelV3` protocol and the
/// ``LanguageModel`` alias moves to it, leaving existing providers compiling against `V2` while
/// they migrate.
public protocol LanguageModelV2: Sendable {
    /// The provider's short name, such as `"openai"`. Used as the namespace key for
    /// ``ProviderOptions`` and ``ProviderMetadata``, and in registry identifiers.
    var provider: String { get }

    /// The provider's identifier for this model, such as `"gpt-5"`.
    var modelID: String { get }

    /// Whether the provider can fetch a URL itself rather than needing inline bytes.
    ///
    /// Returning `true` lets the SDK pass a ``FilePart`` URL straight through, avoiding a
    /// download and a base64 round trip. The default implementation returns `false`, so the SDK
    /// downloads referenced files and substitutes their bytes.
    ///
    /// - Parameters:
    ///   - url: The URL referenced by a file part.
    ///   - mediaType: The part's declared media type.
    /// - Returns: `true` if the provider accepts the URL directly.
    func supportsNativeURL(_ url: URL, mediaType: String) -> Bool

    /// Performs a single, non-streaming generation.
    ///
    /// - Parameter options: The normalized request.
    /// - Returns: The model's complete response.
    /// - Throws: ``APICallError`` for upstream failures, ``UnsupportedFunctionalityError`` when
    ///   the request cannot be served at all.
    func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse

    /// Performs a single, streaming generation.
    ///
    /// The returned stream must begin with ``LanguageModelStreamPart/streamStart(warnings:)`` and
    /// end with ``LanguageModelStreamPart/finish(finishReason:usage:providerMetadata:)``, unless
    /// it terminates by throwing. Failures that occur after the first part has been emitted are
    /// delivered by finishing the stream with an error rather than by throwing from this method.
    ///
    /// - Parameter options: The normalized request.
    /// - Returns: The stream, along with whatever request and response metadata is available at
    ///   the time the response head arrives.
    /// - Throws: ``APICallError`` if the request fails before any part is produced.
    func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse
}

extension LanguageModelV2 {
    public func supportsNativeURL(_ url: URL, mediaType: String) -> Bool { false }
}

/// The current language model specification.
///
/// Prefer this alias in application code; it always names the current revision.
public typealias LanguageModel = LanguageModelV2

// MARK: - Call options

/// A fully normalized request to a language model.
///
/// Providers receive this rather than the caller's arguments, which means prompt handling,
/// defaulting, and tool preparation happen exactly once, in the core, instead of being
/// reimplemented by every provider.
public struct LanguageModelCallOptions: Sendable {
    /// The conversation, including any system message.
    public var prompt: [ModelMessage]

    /// The maximum number of tokens to generate.
    public var maxOutputTokens: Int?

    /// Sampling temperature. Higher values produce more varied output.
    ///
    /// It is generally best to set either this or ``topP``, not both.
    public var temperature: Double?

    /// Nucleus sampling threshold.
    public var topP: Double?

    /// Restricts sampling to the `k` most likely tokens.
    public var topK: Int?

    /// Penalizes tokens that have already appeared, discouraging repetition.
    public var presencePenalty: Double?

    /// Penalizes tokens in proportion to how often they have appeared.
    public var frequencyPenalty: Double?

    /// A seed for deterministic sampling, where the provider supports it.
    public var seed: Int?

    /// Sequences that, once generated, stop the response.
    public var stopSequences: [String]

    /// Whether the model must answer as free text or as JSON matching a schema.
    public var responseFormat: ResponseFormat?

    /// The tools the model may call.
    public var tools: [LanguageModelTool]

    /// How freely the model may choose to call tools.
    public var toolChoice: ToolChoice?

    /// Whether the provider should emit ``LanguageModelStreamPart/raw(_:)`` parts carrying its
    /// undecoded stream events.
    ///
    /// Off by default. Turn it on to access provider features the SDK does not model yet.
    public var includesRawChunks: Bool

    /// Extra HTTP headers to merge into the request.
    public var headers: [String: String]

    /// Provider-specific settings, namespaced by provider name.
    public var providerOptions: ProviderOptions?

    public init(
        prompt: [ModelMessage],
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        seed: Int? = nil,
        stopSequences: [String] = [],
        responseFormat: ResponseFormat? = nil,
        tools: [LanguageModelTool] = [],
        toolChoice: ToolChoice? = nil,
        includesRawChunks: Bool = false,
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil
    ) {
        self.prompt = prompt
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.seed = seed
        self.stopSequences = stopSequences
        self.responseFormat = responseFormat
        self.tools = tools
        self.toolChoice = toolChoice
        self.includesRawChunks = includesRawChunks
        self.headers = headers
        self.providerOptions = providerOptions
    }
}

/// The form a model's answer must take.
public enum ResponseFormat: Sendable, Hashable {
    /// Free-form text. The default.
    case text

    /// JSON, optionally constrained by a schema.
    ///
    /// Providers that support constrained decoding enforce the schema during generation; others
    /// fall back to instructing the model and validating afterwards.
    ///
    /// - Parameters:
    ///   - schema: The shape the JSON must take, or `nil` for any JSON.
    ///   - name: A name for the schema, which some providers require.
    ///   - description: What the JSON represents. Improves adherence noticeably.
    case json(schema: JSONSchema?, name: String? = nil, description: String? = nil)
}

/// A tool as the model sees it: a name, a description, and a parameter schema.
///
/// The executable body does not appear here. The core keeps that on its side of the boundary,
/// which is what allows a tool to have no body at all — a client-executed tool that the
/// application resolves later.
public enum LanguageModelTool: Sendable, Hashable {
    /// An ordinary function tool, executed by the SDK or the application.
    case function(FunctionTool)

    /// A tool implemented by the provider itself, such as web search or a code interpreter.
    ///
    /// The provider runs these and returns results inline; the SDK never executes them.
    case providerDefined(ProviderDefinedTool)

    /// The tool's name as the model refers to it.
    public var name: String {
        switch self {
        case .function(let tool): return tool.name
        case .providerDefined(let tool): return tool.name
        }
    }
}

/// An ordinary function tool.
public struct FunctionTool: Sendable, Hashable {
    /// The name the model uses to call the tool. Must be unique within a request.
    public var name: String

    /// What the tool does and when to use it. This is the model's primary signal for choosing
    /// between tools, so it is worth writing carefully.
    public var description: String

    /// The schema its arguments must satisfy.
    public var inputSchema: JSONSchema

    /// Provider-specific settings scoped to this tool.
    public var providerOptions: ProviderOptions?

    public init(
        name: String,
        description: String,
        inputSchema: JSONSchema,
        providerOptions: ProviderOptions? = nil
    ) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.providerOptions = providerOptions
    }
}

/// A tool the provider implements and executes itself.
public struct ProviderDefinedTool: Sendable, Hashable {
    /// The fully qualified identifier, namespaced by provider, such as `"openai.web_search"`.
    ///
    /// Providers use the namespace to recognize their own tools and ignore other providers'.
    public var id: String

    /// The name the model uses to call the tool.
    public var name: String

    /// Configuration for the tool, in whatever shape the provider defines.
    public var arguments: [String: JSONValue]

    public init(id: String, name: String, arguments: [String: JSONValue] = [:]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// The provider this tool belongs to, taken from the portion of ``id`` before the first dot.
    public var providerNamespace: String {
        String(id.prefix(while: { $0 != "." }))
    }
}

/// How freely a model may choose to call tools.
public enum ToolChoice: Sendable, Hashable {
    /// The model decides whether to call a tool. The default when tools are present.
    case auto

    /// The model must not call tools, even though it can see them.
    ///
    /// Named `never` rather than `none` to avoid colliding with `Optional.none` at use sites.
    case never

    /// The model must call some tool.
    case required

    /// The model must call this specific tool.
    case tool(named: String)
}

// MARK: - Results

/// The complete response from a non-streaming generation.
public struct LanguageModelResponse: Sendable {
    /// Everything the model produced, in the order it produced it.
    public var content: [ModelContent]

    /// Why generation stopped.
    public var finishReason: FinishReason

    /// Token counts for this call.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// What was sent, for debugging.
    public var request: RequestInfo?

    /// Metadata about the response, including its identifier and headers.
    public var response: ResponseInfo?

    public init(
        content: [ModelContent],
        finishReason: FinishReason,
        usage: Usage = .none,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        request: RequestInfo? = nil,
        response: ResponseInfo? = nil
    ) {
        self.content = content
        self.finishReason = finishReason
        self.usage = usage
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.request = request
        self.response = response
    }
}

/// The response from a streaming generation.
public struct LanguageModelStreamResponse: Sendable {
    /// The parts, in the order the provider produced them.
    public var stream: AsyncThrowingStream<LanguageModelStreamPart, any Error>

    /// What was sent, for debugging.
    public var request: RequestInfo?

    /// Metadata available when the response head arrived. Identifiers that only appear in the
    /// body arrive later, as ``LanguageModelStreamPart/responseMetadata(id:modelID:timestamp:)``.
    public var response: ResponseInfo?

    public init(
        stream: AsyncThrowingStream<LanguageModelStreamPart, any Error>,
        request: RequestInfo? = nil,
        response: ResponseInfo? = nil
    ) {
        self.stream = stream
        self.request = request
        self.response = response
    }
}

/// A record of what was sent to a provider.
public struct RequestInfo: Sendable, Hashable {
    /// The serialized request body.
    ///
    /// - Important: Providers must not include credentials here. Headers are excluded entirely
    ///   for that reason.
    public var body: String?

    public init(body: String? = nil) {
        self.body = body
    }
}

/// Metadata about a provider's response.
public struct ResponseInfo: Sendable, Hashable {
    /// The provider's identifier for this response, useful for support requests and for
    /// providers that support resuming from a response.
    public var id: String?

    /// The model that actually served the request, which may be more specific than the one asked
    /// for — a dated snapshot behind an alias, for instance.
    public var modelID: String?

    /// When the provider generated the response.
    public var timestamp: Date?

    /// Response headers. Carries rate-limit and request-identifier metadata.
    public var headers: [String: String]

    /// The raw response body, populated only when a provider is configured to retain it.
    public var body: String?

    public init(
        id: String? = nil,
        modelID: String? = nil,
        timestamp: Date? = nil,
        headers: [String: String] = [:],
        body: String? = nil
    ) {
        self.id = id
        self.modelID = modelID
        self.timestamp = timestamp
        self.headers = headers
        self.body = body
    }
}

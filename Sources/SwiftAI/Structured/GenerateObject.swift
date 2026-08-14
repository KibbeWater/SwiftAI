import AIProviderSpec
import AIProviderUtils
import Foundation

/// The outcome of a structured generation.
public struct GenerateObjectResult<Value: Sendable>: Sendable {
    /// The decoded value.
    public var object: Value

    /// Why generation stopped. Anything other than ``FinishReason/stop`` is worth inspecting even
    /// though the value decoded successfully.
    public var finishReason: FinishReason

    /// Token usage for the call.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// What was sent, for debugging.
    public var request: RequestInfo?

    /// Metadata about the response.
    public var response: ResponseInfo?

    /// The raw JSON the model produced, before decoding.
    ///
    /// Useful when a value decodes but looks wrong, and for logging what the model actually said.
    public var rawText: String

    public init(
        object: Value,
        finishReason: FinishReason,
        usage: Usage,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        request: RequestInfo? = nil,
        response: ResponseInfo? = nil,
        rawText: String
    ) {
        self.object = object
        self.finishReason = finishReason
        self.usage = usage
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.request = request
        self.response = response
        self.rawText = rawText
    }
}

// MARK: - Public API

/// Generates a value of a given type.
///
/// The type's schema is sent to the model, which constrains what it can produce — on providers
/// with constrained decoding, the output is guaranteed to parse. The result is decoded and
/// validated before it is returned, so what you get back is a real Swift value, not a string to
/// parse yourself.
///
/// ```swift
/// @Structured("A recipe with its ingredients.")
/// struct Recipe {
///     @Guidance("The name of the dish.") var name: String
///     @Guidance("How many it serves.", .range(1...12)) var servings: Int
///     var ingredients: [String]
/// }
///
/// let result = try await generateObject(model: model, of: Recipe.self, prompt: "A quick pasta dish.")
/// print(result.object.ingredients)
/// ```
///
/// - Parameters:
///   - model: The model to call.
///   - type: The type to generate. Usually inferred from context.
///   - system: Instructions framing the request.
///   - prompt: What to generate.
///   - settings: Temperature, token limits, retries, and provider-specific options.
/// - Returns: The decoded value, with usage and metadata.
/// - Throws: ``NoObjectGeneratedError`` when the model's output does not parse or does not match
///   the schema — the raw text is attached so you can see what it produced. Also the usual
///   ``APICallError`` for upstream failures.
public func generateObject<Value: StructuredOutput>(
    model: any LanguageModel,
    of type: Value.Type = Value.self,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) async throws -> GenerateObjectResult<Value> {
    let outcome = try await generateStructured(
        model: model,
        mode: .object(Value.jsonSchema),
        system: system,
        prompt: prompt,
        settings: settings
    )
    return try outcome.map { json in try Value(structuredJSON: json) }
}

/// Generates a list of values.
///
/// The list is requested inside a wrapper object, because most providers reject a bare array as a
/// top-level structured response. The wrapper is removed before the result is returned.
///
/// ```swift
/// let result = try await generateObject(
///     model: model,
///     arrayOf: Recipe.self,
///     prompt: "Three weeknight dinners."
/// )
/// ```
public func generateObject<Element: StructuredOutput>(
    model: any LanguageModel,
    arrayOf type: Element.Type,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) async throws -> GenerateObjectResult<[Element]> {
    let outcome = try await generateStructured(
        model: model,
        mode: .array(element: Element.jsonSchema),
        system: system,
        prompt: prompt,
        settings: settings
    )
    return try outcome.map { json in try [Element](structuredJSON: json) }
}

/// Chooses one of a fixed set of values.
///
/// The natural fit for classification, where the answer must be one of a known set and anything
/// else is a defect.
///
/// ```swift
/// let result = try await generateObject(
///     model: model,
///     enumOf: ["positive", "neutral", "negative"],
///     prompt: "Classify: \(review)"
/// )
/// ```
///
/// - Throws: ``InvalidArgumentError`` if `values` is empty.
public func generateObject(
    model: any LanguageModel,
    enumOf values: [String],
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) async throws -> GenerateObjectResult<String> {
    guard !values.isEmpty else {
        throw InvalidArgumentError(argument: "values", message: "At least one value is required.")
    }
    let outcome = try await generateStructured(
        model: model,
        mode: .enumeration(values),
        system: system,
        prompt: prompt,
        settings: settings
    )
    return try outcome.map { json in
        guard let choice = json.stringValue, values.contains(choice) else {
            throw TypeValidationError(
                message: "'\(json)' is not one of: \(values.joined(separator: ", ")).",
                value: json
            )
        }
        return choice
    }
}

/// Generates a value matching a schema built at runtime.
///
/// For shapes that are not known at compile time — loaded from configuration, or assembled from
/// user input. The result is raw JSON, since there is no Swift type to decode into.
public func generateObject(
    model: any LanguageModel,
    schema: JSONSchema,
    schemaName: String? = nil,
    schemaDescription: String? = nil,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) async throws -> GenerateObjectResult<JSONValue> {
    let outcome = try await generateStructured(
        model: model,
        mode: .object(schema),
        schemaName: schemaName,
        schemaDescription: schemaDescription,
        system: system,
        prompt: prompt,
        settings: settings
    )
    return try outcome.map { $0 }
}

/// Generates JSON with no schema at all.
///
/// The model is asked for JSON and the response is parsed, but nothing constrains its shape. Use
/// this only when the structure genuinely cannot be described in advance; a schema improves both
/// reliability and quality.
public func generateObject(
    model: any LanguageModel,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) async throws -> GenerateObjectResult<JSONValue> {
    let outcome = try await generateStructured(
        model: model,
        mode: .free,
        system: system,
        prompt: prompt,
        settings: settings
    )
    return try outcome.map { $0 }
}

// MARK: - Implementation

/// A structured generation before the caller's type has been applied.
struct StructuredOutcome: Sendable {
    var json: JSONValue
    var rawText: String
    var step: StepResult

    /// Applies a decoder, turning any failure into ``NoObjectGeneratedError`` with the model's
    /// output attached.
    func map<Value: Sendable>(_ decode: (JSONValue) throws -> Value) throws -> GenerateObjectResult<Value> {
        let object: Value
        do {
            object = try decode(json)
        } catch {
            throw NoObjectGeneratedError(
                text: rawText,
                finishReason: step.finishReason,
                usage: step.usage,
                cause: error,
                response: step.response
            )
        }
        return GenerateObjectResult(
            object: object,
            finishReason: step.finishReason,
            usage: step.usage,
            warnings: step.warnings,
            providerMetadata: step.providerMetadata,
            request: step.request,
            response: step.response,
            rawText: rawText
        )
    }
}

/// Runs a single structured generation and parses the response.
func generateStructured(
    model: any LanguageModel,
    mode: ObjectOutputMode,
    schemaName: String? = nil,
    schemaDescription: String? = nil,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings
) async throws -> StructuredOutcome {
    let loop = ToolLoop(
        model: model,
        tools: [],
        toolChoice: .auto,
        settings: settings,
        stopWhen: [.stepCount(1)],
        responseFormat: .json(
            schema: mode.requestSchema,
            name: schemaName ?? mode.schemaName,
            description: schemaDescription
        ),
        activeTools: nil,
        prepareStep: nil,
        observer: nil
    )

    let steps = try await loop.run(messages: try prompt.resolved(system: system)) {
        model, options, retryPolicy, _ in
        try await withRetries(policy: retryPolicy) { _ in
            let response = try await model.generate(options)
            return RawStep(
                content: response.content,
                finishReason: response.finishReason,
                usage: response.usage,
                warnings: response.warnings,
                providerMetadata: response.providerMetadata,
                request: response.request,
                response: response.response
            )
        }
    }

    guard let step = steps.last else {
        throw NoObjectGeneratedError(message: "The generation produced no steps.")
    }

    let rawText = step.text
    guard !rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw NoObjectGeneratedError(
            message: "The model returned no text to parse.",
            text: rawText,
            finishReason: step.finishReason,
            usage: step.usage,
            response: step.response
        )
    }

    let parsed: JSONValue
    do {
        parsed = try JSONValue.parse(rawText)
    } catch {
        throw NoObjectGeneratedError(
            text: rawText,
            finishReason: step.finishReason,
            usage: step.usage,
            cause: error,
            response: step.response
        )
    }

    guard let unwrapped = mode.unwrap(parsed) else {
        throw NoObjectGeneratedError(
            message: "The response was missing the expected wrapper property.",
            text: rawText,
            finishReason: step.finishReason,
            usage: step.usage,
            response: step.response
        )
    }

    return StructuredOutcome(json: unwrapped, rawText: rawText, step: step)
}

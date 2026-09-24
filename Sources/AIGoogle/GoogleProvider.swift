import AIProviderSpec
import AIProviderUtils
import Foundation

// MARK: - Wire types

/// A `generateContent` response, and also one chunk of a streamed one — the shapes are identical,
/// which is why a single decoder serves both.
struct GoogleGenerateContentResponse: Decodable {
    var candidates: [Candidate]?
    var usageMetadata: UsageMetadata?
    var modelVersion: String?
    var responseId: String?
    var promptFeedback: PromptFeedback?

    struct Candidate: Decodable {
        var content: Content?
        var finishReason: String?
        var index: Int?
        var citationMetadata: CitationMetadata?
    }

    struct Content: Decodable {
        var role: String?
        var parts: [Part]?
    }

    struct Part: Decodable {
        var text: String?
        /// Marks a reasoning part, when thought summaries are enabled.
        var thought: Bool?
        var functionCall: FunctionCall?
        var inlineData: InlineData?
    }

    struct FunctionCall: Decodable {
        var name: String?
        var args: JSONValue?
    }

    struct InlineData: Decodable {
        var mimeType: String?
        var data: String?
    }

    struct CitationMetadata: Decodable {
        var citationSources: [CitationSource]?

        struct CitationSource: Decodable {
            var uri: String?
            var title: String?
        }
    }

    struct PromptFeedback: Decodable {
        var blockReason: String?
    }

    struct UsageMetadata: Decodable {
        var promptTokenCount: Int?
        var candidatesTokenCount: Int?
        var totalTokenCount: Int?
        var thoughtsTokenCount: Int?
        var cachedContentTokenCount: Int?

        var normalized: Usage {
            Usage(
                inputTokens: promptTokenCount,
                outputTokens: candidatesTokenCount,
                totalTokenCount: totalTokenCount,
                thoughtsTokenCount: thoughtsTokenCount,
                cachedContentTokenCount: cachedContentTokenCount
            )
        }
    }
}

extension Usage {
    /// Builds usage from Gemini's counts.
    ///
    /// Reasoning tokens are reported separately and are *not* included in `candidatesTokenCount`,
    /// so the output count has to be assembled rather than taken directly — otherwise a thinking
    /// model appears far cheaper than it is.
    fileprivate init(
        inputTokens: Int?,
        outputTokens: Int?,
        totalTokenCount: Int?,
        thoughtsTokenCount: Int?,
        cachedContentTokenCount: Int?
    ) {
        self.init(
            inputTokens: inputTokens,
            outputTokens: outputTokens.map { $0 + (thoughtsTokenCount ?? 0) } ?? thoughtsTokenCount,
            totalTokens: totalTokenCount,
            reasoningTokens: thoughtsTokenCount,
            cachedInputTokens: cachedContentTokenCount
        )
    }
}

extension FinishReason {
    /// Maps a Gemini finish reason.
    static func fromGoogle(_ raw: String?, hasFunctionCall: Bool) -> FinishReason {
        // Gemini reports STOP even when the model asked for a tool, so the content decides.
        if hasFunctionCall { return .toolCalls }
        switch raw {
        case "STOP": return .stop
        case "MAX_TOKENS": return .length
        case "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII": return .contentFilter
        case "RECITATION": return .contentFilter
        case "MALFORMED_FUNCTION_CALL": return .error
        case nil: return .unknown
        default: return .other
        }
    }
}

extension GoogleGenerateContentResponse {
    /// The parts of the first candidate, converted to the shared representation.
    func modelContent() -> [ModelContent] {
        guard let parts = candidates?.first?.content?.parts else { return [] }
        var content: [ModelContent] = []

        for part in parts {
            if let call = part.functionCall, let name = call.name {
                content.append(
                    .toolCall(
                        ToolCallPart(
                            // Gemini does not issue tool call identifiers, so one is generated
                            // here and used to correlate the result on the next turn.
                            toolCallID: IdentifierGenerator.generate(prefix: "call"),
                            toolName: name,
                            input: call.args ?? .object([:])
                        )
                    )
                )
            } else if let text = part.text, !text.isEmpty {
                content.append(
                    part.thought == true
                        ? .reasoning(ReasoningPart(text))
                        : .text(TextPart(text))
                )
            } else if let inline = part.inlineData,
                      let mimeType = inline.mimeType,
                      let base64 = inline.data,
                      let data = Data(base64Encoded: base64) {
                content.append(.file(.data(data, mediaType: mimeType)))
            }
        }

        for source in candidates?.first?.citationMetadata?.citationSources ?? [] {
            guard let uri = source.uri, let url = URL(string: uri) else { continue }
            content.append(
                .source(
                    SourcePart(
                        id: IdentifierGenerator.generate(prefix: "source"),
                        kind: .url(url),
                        title: source.title
                    )
                )
            )
        }
        return content
    }

    var hasFunctionCall: Bool {
        candidates?.first?.content?.parts?.contains { $0.functionCall != nil } ?? false
    }

    var normalizedFinishReason: FinishReason {
        FinishReason.fromGoogle(candidates?.first?.finishReason, hasFunctionCall: hasFunctionCall)
    }
}

// MARK: - Provider

/// A provider for Google's Gemini API.
///
/// ```swift
/// let google = GoogleProvider(apiKey: key)
/// let model = google.languageModel("gemini-2.5-flash")
///
/// let result = try await generateText(model: model, prompt: "Explain kinetic energy briefly.")
/// ```
///
/// Gemini-specific settings go through provider options, including thinking budgets and safety
/// thresholds:
///
/// ```swift
/// settings.providerOptions = [
///     "google": [
///         "generationConfig": ["thinkingConfig": ["thinkingBudget": 2048, "includeThoughts": true]],
///         "safetySettings": [["category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"]],
///     ]
/// ]
/// ```
public struct GoogleProvider: AIProvider, Sendable {
    public let name: String

    private let client: ProviderHTTPClient

    /// Creates a provider.
    ///
    /// - Parameters:
    ///   - apiKey: The API key. When `nil`, `GOOGLE_GENERATIVE_AI_API_KEY` is read at request time.
    ///   - baseURL: The API root, including the version path.
    ///   - headers: Extra headers sent with every request.
    ///   - transport: The HTTP transport. Substitute this in tests.
    ///   - name: The provider's short name, used as the options namespace.
    public init(
        apiKey: String? = nil,
        baseURL: URL = URL(string: "https://generativelanguage.googleapis.com/v1beta")!,
        headers: [String: String] = [:],
        transport: any HTTPTransport = URLSessionTransport.shared,
        name: String = "google"
    ) {
        self.name = name
        self.client = ProviderHTTPClient(
            baseURL: baseURL,
            provider: name,
            headers: {
                var merged = headers
                // The header form keeps the key out of URLs, and therefore out of logs.
                merged["x-goog-api-key"] = try ProviderHTTPClient.resolveAPIKey(
                    explicit: apiKey,
                    environmentVariable: "GOOGLE_GENERATIVE_AI_API_KEY",
                    provider: name
                )
                return merged
            },
            transport: transport,
            // Gemini's JSON already uses camel case.
            decoder: ProviderJSON.decoder
        )
    }

    /// Returns a chat model.
    public func languageModel(_ modelID: String) -> any LanguageModel {
        GoogleGenerativeModel(provider: name, modelID: modelID, client: client)
    }

    /// Returns an embedding model.
    public func embeddingModel(_ modelID: String) -> any EmbeddingModel {
        GoogleEmbeddingModel(provider: name, modelID: modelID, client: client)
    }

    /// Returns an evaluation model that answers through structured output.
    ///
    /// Thinking is turned down as far as the model allows, since it rarely helps this kind of
    /// judgment and costs latency. Override with `["google": ["generationConfig": ["thinkingConfig": …]]]`.
    ///
    /// - Important: Experimental. Evaluation may change in a minor release.
    public func evaluationModel(_ modelID: String) -> any EvaluationModel {
        LanguageModelEvaluationModel(
            model: languageModel(modelID),
            provider: name,
            defaultProviderOptions: Self.minimalThinking(for: modelID).map { config in
                [name: ["generationConfig": ["thinkingConfig": config]]]
            }
        )
    }

    /// The least thinking a model accepts, or `nil` when it does not think or the right setting is
    /// unknown.
    ///
    /// Gemini 3 takes a level, and its newer non-lite Flash models no longer accept `minimal`.
    /// Gemini 2.5 takes a budget: zero turns thinking off, except on Pro, whose floor is 128.
    static func minimalThinking(for modelID: String) -> JSONValue? {
        let id = modelID.hasPrefix("models/") ? String(modelID.dropFirst("models/".count)) : modelID
        if id == "gemini-flash-latest" { return ["thinkingLevel": "low"] }

        guard id.hasPrefix("gemini-") else { return nil }
        let version = id.dropFirst("gemini-".count).prefix { $0.isNumber || $0 == "." }
        let components = version.split(separator: ".").compactMap { Int($0) }
        guard let major = components.first else { return nil }
        let minor = components.count > 1 ? components[1] : 0

        if major >= 3 {
            let isFlash = id.contains("-flash") && !id.contains("-lite")
            let needsLow = isFlash && (major > 3 || minor >= 7)
            return ["thinkingLevel": .string(needsLow ? "low" : "minimal")]
        }
        if major == 2, minor == 5 {
            return ["thinkingBudget": .int(id.contains("-pro") ? 128 : 0)]
        }
        return nil
    }
}

/// A language model served by the Gemini API.
public struct GoogleGenerativeModel: LanguageModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient

    /// The model path, normalized so callers may write either `"gemini-2.5-flash"` or the fully
    /// qualified `"models/gemini-2.5-flash"`.
    private var modelPath: String {
        modelID.hasPrefix("models/") ? modelID : "models/\(modelID)"
    }

    public func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        let request = try GoogleRequestBuilder(modelID: modelID).build(options)

        let (response, head, requestBody) = try await client.postJSON(
            path: "\(modelPath):generateContent",
            body: request.body,
            additionalHeaders: options.headers,
            as: GoogleGenerateContentResponse.self
        )

        var warnings = request.warnings
        if let blockReason = response.promptFeedback?.blockReason {
            warnings.append(.other(message: "The prompt was blocked: \(blockReason)."))
        }

        return LanguageModelResponse(
            content: response.modelContent(),
            finishReason: response.normalizedFinishReason,
            usage: response.usageMetadata?.normalized ?? .none,
            warnings: warnings,
            request: RequestInfo(body: requestBody),
            response: ResponseInfo(
                id: response.responseId,
                modelID: response.modelVersion,
                headers: head.headers
            )
        )
    }

    public func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        let request = try GoogleRequestBuilder(modelID: modelID).build(options)

        let (events, head, requestBody) = try await client.postJSONForServerSentEvents(
            path: "\(modelPath):streamGenerateContent",
            body: request.body,
            queryItems: [URLQueryItem(name: "alt", value: "sse")],
            additionalHeaders: options.headers,
            // Gemini simply closes the stream; there is no sentinel payload.
            stopAtDoneSentinel: false
        )

        let warnings = request.warnings
        let includesRawChunks = options.includesRawChunks
        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()

        let task = Task {
            var decoder = GoogleStreamDecoder()
            continuation.yield(.streamStart(warnings: warnings))

            do {
                for try await sseEvent in events {
                    try Task.checkCancellation()
                    guard !sseEvent.data.isEmpty else { continue }

                    if includesRawChunks, let raw = try? JSONValue.parse(sseEvent.data) {
                        continuation.yield(.raw(raw))
                    }

                    let chunk = try ProviderJSON.decode(
                        GoogleGenerateContentResponse.self,
                        from: Data(sseEvent.data.utf8),
                        context: "a streamed chunk",
                        decoder: ProviderJSON.decoder
                    )
                    for part in decoder.consume(chunk) {
                        continuation.yield(part)
                    }
                }
                for part in decoder.finish() {
                    continuation.yield(part)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }

        return LanguageModelStreamResponse(
            stream: stream,
            request: RequestInfo(body: requestBody),
            response: ResponseInfo(headers: head.headers)
        )
    }
}

/// An embedding model served by the Gemini API.
public struct GoogleEmbeddingModel: EmbeddingModel {
    public let provider: String
    public let modelID: String
    public let maxEmbeddingsPerCall: Int? = 100
    public let supportsParallelCalls = true

    let client: ProviderHTTPClient

    private var modelPath: String {
        modelID.hasPrefix("models/") ? modelID : "models/\(modelID)"
    }

    private struct Response: Decodable {
        var embeddings: [Embedding]

        struct Embedding: Decodable {
            var values: [Double]
        }
    }

    public func embed(_ options: EmbeddingModelCallOptions) async throws -> EmbeddingModelResponse {
        let requests = options.values.map { value in
            JSONValue.object([
                "model": .string(modelPath),
                "content": .object(["parts": .array([.object(["text": .string(value)])])]),
            ])
        }

        let (response, head, _) = try await client.postJSON(
            path: "\(modelPath):batchEmbedContents",
            body: JSONValue.object(["requests": .array(requests)]),
            additionalHeaders: options.headers,
            as: Response.self
        )

        return EmbeddingModelResponse(
            embeddings: response.embeddings.map(\.values),
            response: ResponseInfo(headers: head.headers)
        )
    }
}

import AIProviderSpec
import AIProviderUtils
import Foundation

/// A provider for Anthropic's Messages API.
///
/// ```swift
/// let anthropic = AnthropicProvider(apiKey: key)
/// let model = anthropic.languageModel("claude-sonnet-4-5")
///
/// let result = try await generateText(model: model, prompt: "Explain kinetic energy briefly.")
/// ```
///
/// Extended thinking is enabled through provider options:
///
/// ```swift
/// var settings = GenerationSettings(maxOutputTokens: 16_000)
/// settings.providerOptions = [
///     "anthropic": ["thinking": ["type": "enabled", "budget_tokens": 8_000]]
/// ]
/// ```
///
/// Prompt caching is enabled by marking where the reusable prefix ends, on the message or content
/// part that concludes it:
///
/// ```swift
/// .system(longInstructions, providerOptions: ["anthropic": ["cacheControl": ["type": "ephemeral"]]])
/// ```
public struct AnthropicProvider: AIProvider, Sendable {
    public let name: String

    private let client: ProviderHTTPClient
    private let defaultMaxTokens: Int

    /// Creates a provider.
    ///
    /// - Parameters:
    ///   - apiKey: The API key. When `nil`, `ANTHROPIC_API_KEY` is read at request time.
    ///   - baseURL: The API root. Override for a proxy or gateway.
    ///   - apiVersion: The value of the `anthropic-version` header.
    ///   - defaultMaxTokens: Used when a call does not set ``GenerationSettings/maxOutputTokens``.
    ///     The API requires the field, so a default has to exist; this one is generous enough not
    ///     to truncate ordinary answers.
    ///   - headers: Extra headers sent with every request. Use this for beta feature flags.
    ///   - transport: The HTTP transport. Substitute this in tests.
    ///   - name: The provider's short name, used as the options namespace.
    public init(
        apiKey: String? = nil,
        baseURL: URL = URL(string: "https://api.anthropic.com/v1")!,
        apiVersion: String = "2023-06-01",
        defaultMaxTokens: Int = 8192,
        headers: [String: String] = [:],
        transport: any HTTPTransport = URLSessionTransport.shared,
        name: String = "anthropic"
    ) {
        self.name = name
        self.defaultMaxTokens = defaultMaxTokens
        self.client = ProviderHTTPClient(
            baseURL: baseURL,
            provider: name,
            headers: {
                var merged = headers
                merged["x-api-key"] = try ProviderHTTPClient.resolveAPIKey(
                    explicit: apiKey,
                    environmentVariable: "ANTHROPIC_API_KEY",
                    provider: name
                )
                merged["anthropic-version"] = apiVersion
                return merged
            },
            transport: transport,
            decoder: ProviderJSON.snakeCaseDecoder
        )
    }

    /// Returns a chat model.
    public func languageModel(_ modelID: String) -> any LanguageModel {
        AnthropicMessagesModel(
            provider: name,
            modelID: modelID,
            client: client,
            defaultMaxTokens: defaultMaxTokens
        )
    }

    /// Returns an evaluation model that answers through structured output.
    ///
    /// No thinking configuration is sent, which leaves thinking off on every model that allows
    /// it. Sending `thinking: {type: "disabled"}` explicitly would fail on models that always
    /// think adaptively. Pass `["anthropic": ["thinking": …]]` to enable it for harder
    /// evaluations.
    ///
    /// - Important: Experimental. Evaluation may change in a minor release.
    public func evaluationModel(_ modelID: String) -> any EvaluationModel {
        LanguageModelEvaluationModel(model: languageModel(modelID), provider: name)
    }
}

/// A language model served by the Messages API.
public struct AnthropicMessagesModel: LanguageModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient
    let defaultMaxTokens: Int

    /// Anthropic fetches image and document URLs itself, so there is no need to download and
    /// re-encode them.
    public func supportsNativeURL(_ url: URL, mediaType: String) -> Bool {
        url.scheme == "https" && (mediaType.hasPrefix("image/") || mediaType == "application/pdf")
    }

    public func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        let request = try AnthropicRequestBuilder(modelID: modelID, defaultMaxTokens: defaultMaxTokens)
            .build(options, stream: false)

        let (message, head, requestBody) = try await client.postJSON(
            path: "messages",
            body: request.body,
            additionalHeaders: options.headers,
            as: AnthropicMessage.self
        )

        return LanguageModelResponse(
            content: message.modelContent(),
            finishReason: message.normalizedFinishReason,
            usage: message.usage?.normalized ?? .none,
            warnings: request.warnings,
            providerMetadata: message.providerMetadata,
            request: RequestInfo(body: requestBody),
            response: ResponseInfo(id: message.id, modelID: message.model, headers: head.headers)
        )
    }

    public func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        let request = try AnthropicRequestBuilder(modelID: modelID, defaultMaxTokens: defaultMaxTokens)
            .build(options, stream: true)

        let (events, head, requestBody) = try await client.postJSONForServerSentEvents(
            path: "messages",
            body: request.body,
            additionalHeaders: options.headers,
            // Anthropic ends its stream with `message_stop`, not a sentinel payload.
            stopAtDoneSentinel: false
        )

        let warnings = request.warnings
        let includesRawChunks = options.includesRawChunks
        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()

        let task = Task {
            var decoder = AnthropicStreamDecoder()
            continuation.yield(.streamStart(warnings: warnings))

            do {
                for try await sseEvent in events {
                    try Task.checkCancellation()
                    guard !sseEvent.data.isEmpty else { continue }

                    if includesRawChunks, let raw = try? JSONValue.parse(sseEvent.data) {
                        continuation.yield(.raw(raw))
                    }

                    let event = try ProviderJSON.decode(
                        AnthropicStreamEvent.self,
                        from: Data(sseEvent.data.utf8),
                        context: "a streamed event",
                        decoder: ProviderJSON.snakeCaseDecoder
                    )
                    for part in try decoder.consume(event) {
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

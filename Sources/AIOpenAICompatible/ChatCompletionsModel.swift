import AIProviderSpec
import AIProviderUtils
import Foundation

/// A language model served by the Chat Completions API.
///
/// The endpoint is supported by OpenAI and by essentially every self-hosted or third-party server
/// that advertises OpenAI compatibility — Ollama, vLLM, Groq, Together, LM Studio, OpenRouter and
/// others. Differences between them are handled by ``OpenAICompatibleQuirks`` rather than by
/// separate implementations.
public struct ChatCompletionsModel: LanguageModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient
    let quirks: OpenAICompatibleQuirks

    init(provider: String, modelID: String, client: ProviderHTTPClient, quirks: OpenAICompatibleQuirks) {
        self.provider = provider
        self.modelID = modelID
        self.client = client
        self.quirks = quirks
    }

    /// Whether the server fetches image URLs itself.
    ///
    /// OpenAI does, which avoids downloading and base64-encoding an image the API could have
    /// fetched directly. Self-hosted servers frequently do not, so this is configurable.
    public func supportsNativeURL(_ url: URL, mediaType: String) -> Bool {
        quirks.fetchesImageURLs && mediaType.hasPrefix("image/")
    }

    // MARK: - Generate

    public func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        let request = try ChatCompletionsRequestBuilder(modelID: modelID, quirks: quirks)
            .build(options, stream: false)

        let (response, head, requestBody) = try await client.postJSON(
            path: quirks.chatCompletionsPath,
            body: request.body,
            additionalHeaders: options.headers,
            as: ChatCompletionResponse.self
        )

        var responseInfo = response.responseInfo
        responseInfo.headers = head.headers

        return LanguageModelResponse(
            content: try response.modelContent(),
            finishReason: response.normalizedFinishReason,
            usage: response.usage?.normalized ?? .none,
            warnings: request.warnings,
            request: RequestInfo(body: requestBody),
            response: responseInfo
        )
    }

    // MARK: - Stream

    public func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        let request = try ChatCompletionsRequestBuilder(modelID: modelID, quirks: quirks)
            .build(options, stream: true)

        let (events, head, requestBody) = try await client.postJSONForServerSentEvents(
            path: quirks.chatCompletionsPath,
            body: request.body,
            additionalHeaders: options.headers
        )

        let warnings = request.warnings
        let includesRawChunks = options.includesRawChunks
        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()

        let task = Task {
            var decoder = ChatCompletionsStreamDecoder()
            continuation.yield(.streamStart(warnings: warnings))

            do {
                for try await event in events {
                    try Task.checkCancellation()
                    guard !event.data.isEmpty else { continue }

                    if includesRawChunks, let raw = try? JSONValue.parse(event.data) {
                        continuation.yield(.raw(raw))
                    }

                    let chunk = try ProviderJSON.decode(
                        ChatCompletionChunk.self,
                        from: Data(event.data.utf8),
                        context: "a streamed chunk",
                        decoder: ProviderJSON.snakeCaseDecoder
                    )
                    for part in try decoder.consume(chunk) {
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

/// An embedding model served by the OpenAI-compatible embeddings endpoint.
public struct OpenAICompatibleEmbeddingModel: EmbeddingModel {
    public let provider: String
    public let modelID: String
    public let maxEmbeddingsPerCall: Int?
    public let supportsParallelCalls: Bool

    let client: ProviderHTTPClient
    let path: String

    init(
        provider: String,
        modelID: String,
        client: ProviderHTTPClient,
        path: String,
        maxEmbeddingsPerCall: Int?,
        supportsParallelCalls: Bool
    ) {
        self.provider = provider
        self.modelID = modelID
        self.client = client
        self.path = path
        self.maxEmbeddingsPerCall = maxEmbeddingsPerCall
        self.supportsParallelCalls = supportsParallelCalls
    }

    public func embed(_ options: EmbeddingModelCallOptions) async throws -> EmbeddingModelResponse {
        var fields: [String: JSONValue] = [
            "model": .string(modelID),
            "input": .array(options.values.map(JSONValue.string)),
        ]
        for (key, value) in options.providerOptions?[provider] ?? [:] {
            fields[key] = value
        }

        let (response, head, _) = try await client.postJSON(
            path: path,
            body: JSONValue.object(fields),
            additionalHeaders: options.headers,
            as: EmbeddingsResponse.self
        )

        // The API documents that results come back in request order, but it also returns an
        // explicit index; sorting by it costs nothing and removes the assumption.
        let ordered = response.data.sorted { ($0.index ?? 0) < ($1.index ?? 0) }
        return EmbeddingModelResponse(
            embeddings: ordered.map(\.embedding),
            usage: Usage(
                inputTokens: response.usage?.promptTokens,
                totalTokens: response.usage?.totalTokens
            ),
            response: ResponseInfo(modelID: response.model, headers: head.headers)
        )
    }
}

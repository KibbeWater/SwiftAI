import AIProviderSpec
import AIProviderUtils
import Foundation

/// A provider for OpenRouter, which routes one API to hundreds of models across many upstreams.
///
/// ```swift
/// let openrouter = OpenRouterProvider(appName: "Weather Bot", appURL: URL(string: "https://example.com"))
/// let result = try await generateText(
///     model: openrouter.languageModel("anthropic/claude-haiku-4.5"),
///     prompt: "Explain kinetic energy in one sentence."
/// )
/// print(result.providerMetadata?["openrouter"]?["cost"] ?? "")
/// ```
///
/// OpenRouter's request options are passed under the `"openrouter"` namespace with their
/// documented wire names, and merged into the request body verbatim:
///
/// ```swift
/// settings.providerOptions = ["openrouter": [
///     "models": ["anthropic/claude-haiku-4.5", "openai/gpt-5-mini"],  // Fallbacks, in order.
///     "provider": ["sort": "throughput", "data_collection": "deny"],   // Routing preferences.
///     "reasoning": ["effort": "low"],
/// ]]
/// ```
///
/// Reasoning is replayed across turns automatically, provided the conversation is continued with
/// the result's `responseMessages`: OpenRouter's `reasoning_details` travel on the reasoning parts
/// and tool calls those messages contain.
public struct OpenRouterProvider: AIProvider, Sendable {
    public let name: String

    let client: ProviderHTTPClient
    let decisionsClient: ProviderHTTPClient?
    let extraBody: [String: JSONValue]

    /// Creates a provider.
    ///
    /// - Parameters:
    ///   - apiKey: The API key. When `nil`, `OPENROUTER_API_KEY` is read at request time.
    ///   - baseURL: The API root.
    ///   - decisionsBaseURL: The root of the Decisions API used by ``decisionModel(_:)``. Derived
    ///     from `baseURL` by replacing a trailing `/v1` with `/alpha` when omitted.
    ///   - appName: Your application's name, shown in OpenRouter's rankings and activity log.
    ///   - appURL: Your application's URL, used for the same attribution.
    ///   - providerAPIKeys: Your own keys for upstream providers, keyed by provider slug such as
    ///     `"anthropic"`. Requests routed to those providers bill your key rather than OpenRouter
    ///     credits.
    ///   - headers: Extra headers sent with every request.
    ///   - extraBody: Fields merged into every request body, beneath per-call provider options.
    ///   - transport: The HTTP transport. Substitute this in tests.
    ///   - name: The provider's short name, used as the options and metadata namespace.
    public init(
        apiKey: String? = nil,
        baseURL: URL = URL(string: "https://openrouter.ai/api/v1")!,
        decisionsBaseURL: URL? = nil,
        appName: String? = nil,
        appURL: URL? = nil,
        providerAPIKeys: [String: String] = [:],
        headers: [String: String] = [:],
        extraBody: [String: JSONValue] = [:],
        transport: any HTTPTransport = URLSessionTransport.shared,
        name: String = "openrouter"
    ) {
        self.name = name
        self.extraBody = extraBody

        let resolveHeaders: @Sendable () async throws -> [String: String] = {
            var merged: [String: String] = [:]
            if let appName { merged["X-OpenRouter-Title"] = appName }
            if let appURL { merged["HTTP-Referer"] = appURL.absoluteString }
            // Caller headers override the attribution headers, but not credentials.
            for (key, value) in headers { merged[key] = value }
            merged["Authorization"] = "Bearer " + (try ProviderHTTPClient.resolveAPIKey(
                explicit: apiKey,
                environmentVariable: "OPENROUTER_API_KEY",
                provider: name
            ))
            if !providerAPIKeys.isEmpty {
                merged["X-Provider-API-Keys"] = JSONValue.object(providerAPIKeys.mapValues(JSONValue.string)).serialized(sortedKeys: true)
            }
            return merged
        }

        func client(_ baseURL: URL) -> ProviderHTTPClient {
            ProviderHTTPClient(
                baseURL: baseURL,
                provider: name,
                headers: resolveHeaders,
                transport: transport,
                errorMapper: OpenRouterResponse.errorMapper,
                // Not the snake-case decoder: responses are read as raw JSON, and their nested
                // keys are replayed verbatim.
                decoder: ProviderJSON.decoder
            )
        }

        self.client = client(baseURL)
        self.decisionsClient = (decisionsBaseURL ?? Self.deriveDecisionsBaseURL(from: baseURL)).map(client)
    }

    /// Replaces a trailing `/v1` with `/alpha`, where OpenRouter serves its Decisions API.
    static func deriveDecisionsBaseURL(from baseURL: URL) -> URL? {
        var string = baseURL.absoluteString
        while string.hasSuffix("/") { string.removeLast() }
        guard string.hasSuffix("/v1") else { return nil }
        return URL(string: String(string.dropLast(3)) + "/alpha")
    }

    /// Returns a chat model. `openrouter/auto` lets OpenRouter pick the model per request.
    public func languageModel(_ modelID: String) -> any LanguageModel {
        OpenRouterChatModel(provider: name, modelID: modelID, client: client, extraBody: extraBody)
    }

    /// Returns an embedding model.
    public func embeddingModel(_ modelID: String) -> any EmbeddingModel {
        OpenRouterEmbeddingModel(provider: name, modelID: modelID, client: client, extraBody: extraBody)
    }

    /// Returns an image generation model, served by OpenRouter's images endpoint.
    public func imageModel(_ modelID: String) -> any ImageModel {
        OpenRouterImageModel(provider: name, modelID: modelID, client: client, extraBody: extraBody)
    }

    /// Returns a decision model, served by OpenRouter's Decisions API.
    ///
    /// Decision models answer choice, score, and boolean questions natively, with probability
    /// distributions and a per-question confidence statistic. Use them with `evaluate`:
    ///
    /// ```swift
    /// let result = try await evaluate(
    ///     model: openrouter.decisionModel("typesafe/jev-1.13"),
    ///     state: "I was charged twice.",
    ///     questions: ["refund": .boolean("Is the customer asking for money back?")]
    /// )
    /// ```
    ///
    /// - Important: Experimental. The Decisions API is in alpha, and evaluation may change in a
    ///   minor release.
    public func decisionModel(_ modelID: String) -> any EvaluationModel {
        OpenRouterDecisionModel(provider: name, modelID: modelID, client: decisionsClient, extraBody: extraBody)
    }

    /// The same model as ``decisionModel(_:)``, under the name ``AIProvider`` uses, so registries
    /// resolve OpenRouter decision models like any other evaluation model.
    ///
    /// - Important: Experimental. Evaluation may change in a minor release.
    public func evaluationModel(_ modelID: String) -> any EvaluationModel {
        decisionModel(modelID)
    }
}

/// OpenRouter's server-side tools.
///
/// ```swift
/// let result = try await generateText(
///     model: openrouter.languageModel("openai/gpt-5-mini"),
///     prompt: "What changed in the latest Swift release?",
///     tools: [OpenRouterTools.webSearch(maxResults: 3)]
/// )
/// ```
public enum OpenRouterTools {
    /// Searches the web on OpenRouter's side. Results come back as sources on the response.
    ///
    /// - Parameters:
    ///   - maxResults: How many results to retrieve.
    ///   - searchPrompt: Instructions for incorporating the results.
    ///   - engine: `"native"` for the model's own search, `"exa"`, or `"auto"`.
    public static func webSearch(
        maxResults: Int? = nil,
        searchPrompt: String? = nil,
        engine: String? = nil
    ) -> LanguageModelTool {
        var arguments: [String: JSONValue] = [:]
        if let maxResults { arguments["maxResults"] = .int(maxResults) }
        if let searchPrompt { arguments["searchPrompt"] = .string(searchPrompt) }
        if let engine { arguments["engine"] = .string(engine) }
        return .providerDefined(ProviderDefinedTool(id: "openrouter.web_search", name: "web_search", arguments: arguments))
    }
}

// MARK: - Chat

/// A language model served by OpenRouter's chat completions endpoint.
public struct OpenRouterChatModel: LanguageModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient
    let extraBody: [String: JSONValue]

    /// OpenRouter fetches image URLs with a recognizable image extension, and any document URL.
    /// Everything else is downloaded by the core and sent inline.
    public func supportsNativeURL(_ url: URL, mediaType: String) -> Bool {
        guard url.scheme == "https" || url.scheme == "http" else { return false }
        if mediaType.hasPrefix("image/") {
            return ["jpg", "jpeg", "png", "gif", "webp"].contains(url.pathExtension.lowercased())
        }
        return mediaType.hasPrefix("application/")
    }

    private var builder: OpenRouterChatRequestBuilder {
        OpenRouterChatRequestBuilder(modelID: modelID, providerName: provider, extraBody: extraBody)
    }

    public func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        let request = try builder.build(options, stream: false)
        let (body, head, requestBody) = try await client.postJSON(
            path: "chat/completions",
            body: request.body,
            additionalHeaders: options.headers,
            as: JSONValue.self
        )
        try OpenRouterResponse.throwIfError(body, url: try client.url(path: "chat/completions"), headers: head.headers)

        guard let choice = body["choices"]?[0], let message = choice["message"] else {
            throw InvalidResponseDataError(message: "OpenRouter returned no choices.", data: body)
        }
        let content = try OpenRouterResponse.content(of: message, providerName: provider)
        let hasToolCalls = content.contains { if case .toolCall = $0 { true } else { false } }
        let details = ReasoningDetails.entries(message["reasoning_details"])

        return LanguageModelResponse(
            content: content,
            finishReason: OpenRouterResponse.finishReason(choice["finish_reason"]?.stringValue, hasToolCalls: hasToolCalls),
            usage: OpenRouterResponse.usage(body["usage"]),
            warnings: request.warnings,
            providerMetadata: OpenRouterResponse.metadata(
                provider: body["provider"]?.stringValue,
                usage: body["usage"],
                reasoningDetails: details.isEmpty ? nil : details,
                fileAnnotations: OpenRouterResponse.fileAnnotations(message["annotations"]),
                providerName: provider
            ),
            request: RequestInfo(body: requestBody),
            response: ResponseInfo(
                id: body["id"]?.stringValue,
                modelID: body["model"]?.stringValue,
                timestamp: body["created"]?.numberValue.map { Date(timeIntervalSince1970: $0) },
                headers: head.headers
            )
        )
    }

    public func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        let request = try builder.build(options, stream: true)
        let (events, head, requestBody) = try await client.postJSONForServerSentEvents(
            path: "chat/completions",
            body: request.body,
            additionalHeaders: options.headers
        )
        let url = try client.url(path: "chat/completions")

        let warnings = request.warnings
        let includesRawChunks = options.includesRawChunks
        let providerName = provider
        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()

        let task = Task {
            var decoder = OpenRouterStreamDecoder(providerName: providerName)
            continuation.yield(.streamStart(warnings: warnings))
            do {
                for try await event in events {
                    try Task.checkCancellation()
                    guard !event.data.isEmpty else { continue }

                    let chunk: JSONValue
                    do {
                        chunk = try JSONValue.parse(event.data)
                    } catch {
                        throw InvalidResponseDataError(message: "OpenRouter sent a chunk that is not valid JSON.", data: .string(event.data))
                    }
                    if includesRawChunks { continuation.yield(.raw(chunk)) }
                    // An upstream failure after the stream has begun arrives as a chunk, since
                    // the status line has already been sent.
                    try OpenRouterResponse.throwIfError(chunk, url: url, headers: head.headers)

                    for part in try decoder.consume(chunk) { continuation.yield(part) }
                }
                for part in decoder.finish() { continuation.yield(part) }
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

// MARK: - Embeddings

/// An embedding model served by OpenRouter's embeddings endpoint.
public struct OpenRouterEmbeddingModel: EmbeddingModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient
    let extraBody: [String: JSONValue]

    public func embed(_ options: EmbeddingModelCallOptions) async throws -> EmbeddingModelResponse {
        var fields = extraBody
        for (key, value) in options.providerOptions?[provider] ?? [:] { fields[key] = value }
        fields["model"] = .string(modelID)
        fields["input"] = .array(options.values.map(JSONValue.string))

        let (body, head, _) = try await client.postJSON(
            path: "embeddings",
            body: JSONValue.object(fields),
            additionalHeaders: options.headers,
            as: JSONValue.self
        )
        try OpenRouterResponse.throwIfError(body, url: try client.url(path: "embeddings"), headers: head.headers)

        // Sorted by the explicit index rather than trusting the documented order.
        let items = (body["data"]?.arrayValue ?? []).sorted {
            ($0["index"]?.intValue ?? 0) < ($1["index"]?.intValue ?? 0)
        }
        let embeddings = try items.map { item -> Embedding in
            guard let vector = item["embedding"]?.arrayValue?.compactMap(\.numberValue) else {
                throw InvalidResponseDataError(message: "OpenRouter returned an embedding that is not a list of numbers.", data: item)
            }
            return vector
        }

        return EmbeddingModelResponse(
            embeddings: embeddings,
            usage: Usage(inputTokens: body["usage"]?["prompt_tokens"]?.intValue, totalTokens: body["usage"]?["total_tokens"]?.intValue),
            providerMetadata: OpenRouterResponse.metadata(
                provider: body["provider"]?.stringValue,
                usage: body["usage"],
                reasoningDetails: nil,
                fileAnnotations: [],
                providerName: provider
            ),
            response: ResponseInfo(modelID: body["model"]?.stringValue, headers: head.headers)
        )
    }
}

// MARK: - Images

/// An image model served by OpenRouter's images endpoint.
public struct OpenRouterImageModel: ImageModel {
    public let provider: String
    public let modelID: String
    public let maxImagesPerCall: Int? = 10

    let client: ProviderHTTPClient
    let extraBody: [String: JSONValue]

    public func generate(_ options: ImageModelCallOptions) async throws -> ImageModelResponse {
        var fields = extraBody
        fields["model"] = .string(modelID)
        fields["prompt"] = .string(options.prompt)
        if options.count != 1 { fields["n"] = .int(options.count) }
        if let size = options.size { fields["size"] = .string(size.stringValue) }
        if let aspectRatio = options.aspectRatio { fields["aspect_ratio"] = .string(aspectRatio.stringValue) }
        if let seed = options.seed { fields["seed"] = .int(seed) }
        for (key, value) in options.providerOptions?[provider] ?? [:] { fields[key] = value }

        let (body, head, _) = try await client.postJSON(
            path: "images",
            body: JSONValue.object(fields),
            additionalHeaders: options.headers,
            as: JSONValue.self
        )
        try OpenRouterResponse.throwIfError(body, url: try client.url(path: "images"), headers: head.headers)

        let images = (body["data"]?.arrayValue ?? []).compactMap { item -> GeneratedImage? in
            guard let base64 = item["b64_json"]?.stringValue, let data = Data(base64Encoded: base64) else { return nil }
            return GeneratedImage(data: data, mediaType: item["media_type"]?.stringValue ?? "image/png")
        }
        guard !images.isEmpty else {
            throw InvalidResponseDataError(message: "OpenRouter returned no images.", data: body)
        }

        return ImageModelResponse(
            images: images,
            providerMetadata: OpenRouterResponse.metadata(
                provider: body["provider"]?.stringValue,
                usage: body["usage"],
                reasoningDetails: nil,
                fileAnnotations: [],
                providerName: provider
            ),
            response: ResponseInfo(
                modelID: body["model"]?.stringValue ?? modelID,
                timestamp: body["created"]?.numberValue.flatMap { $0 > 0 ? Date(timeIntervalSince1970: $0) : nil },
                headers: head.headers
            )
        )
    }
}

import AIOpenAICompatible
import AIProviderSpec
import AIProviderUtils
import Foundation

/// A provider for OpenAI's API.
///
/// ```swift
/// let openai = OpenAIProvider(apiKey: key)
/// let model = openai.languageModel("gpt-5")
///
/// let result = try await generateText(model: model, prompt: "Explain kinetic energy briefly.")
/// ```
///
/// ``languageModel(_:)`` uses the Responses API, which is where reasoning items and
/// provider-executed tools live. ``chatModel(_:)`` uses Chat Completions for the settings only it
/// supports — `frequency_penalty`, `seed`, `stop`.
///
/// Reasoning effort is set through provider options:
///
/// ```swift
/// settings.providerOptions = ["openai": ["reasoning": ["effort": "high", "summary": "auto"]]]
/// ```
public struct OpenAIProvider: AIProvider, Sendable {
    public let name: String

    private let client: ProviderHTTPClient

    /// Creates a provider.
    ///
    /// - Parameters:
    ///   - apiKey: The API key. When `nil`, `OPENAI_API_KEY` is read at request time.
    ///   - organization: An organization identifier, for accounts that belong to several.
    ///   - project: A project identifier, for attributing usage.
    ///   - baseURL: The API root. Override for Azure or a gateway.
    ///   - headers: Extra headers sent with every request.
    ///   - transport: The HTTP transport. Substitute this in tests.
    ///   - name: The provider's short name, used as the options namespace.
    public init(
        apiKey: String? = nil,
        organization: String? = nil,
        project: String? = nil,
        baseURL: URL = URL(string: "https://api.openai.com/v1")!,
        headers: [String: String] = [:],
        transport: any HTTPTransport = URLSessionTransport.shared,
        name: String = "openai"
    ) {
        self.name = name
        self.client = ProviderHTTPClient(
            baseURL: baseURL,
            provider: name,
            headers: {
                var merged = headers
                merged["Authorization"] = "Bearer " + (try ProviderHTTPClient.resolveAPIKey(
                    explicit: apiKey,
                    environmentVariable: "OPENAI_API_KEY",
                    provider: name
                ))
                if let organization { merged["OpenAI-Organization"] = organization }
                if let project { merged["OpenAI-Project"] = project }
                return merged
            },
            transport: transport,
            decoder: ProviderJSON.snakeCaseDecoder
        )
    }

    /// Returns a model served by the Responses API.
    public func languageModel(_ modelID: String) -> any LanguageModel {
        OpenAIResponsesModel(provider: name, modelID: modelID, client: client)
    }

    /// Returns a model served by the Chat Completions API.
    ///
    /// Use this for the settings the Responses API does not expose — frequency and presence
    /// penalties, seeds, and stop sequences — or to talk to a deployment that only offers the
    /// older endpoint.
    public func chatModel(_ modelID: String) -> any LanguageModel {
        OpenAICompatibleProvider(name: name, client: client, quirks: .openAI).languageModel(modelID)
    }

    /// Returns an embedding model.
    public func embeddingModel(_ modelID: String) -> any EmbeddingModel {
        OpenAICompatibleProvider(name: name, client: client, quirks: .openAI)
            .embeddingModel(modelID, maxEmbeddingsPerCall: 2048)
    }

    /// Returns an image generation model.
    public func imageModel(_ modelID: String) -> any ImageModel {
        OpenAIImageModel(provider: name, modelID: modelID, client: client)
    }

    /// Returns a speech synthesis model.
    public func speechModel(_ modelID: String) -> any SpeechModel {
        OpenAISpeechModel(provider: name, modelID: modelID, client: client)
    }

    /// Returns a transcription model.
    public func transcriptionModel(_ modelID: String) -> any TranscriptionModel {
        OpenAITranscriptionModel(provider: name, modelID: modelID, client: client)
    }
}

// MARK: - Images

/// An image model served by the images endpoint.
public struct OpenAIImageModel: ImageModel {
    public let provider: String
    public let modelID: String

    /// The documented per-request maximum. `gpt-image-1` allows fewer for some sizes, and reports
    /// that itself.
    public let maxImagesPerCall: Int? = 10

    let client: ProviderHTTPClient

    private struct Response: Decodable {
        var data: [Item]?

        struct Item: Decodable {
            var b64Json: String?
            var url: String?
            var revisedPrompt: String?
        }
    }

    public func generate(_ options: ImageModelCallOptions) async throws -> ImageModelResponse {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = [
            "model": .string(modelID),
            "prompt": .string(options.prompt),
            "n": .int(options.count),
        ]
        if let size = options.size {
            fields["size"] = .string(size.stringValue)
        }
        if options.aspectRatio != nil {
            warnings.append(
                .unsupportedSetting(
                    setting: "aspectRatio",
                    details: "OpenAI takes explicit dimensions; use 'size' instead."
                )
            )
        }
        if options.seed != nil {
            warnings.append(
                .unsupportedSetting(setting: "seed", details: "OpenAI image models are not seedable.")
            )
        }
        // `gpt-image-1` always returns base64; the DALL·E models default to a URL that expires.
        if modelID.hasPrefix("dall-e") {
            fields["response_format"] = .string("b64_json")
        }
        for (key, value) in options.providerOptions?[provider] ?? [:] {
            fields[key] = value
        }

        let (response, head, _) = try await client.postJSON(
            path: "images/generations",
            body: JSONValue.object(fields),
            additionalHeaders: options.headers,
            as: Response.self
        )

        let images = (response.data ?? []).compactMap { item -> GeneratedImage? in
            guard let base64 = item.b64Json, let data = Data(base64Encoded: base64) else { return nil }
            return GeneratedImage(data: data, mediaType: "image/png")
        }
        return ImageModelResponse(
            images: images,
            warnings: warnings,
            response: ResponseInfo(headers: head.headers)
        )
    }
}

// MARK: - Speech

/// A speech model served by the audio endpoint.
public struct OpenAISpeechModel: SpeechModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient

    public func generate(_ options: SpeechModelCallOptions) async throws -> SpeechModelResponse {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = [
            "model": .string(modelID),
            "input": .string(options.text),
            "voice": .string(options.voice ?? "alloy"),
        ]
        let format = options.outputFormat ?? "mp3"
        fields["response_format"] = .string(format)
        if let speed = options.speed { fields["speed"] = .double(speed) }
        if let instructions = options.instructions { fields["instructions"] = .string(instructions) }
        if options.language != nil {
            warnings.append(
                .unsupportedSetting(
                    setting: "language",
                    details: "OpenAI infers the language from the text."
                )
            )
        }
        for (key, value) in options.providerOptions?[provider] ?? [:] {
            fields[key] = value
        }

        // The response is audio, not JSON.
        let (data, head, _) = try await client.postForData(
            path: "audio/speech",
            body: JSONValue.object(fields),
            additionalHeaders: options.headers
        )

        return SpeechModelResponse(
            audio: GeneratedAudio(data: data, mediaType: Self.mediaType(for: format)),
            warnings: warnings,
            response: ResponseInfo(headers: head.headers)
        )
    }

    private static func mediaType(for format: String) -> String {
        switch format {
        case "mp3": return "audio/mpeg"
        case "opus": return "audio/opus"
        case "aac": return "audio/aac"
        case "flac": return "audio/flac"
        case "wav": return "audio/wav"
        case "pcm": return "audio/L16"
        default: return "audio/\(format)"
        }
    }
}

// MARK: - Transcription

/// A transcription model served by the audio endpoint.
public struct OpenAITranscriptionModel: TranscriptionModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient

    private struct Response: Decodable {
        var text: String
        var language: String?
        var duration: Double?
        var segments: [Segment]?

        struct Segment: Decodable {
            var text: String?
            var start: Double?
            var end: Double?
        }
    }

    public func transcribe(
        _ options: TranscriptionModelCallOptions
    ) async throws -> TranscriptionModelResponse {
        var form = MultipartFormData()
        form.addFile(
            name: "file",
            filename: options.filename ?? suggestedAudioFilename(for: options.mediaType),
            mediaType: options.mediaType,
            data: options.audio
        )
        form.addField(name: "model", value: modelID)

        // Only the Whisper models return timed segments; the newer ones reject `verbose_json`.
        var warnings: [CallWarning] = []
        let wantsSegments = modelID.contains("whisper")
        form.addField(name: "response_format", value: wantsSegments ? "verbose_json" : "json")
        if !wantsSegments {
            warnings.append(
                .other(message: "'\(modelID)' does not return timed segments; only the transcript is available.")
            )
        }

        for (key, value) in options.providerOptions?[provider] ?? [:] {
            guard let text = value.stringValue ?? value.numberValue.map({ String($0) }) else { continue }
            form.addField(name: key, value: text)
        }

        let (response, head) = try await client.post(
            path: "audio/transcriptions",
            bodyData: form.encoded(),
            contentType: form.contentType,
            additionalHeaders: options.headers,
            as: Response.self
        )

        return TranscriptionModelResponse(
            text: response.text,
            segments: (response.segments ?? []).compactMap { segment in
                guard let text = segment.text, let start = segment.start, let end = segment.end
                else { return nil }
                return TranscriptionSegment(text: text, startSecond: start, endSecond: end)
            },
            language: response.language,
            durationInSeconds: response.duration,
            warnings: warnings,
            response: ResponseInfo(headers: head.headers)
        )
    }
}

// MARK: - Provider-executed tools

extension OpenAIProvider {
    /// Tools OpenAI runs on its own side.
    ///
    /// These are declared like any other tool but never execute locally: the provider performs the
    /// work and returns the results inline.
    ///
    /// ```swift
    /// let result = try await generateText(
    ///     model: openai.languageModel("gpt-5"),
    ///     prompt: "What happened in Swedish politics this week?",
    ///     tools: [OpenAIProvider.Tools.webSearch()]
    /// )
    /// ```
    public enum Tools {
        /// Lets the model search the web and cite what it finds.
        ///
        /// - Parameters:
        ///   - searchContextSize: How much context to retrieve: `"low"`, `"medium"`, or `"high"`.
        ///   - userLocation: An approximate location to bias results, as the API documents it.
        public static func webSearch(
            searchContextSize: String? = nil,
            userLocation: JSONValue? = nil
        ) -> ProviderDefinedTool {
            var arguments: [String: JSONValue] = [:]
            if let searchContextSize { arguments["search_context_size"] = .string(searchContextSize) }
            if let userLocation { arguments["user_location"] = userLocation }
            return ProviderDefinedTool(id: "openai.web_search", name: "web_search", arguments: arguments)
        }

        /// Lets the model run Python in a sandbox.
        public static func codeInterpreter(container: JSONValue = .object(["type": "auto"])) -> ProviderDefinedTool {
            ProviderDefinedTool(
                id: "openai.code_interpreter",
                name: "code_interpreter",
                arguments: ["container": container]
            )
        }
    }
}

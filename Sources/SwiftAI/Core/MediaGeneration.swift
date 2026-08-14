import AIProviderSpec
import AIProviderUtils
import Foundation

// MARK: - Images

/// The result of generating images.
public struct GenerateImageResult: Sendable {
    /// Every image produced, in order.
    public var images: [GeneratedImage]

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about each response, one per request.
    public var responses: [ResponseInfo]

    /// The first image.
    ///
    /// Most requests ask for one, so this saves reaching into the array.
    public var image: GeneratedImage? { images.first }
}

/// Generates images from a prompt.
///
/// ```swift
/// let result = try await generateImage(
///     model: openai.imageModel("gpt-image-1"),
///     prompt: "A watercolour of Malmö in autumn.",
///     size: .square1024
/// )
/// ```
///
/// Providers cap how many images one request may produce; asking for more issues several requests
/// concurrently and combines the results.
///
/// - Parameters:
///   - model: The image model.
///   - prompt: What to depict.
///   - count: How many images to produce.
///   - size: Exact dimensions. Mutually exclusive with `aspectRatio` on most providers.
///   - aspectRatio: The shape to produce, when the provider chooses dimensions itself.
///   - seed: A seed for reproducible generation, where supported.
///   - retryPolicy: How transient failures are retried.
///   - headers: Extra HTTP headers.
///   - providerOptions: Provider-specific settings, such as quality or style.
/// - Returns: The images, with any warnings.
public func generateImage(
    model: any ImageModel,
    prompt: String,
    count: Int = 1,
    size: ImageSize? = nil,
    aspectRatio: AspectRatio? = nil,
    seed: Int? = nil,
    retryPolicy: RetryPolicy = .default,
    headers: [String: String] = [:],
    providerOptions: ProviderOptions? = nil
) async throws -> GenerateImageResult {
    guard count > 0 else {
        throw InvalidArgumentError(argument: "count", message: "At least one image must be requested.")
    }

    let perCall = model.maxImagesPerCall ?? count
    let batches = stride(from: 0, to: count, by: max(1, perCall)).map { start in
        min(max(1, perCall), count - start)
    }

    let responses = try await withThrowingTaskGroup(
        of: (Int, ImageModelResponse).self,
        returning: [(Int, ImageModelResponse)].self
    ) { group in
        for (index, batchSize) in batches.enumerated() {
            group.addTask {
                let response = try await withRetries(policy: retryPolicy) { _ in
                    try await model.generate(
                        ImageModelCallOptions(
                            prompt: prompt,
                            count: batchSize,
                            size: size,
                            aspectRatio: aspectRatio,
                            // Reusing one seed across batches would produce identical images, so
                            // each batch is offset.
                            seed: seed.map { $0 + index },
                            headers: headers,
                            providerOptions: providerOptions
                        )
                    )
                }
                return (index, response)
            }
        }
        var collected: [(Int, ImageModelResponse)] = []
        for try await element in group { collected.append(element) }
        return collected
    }

    let ordered = responses.sorted { $0.0 < $1.0 }.map(\.1)
    return GenerateImageResult(
        images: ordered.flatMap(\.images),
        warnings: ordered.flatMap(\.warnings),
        providerMetadata: ordered.compactMap(\.providerMetadata).reduce(nil) { ProviderMetadata.merging($0, $1) },
        responses: ordered.compactMap(\.response)
    )
}

// MARK: - Speech

/// The result of synthesizing speech.
public struct GenerateSpeechResult: Sendable {
    /// The synthesized audio.
    public var audio: GeneratedAudio

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about the response.
    public var response: ResponseInfo?
}

/// Synthesizes speech from text.
///
/// ```swift
/// let result = try await generateSpeech(
///     model: openai.speechModel("gpt-4o-mini-tts"),
///     text: "Good morning.",
///     voice: "alloy"
/// )
/// try result.audio.data.write(to: url)
/// ```
///
/// - Parameters:
///   - model: The speech model.
///   - text: What to say.
///   - voice: The provider's identifier for a voice.
///   - outputFormat: The container format, such as `"mp3"`.
///   - instructions: Free-text direction on delivery, for providers that accept it.
///   - speed: A speaking rate multiplier, where `1.0` is the provider's default.
///   - language: A BCP 47 language tag, where the provider supports selecting one.
///   - retryPolicy: How transient failures are retried.
///   - headers: Extra HTTP headers.
///   - providerOptions: Provider-specific settings.
/// - Returns: The audio, with any warnings.
public func generateSpeech(
    model: any SpeechModel,
    text: String,
    voice: String? = nil,
    outputFormat: String? = nil,
    instructions: String? = nil,
    speed: Double? = nil,
    language: String? = nil,
    retryPolicy: RetryPolicy = .default,
    headers: [String: String] = [:],
    providerOptions: ProviderOptions? = nil
) async throws -> GenerateSpeechResult {
    let response = try await withRetries(policy: retryPolicy) { _ in
        try await model.generate(
            SpeechModelCallOptions(
                text: text,
                voice: voice,
                outputFormat: outputFormat,
                instructions: instructions,
                speed: speed,
                language: language,
                headers: headers,
                providerOptions: providerOptions
            )
        )
    }
    return GenerateSpeechResult(
        audio: response.audio,
        warnings: response.warnings,
        providerMetadata: response.providerMetadata,
        response: response.response
    )
}

// MARK: - Transcription

/// The result of transcribing audio.
public struct TranscribeResult: Sendable {
    /// The full transcript.
    public var text: String

    /// Timed spans, when the provider returns them.
    public var segments: [TranscriptionSegment]

    /// The detected language as a BCP 47 tag, when reported.
    public var language: String?

    /// The audio's duration in seconds, when reported.
    public var durationInSeconds: Double?

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about the response.
    public var response: ResponseInfo?
}

/// Transcribes audio to text.
///
/// ```swift
/// let result = try await transcribe(
///     model: openai.transcriptionModel("whisper-1"),
///     audio: try Data(contentsOf: recording),
///     mediaType: "audio/mpeg"
/// )
/// print(result.text)
/// ```
///
/// - Parameters:
///   - model: The transcription model.
///   - audio: The audio bytes.
///   - mediaType: The audio's IANA media type. Required — providers use it to choose a decoder,
///     and several reject requests without it.
///   - filename: A filename hint. Some providers infer the container format from its extension,
///     so one is derived from `mediaType` when none is given.
///   - retryPolicy: How transient failures are retried.
///   - headers: Extra HTTP headers.
///   - providerOptions: Provider-specific settings, such as a language hint or a prompt.
/// - Returns: The transcript, with any segments the provider produced.
public func transcribe(
    model: any TranscriptionModel,
    audio: Data,
    mediaType: String,
    filename: String? = nil,
    retryPolicy: RetryPolicy = .default,
    headers: [String: String] = [:],
    providerOptions: ProviderOptions? = nil
) async throws -> TranscribeResult {
    guard !audio.isEmpty else {
        throw InvalidArgumentError(argument: "audio", message: "The audio data is empty.")
    }

    let response = try await withRetries(policy: retryPolicy) { _ in
        try await model.transcribe(
            TranscriptionModelCallOptions(
                audio: audio,
                mediaType: mediaType,
                filename: filename ?? suggestedAudioFilename(for: mediaType),
                headers: headers,
                providerOptions: providerOptions
            )
        )
    }
    return TranscribeResult(
        text: response.text,
        segments: response.segments,
        language: response.language,
        durationInSeconds: response.durationInSeconds,
        warnings: response.warnings,
        providerMetadata: response.providerMetadata,
        response: response.response
    )
}

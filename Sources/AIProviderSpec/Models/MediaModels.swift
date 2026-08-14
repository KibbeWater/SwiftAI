import Foundation

// MARK: - Images

/// The pixel dimensions of a generated image.
public struct ImageSize: Sendable, Hashable {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    /// The `"{width}x{height}"` form most providers expect.
    public var stringValue: String { "\(width)x\(height)" }

    public static let square1024 = ImageSize(width: 1024, height: 1024)
    public static let landscape1536x1024 = ImageSize(width: 1536, height: 1024)
    public static let portrait1024x1536 = ImageSize(width: 1024, height: 1536)
}

/// The shape of a generated image, for providers that size images themselves.
public struct AspectRatio: Sendable, Hashable {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    /// The `"{width}:{height}"` form most providers expect.
    public var stringValue: String { "\(width):\(height)" }

    public static let square = AspectRatio(width: 1, height: 1)
    public static let landscape = AspectRatio(width: 16, height: 9)
    public static let portrait = AspectRatio(width: 9, height: 16)
}

/// An image produced by a model.
public struct GeneratedImage: Sendable, Hashable {
    /// The image bytes.
    public var data: Data

    /// The IANA media type, such as `image/png`.
    public var mediaType: String

    public init(data: Data, mediaType: String) {
        self.data = data
        self.mediaType = mediaType
    }

    /// The bytes as a base64 string.
    public var base64EncodedString: String { data.base64EncodedString() }
}

/// The contract every image generation provider implements.
public protocol ImageModelV2: Sendable {
    var provider: String { get }
    var modelID: String { get }

    /// The most images the provider produces in one request, or `nil` when there is no limit.
    ///
    /// The core splits larger requests into several calls.
    var maxImagesPerCall: Int? { get }

    /// Generates images from a prompt.
    func generate(_ options: ImageModelCallOptions) async throws -> ImageModelResponse
}

extension ImageModelV2 {
    public var maxImagesPerCall: Int? { nil }
}

/// The current image model specification.
public typealias ImageModel = ImageModelV2

/// A request to generate images.
public struct ImageModelCallOptions: Sendable {
    /// What to depict.
    public var prompt: String

    /// How many images to produce. Never larger than ``ImageModelV2/maxImagesPerCall``.
    public var count: Int

    /// Exact dimensions. Mutually exclusive with ``aspectRatio`` on most providers.
    public var size: ImageSize?

    /// The shape to produce, when the provider chooses exact dimensions itself.
    public var aspectRatio: AspectRatio?

    /// A seed for reproducible generation, where supported.
    public var seed: Int?

    public var headers: [String: String]
    public var providerOptions: ProviderOptions?

    public init(
        prompt: String,
        count: Int = 1,
        size: ImageSize? = nil,
        aspectRatio: AspectRatio? = nil,
        seed: Int? = nil,
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil
    ) {
        self.prompt = prompt
        self.count = count
        self.size = size
        self.aspectRatio = aspectRatio
        self.seed = seed
        self.headers = headers
        self.providerOptions = providerOptions
    }
}

/// The response from an image generation request.
public struct ImageModelResponse: Sendable {
    public var images: [GeneratedImage]
    public var warnings: [CallWarning]
    public var providerMetadata: ProviderMetadata?
    public var response: ResponseInfo?

    public init(
        images: [GeneratedImage],
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        response: ResponseInfo? = nil
    ) {
        self.images = images
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.response = response
    }
}

// MARK: - Speech

/// Audio produced by a model.
public struct GeneratedAudio: Sendable, Hashable {
    /// The audio bytes.
    public var data: Data

    /// The IANA media type, such as `audio/mpeg`.
    public var mediaType: String

    public init(data: Data, mediaType: String) {
        self.data = data
        self.mediaType = mediaType
    }
}

/// The contract every speech synthesis provider implements.
public protocol SpeechModelV2: Sendable {
    var provider: String { get }
    var modelID: String { get }

    /// Synthesizes speech from text.
    func generate(_ options: SpeechModelCallOptions) async throws -> SpeechModelResponse
}

/// The current speech model specification.
public typealias SpeechModel = SpeechModelV2

/// A request to synthesize speech.
public struct SpeechModelCallOptions: Sendable {
    /// The text to speak.
    public var text: String

    /// The provider's identifier for a voice.
    public var voice: String?

    /// The desired container format, such as `"mp3"` or `"wav"`.
    public var outputFormat: String?

    /// Free-text direction on delivery, for providers that accept it.
    public var instructions: String?

    /// A speaking rate multiplier, where `1.0` is the provider's default.
    public var speed: Double?

    /// A BCP 47 language tag, where the provider supports selecting one.
    public var language: String?

    public var headers: [String: String]
    public var providerOptions: ProviderOptions?

    public init(
        text: String,
        voice: String? = nil,
        outputFormat: String? = nil,
        instructions: String? = nil,
        speed: Double? = nil,
        language: String? = nil,
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil
    ) {
        self.text = text
        self.voice = voice
        self.outputFormat = outputFormat
        self.instructions = instructions
        self.speed = speed
        self.language = language
        self.headers = headers
        self.providerOptions = providerOptions
    }
}

/// The response from a speech synthesis request.
public struct SpeechModelResponse: Sendable {
    public var audio: GeneratedAudio
    public var warnings: [CallWarning]
    public var providerMetadata: ProviderMetadata?
    public var response: ResponseInfo?

    public init(
        audio: GeneratedAudio,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        response: ResponseInfo? = nil
    ) {
        self.audio = audio
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.response = response
    }
}

// MARK: - Transcription

/// A timed span of transcribed speech.
public struct TranscriptionSegment: Sendable, Hashable {
    /// What was said.
    public var text: String

    /// When the span begins, in seconds from the start of the audio.
    public var startSecond: Double

    /// When the span ends, in seconds from the start of the audio.
    public var endSecond: Double

    public init(text: String, startSecond: Double, endSecond: Double) {
        self.text = text
        self.startSecond = startSecond
        self.endSecond = endSecond
    }
}

/// The contract every transcription provider implements.
public protocol TranscriptionModelV2: Sendable {
    var provider: String { get }
    var modelID: String { get }

    /// Transcribes audio to text.
    func transcribe(_ options: TranscriptionModelCallOptions) async throws -> TranscriptionModelResponse
}

/// The current transcription model specification.
public typealias TranscriptionModel = TranscriptionModelV2

/// A request to transcribe audio.
public struct TranscriptionModelCallOptions: Sendable {
    /// The audio bytes.
    public var audio: Data

    /// The IANA media type of the audio, such as `audio/mpeg`.
    ///
    /// Providers use this to choose a decoder and several reject requests without it.
    public var mediaType: String

    /// A filename hint, which some providers require in their multipart uploads.
    public var filename: String?

    public var headers: [String: String]
    public var providerOptions: ProviderOptions?

    public init(
        audio: Data,
        mediaType: String,
        filename: String? = nil,
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil
    ) {
        self.audio = audio
        self.mediaType = mediaType
        self.filename = filename
        self.headers = headers
        self.providerOptions = providerOptions
    }
}

/// The response from a transcription request.
public struct TranscriptionModelResponse: Sendable {
    /// The full transcript.
    public var text: String

    /// Timed spans, when the provider returns them.
    public var segments: [TranscriptionSegment]

    /// The detected language as a BCP 47 tag, when the provider reports one.
    public var language: String?

    /// The audio's duration in seconds, when the provider reports it.
    public var durationInSeconds: Double?

    public var warnings: [CallWarning]
    public var providerMetadata: ProviderMetadata?
    public var response: ResponseInfo?

    public init(
        text: String,
        segments: [TranscriptionSegment] = [],
        language: String? = nil,
        durationInSeconds: Double? = nil,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        response: ResponseInfo? = nil
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.durationInSeconds = durationInSeconds
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.response = response
    }
}

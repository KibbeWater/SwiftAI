import AIProviderSpec
import Foundation

/// A scripted ``EmbeddingModel`` for testing batching and ordering.
public final class MockEmbeddingModel: EmbeddingModel, @unchecked Sendable {
    public let provider: String
    public let modelID: String
    public let maxEmbeddingsPerCall: Int?
    public let supportsParallelCalls: Bool

    /// The dimension of the vectors produced.
    public let dimensions: Int

    /// An error to throw instead of embedding.
    public var error: (any Error)?

    private let lock = NSLock()
    private var recorded: [[String]] = []

    public init(
        provider: String = "mock",
        modelID: String = "mock-embedding",
        dimensions: Int = 4,
        maxEmbeddingsPerCall: Int? = nil,
        supportsParallelCalls: Bool = true
    ) {
        self.provider = provider
        self.modelID = modelID
        self.dimensions = dimensions
        self.maxEmbeddingsPerCall = maxEmbeddingsPerCall
        self.supportsParallelCalls = supportsParallelCalls
    }

    /// The batches the model was asked to embed, in the order the requests were made.
    public var recordedBatches: [[String]] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    public func embed(_ options: EmbeddingModelCallOptions) async throws -> EmbeddingModelResponse {
        if let error { throw error }
        lock.withLock { recorded.append(options.values) }

        // A deterministic vector derived from the text, so tests can assert which value produced
        // which embedding after reordering.
        let embeddings = options.values.map { value in
            (0..<dimensions).map { index in Double(value.count + index) }
        }
        return EmbeddingModelResponse(
            embeddings: embeddings,
            usage: Usage(inputTokens: options.values.count),
            response: ResponseInfo(id: "mock-embed")
        )
    }
}

/// A scripted ``ImageModel``.
public final class MockImageModel: ImageModel, @unchecked Sendable {
    public let provider: String
    public let modelID: String
    public let maxImagesPerCall: Int?

    private let lock = NSLock()
    private var recorded: [ImageModelCallOptions] = []

    public init(
        provider: String = "mock",
        modelID: String = "mock-image",
        maxImagesPerCall: Int? = nil
    ) {
        self.provider = provider
        self.modelID = modelID
        self.maxImagesPerCall = maxImagesPerCall
    }

    /// Every request the model received.
    public var recordedCalls: [ImageModelCallOptions] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    public func generate(_ options: ImageModelCallOptions) async throws -> ImageModelResponse {
        lock.withLock { recorded.append(options) }

        let images = (0..<options.count).map { index in
            GeneratedImage(data: Data("image-\(index)".utf8), mediaType: "image/png")
        }
        return ImageModelResponse(images: images, response: ResponseInfo(id: "mock-image"))
    }
}

/// A scripted ``SpeechModel``.
public final class MockSpeechModel: SpeechModel, @unchecked Sendable {
    public let provider: String
    public let modelID: String

    private let lock = NSLock()
    private var recorded: [SpeechModelCallOptions] = []

    public init(provider: String = "mock", modelID: String = "mock-speech") {
        self.provider = provider
        self.modelID = modelID
    }

    public var recordedCalls: [SpeechModelCallOptions] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    public func generate(_ options: SpeechModelCallOptions) async throws -> SpeechModelResponse {
        lock.withLock { recorded.append(options) }
        return SpeechModelResponse(
            audio: GeneratedAudio(data: Data(options.text.utf8), mediaType: "audio/mpeg")
        )
    }
}

/// A scripted ``TranscriptionModel``.
public final class MockTranscriptionModel: TranscriptionModel, @unchecked Sendable {
    public let provider: String
    public let modelID: String

    private let lock = NSLock()
    private var recorded: [TranscriptionModelCallOptions] = []

    public init(provider: String = "mock", modelID: String = "mock-transcription") {
        self.provider = provider
        self.modelID = modelID
    }

    public var recordedCalls: [TranscriptionModelCallOptions] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    public func transcribe(
        _ options: TranscriptionModelCallOptions
    ) async throws -> TranscriptionModelResponse {
        lock.withLock { recorded.append(options) }
        return TranscriptionModelResponse(
            text: String(decoding: options.audio, as: UTF8.self),
            segments: [TranscriptionSegment(text: "segment", startSecond: 0, endSecond: 1)],
            language: "en",
            durationInSeconds: 1
        )
    }
}

/// A provider that vends whichever mock models it was given.
public struct MockProvider: AIProvider, Sendable {
    public let name: String
    private let language: (any LanguageModel)?
    private let embedding: (any EmbeddingModel)?
    private let image: (any ImageModel)?

    public init(
        name: String = "mock",
        language: (any LanguageModel)? = nil,
        embedding: (any EmbeddingModel)? = nil,
        image: (any ImageModel)? = nil
    ) {
        self.name = name
        self.language = language
        self.embedding = embedding
        self.image = image
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        guard let language else { throw NoSuchModelError(modelID: modelID, modelKind: .language) }
        return language
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        guard let embedding else { throw NoSuchModelError(modelID: modelID, modelKind: .embedding) }
        return embedding
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        guard let image else { throw NoSuchModelError(modelID: modelID, modelKind: .image) }
        return image
    }
}

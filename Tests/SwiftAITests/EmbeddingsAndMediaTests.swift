import AITestSupport
import Foundation
import Testing

@testable import SwiftAI

@Suite("Embeddings")
struct EmbeddingsTests {
    @Test("Embeds a single value")
    func embedsSingleValue() async throws {
        let model = MockEmbeddingModel(dimensions: 3)
        let result = try await embed(model: model, value: "hello")

        #expect(result.value == "hello")
        #expect(result.embedding.count == 3)
        #expect(model.recordedBatches == [["hello"]])
    }

    @Test("Sends everything in one request when the provider has no limit")
    func singleBatchWhenUnlimited() async throws {
        let model = MockEmbeddingModel()
        let values = (0..<10).map { "value-\($0)" }

        let result = try await embedMany(model: model, values: values)
        #expect(result.embeddings.count == 10)
        #expect(model.recordedBatches.count == 1)
    }

    @Test("Splits into batches the provider accepts")
    func splitsIntoBatches() async throws {
        let model = MockEmbeddingModel(maxEmbeddingsPerCall: 3)
        let values = (0..<7).map { "value-\($0)" }

        let result = try await embedMany(model: model, values: values)
        #expect(result.embeddings.count == 7)
        #expect(model.recordedBatches.map(\.count).sorted() == [1, 3, 3])
    }

    @Test("Preserves the input order across concurrent batches")
    func preservesOrder() async throws {
        // The mock derives each vector from its input's length, so ordering is verifiable.
        let model = MockEmbeddingModel(dimensions: 1, maxEmbeddingsPerCall: 2)
        let values = ["a", "bb", "ccc", "dddd", "eeeee"]

        let result = try await embedMany(model: model, values: values)
        #expect(result.embeddings.map { $0[0] } == [1, 2, 3, 4, 5])
        #expect(result.values == values)
    }

    @Test("Sends batches one at a time when the provider forbids parallel calls")
    func respectsSequentialProviders() async throws {
        let model = MockEmbeddingModel(maxEmbeddingsPerCall: 1, supportsParallelCalls: false)
        _ = try await embedMany(model: model, values: ["a", "b", "c"])
        #expect(model.recordedBatches.count == 3)
    }

    @Test("Combines usage across batches")
    func combinesUsage() async throws {
        let model = MockEmbeddingModel(maxEmbeddingsPerCall: 2)
        let result = try await embedMany(model: model, values: ["a", "b", "c", "d"])
        #expect(result.usage.inputTokens == 4)
    }

    @Test("An empty input makes no requests")
    func emptyInputMakesNoRequests() async throws {
        let model = MockEmbeddingModel()
        let result = try await embedMany(model: model, values: [])

        #expect(result.embeddings.isEmpty)
        #expect(model.recordedBatches.isEmpty)
    }

    @Test("Pairs values with their embeddings")
    func pairsValuesWithEmbeddings() async throws {
        let result = try await embedMany(model: MockEmbeddingModel(), values: ["a", "bb"])
        #expect(result.pairs.map(\.value) == ["a", "bb"])
        #expect(result.pairs.count == 2)
    }

    @Test("Propagates a provider failure")
    func propagatesFailure() async {
        let model = MockEmbeddingModel()
        model.error = APICallError(
            message: "Bad request.",
            url: URL(string: "https://example.com")!,
            statusCode: 400
        )
        await #expect(throws: APICallError.self) {
            try await embed(model: model, value: "hello")
        }
    }

    // MARK: Similarity

    @Test("Cosine similarity is one for identical vectors")
    func similarityOfIdenticalVectors() throws {
        let vector: Embedding = [1, 2, 3]
        #expect(abs(try cosineSimilarity(vector, vector) - 1) < 1e-12)
    }

    @Test("Cosine similarity is zero for orthogonal vectors")
    func similarityOfOrthogonalVectors() throws {
        #expect(try cosineSimilarity([1, 0], [0, 1]) == 0)
    }

    @Test("Cosine similarity is negative one for opposed vectors")
    func similarityOfOpposedVectors() throws {
        #expect(abs(try cosineSimilarity([1, 2], [-1, -2]) + 1) < 1e-12)
    }

    @Test("Cosine similarity ignores magnitude")
    func similarityIgnoresMagnitude() throws {
        // The same direction at different scales is still the same meaning.
        let similarity = try cosineSimilarity([1, 1], [100, 100])
        #expect(abs(similarity - 1) < 1e-12)
    }

    @Test("A zero vector has no direction to compare")
    func similarityOfZeroVector() throws {
        #expect(try cosineSimilarity([0, 0], [1, 1]) == 0)
    }

    @Test("Rejects vectors of different lengths")
    func rejectsMismatchedLengths() {
        // Almost always means two different models produced them.
        #expect(throws: InvalidArgumentError.self) {
            try cosineSimilarity([1, 2, 3], [1, 2])
        }
    }
}

@Suite("Media generation")
struct MediaGenerationTests {
    @Test("Generates images in a single request when allowed")
    func generatesImagesInOneRequest() async throws {
        let model = MockImageModel()
        let result = try await generateImage(model: model, prompt: "A cat.", count: 3)

        #expect(result.images.count == 3)
        #expect(model.recordedCalls.count == 1)
        #expect(result.image != nil)
    }

    @Test("Splits image requests to fit the provider's limit")
    func splitsImageRequests() async throws {
        let model = MockImageModel(maxImagesPerCall: 2)
        let result = try await generateImage(model: model, prompt: "A cat.", count: 5)

        #expect(result.images.count == 5)
        #expect(model.recordedCalls.map(\.count).sorted() == [1, 2, 2])
    }

    @Test("Offsets the seed per batch so images differ")
    func offsetsSeedPerBatch() async throws {
        let model = MockImageModel(maxImagesPerCall: 1)
        _ = try await generateImage(model: model, prompt: "A cat.", count: 3, seed: 100)

        // Reusing one seed would produce three identical images.
        #expect(Set(model.recordedCalls.compactMap(\.seed)).count == 3)
    }

    @Test("Passes size and provider options through")
    func passesImageOptions() async throws {
        let model = MockImageModel()
        _ = try await generateImage(
            model: model,
            prompt: "A cat.",
            size: .square1024,
            providerOptions: ["mock": ["quality": "high"]]
        )

        let call = try #require(model.recordedCalls.first)
        #expect(call.size == .square1024)
        #expect(call.providerOptions?.value("quality", for: "mock")?.stringValue == "high")
    }

    @Test("Rejects a request for no images")
    func rejectsZeroImages() async {
        await #expect(throws: InvalidArgumentError.self) {
            try await generateImage(model: MockImageModel(), prompt: "A cat.", count: 0)
        }
    }

    @Test("Synthesizes speech")
    func synthesizesSpeech() async throws {
        let model = MockSpeechModel()
        let result = try await generateSpeech(model: model, text: "Hello.", voice: "alloy")

        #expect(String(decoding: result.audio.data, as: UTF8.self) == "Hello.")
        #expect(model.recordedCalls.first?.voice == "alloy")
    }

    @Test("Transcribes audio")
    func transcribesAudio() async throws {
        let model = MockTranscriptionModel()
        let result = try await transcribe(
            model: model,
            audio: Data("spoken words".utf8),
            mediaType: "audio/mpeg"
        )

        #expect(result.text == "spoken words")
        #expect(result.segments.count == 1)
        #expect(result.language == "en")
    }

    @Test("Derives a filename from the media type")
    func derivesFilename() async throws {
        // Several providers infer the container format from the filename extension.
        let model = MockTranscriptionModel()
        _ = try await transcribe(model: model, audio: Data("x".utf8), mediaType: "audio/mpeg")
        #expect(model.recordedCalls.first?.filename == "audio.mp3")
    }

    @Test(
        "Maps common audio media types to plausible extensions",
        arguments: [
            ("audio/mpeg", "audio.mp3"),
            ("audio/wav", "audio.wav"),
            ("audio/x-m4a", "audio.m4a"),
            ("audio/webm", "audio.webm"),
            ("audio/mpeg; codecs=mp3", "audio.mp3"),
            ("audio/unknown-format", "audio.unknown-format"),
        ]
    )
    func mapsAudioMediaTypes(mediaType: String, expected: String) {
        #expect(suggestedAudioFilename(for: mediaType) == expected)
    }

    @Test("Rejects empty audio")
    func rejectsEmptyAudio() async {
        await #expect(throws: InvalidArgumentError.self) {
            try await transcribe(model: MockTranscriptionModel(), audio: Data(), mediaType: "audio/mpeg")
        }
    }
}

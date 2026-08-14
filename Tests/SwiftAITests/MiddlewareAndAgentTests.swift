import AITestSupport
import Foundation
import Testing

@testable import SwiftAI

@Suite("Middleware")
struct MiddlewareTests {
    /// Records the order in which middleware sees a call.
    private struct OrderRecorder: LanguageModelMiddleware {
        let label: String
        let recorder: Recorder

        func generate(
            _ options: LanguageModelCallOptions,
            model: any LanguageModel,
            next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse
        ) async throws -> LanguageModelResponse {
            recorder.record("\(label):before")
            let response = try await next(options)
            recorder.record("\(label):after")
            return response
        }
    }

    @Test("Applies middleware outside-in")
    func appliesOutsideIn() async throws {
        let recorder = Recorder()
        let model = wrapLanguageModel(
            model: MockLanguageModel(text: "ok"),
            middleware: [
                OrderRecorder(label: "outer", recorder: recorder),
                OrderRecorder(label: "inner", recorder: recorder),
            ]
        )

        _ = try await generateText(model: model, prompt: "Hi")
        #expect(recorder.recorded == ["outer:before", "inner:before", "inner:after", "outer:after"])
    }

    @Test("An empty middleware list returns the model unchanged")
    func emptyMiddlewareIsIdentity() {
        let base = MockLanguageModel(text: "ok")
        let wrapped = wrapLanguageModel(model: base, middleware: [])
        #expect(wrapped is MockLanguageModel)
    }

    @Test("Middleware can short-circuit the call")
    func middlewareCanShortCircuit() async throws {
        struct Cache: LanguageModelMiddleware {
            func generate(
                _ options: LanguageModelCallOptions,
                model: any LanguageModel,
                next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse
            ) async throws -> LanguageModelResponse {
                LanguageModelResponse(content: [.text("cached")], finishReason: .stop)
            }
        }

        let base = MockLanguageModel(text: "from the model")
        let model = wrapLanguageModel(model: base, middleware: [Cache()])
        let result = try await generateText(model: model, prompt: "Hi")

        #expect(result.text == "cached")
        // The model was never reached, which is the point of a cache.
        #expect(base.recordedCalls.isEmpty)
    }

    @Test("Overrides the reported identity")
    func overridesIdentity() {
        let model = wrapLanguageModel(
            model: MockLanguageModel(text: "ok"),
            middleware: [],
            provider: "custom",
            modelID: "aliased"
        )
        #expect(model.provider == "custom")
        #expect(model.modelID == "aliased")
    }

    // MARK: Default settings

    @Test("Fills in settings the call did not specify")
    func fillsMissingSettings() async throws {
        let base = MockLanguageModel(text: "ok")
        let model = wrapLanguageModel(
            model: base,
            middleware: [DefaultSettingsMiddleware(temperature: 0.3, maxOutputTokens: 500)]
        )

        _ = try await generateText(model: model, prompt: "Hi")
        #expect(base.recordedOptions()?.temperature == 0.3)
        #expect(base.recordedOptions()?.maxOutputTokens == 500)
    }

    @Test("The call site wins over a default")
    func callSiteWinsOverDefault() async throws {
        let base = MockLanguageModel(text: "ok")
        let model = wrapLanguageModel(
            model: base,
            middleware: [DefaultSettingsMiddleware(temperature: 0.3)]
        )

        _ = try await generateText(model: model, prompt: "Hi", settings: GenerationSettings(temperature: 1.0))
        #expect(base.recordedOptions()?.temperature == 1.0)
    }

    @Test("Provider options merge rather than replace")
    func providerOptionsMerge() async throws {
        let base = MockLanguageModel(text: "ok")
        let model = wrapLanguageModel(
            model: base,
            middleware: [
                DefaultSettingsMiddleware(providerOptions: ["mock": ["a": 1, "b": 2]])
            ]
        )

        var settings = GenerationSettings()
        settings.providerOptions = ["mock": ["b": 99]]
        _ = try await generateText(model: model, prompt: "Hi", settings: settings)

        let options = try #require(base.recordedOptions()?.providerOptions)
        #expect(options.value("a", for: "mock")?.intValue == 1)
        #expect(options.value("b", for: "mock")?.intValue == 99)
    }

    // MARK: Simulated streaming

    @Test("Turns a buffered response into a well-formed stream")
    func simulatesStreaming() async throws {
        let base = MockLanguageModel(responses: [.reasoningThenText(reasoning: "Hmm.", text: "Answer.")])
        let model = wrapLanguageModel(model: base, middleware: [SimulateStreamingMiddleware()])

        let stream = streamText(model: model, prompt: "Why?")
        let parts = try await stream.fullStream.collect()

        #expect(parts.first?.kind == .start)
        #expect(parts.last?.kind == .finish)
        #expect(try await stream.text == "Answer.")
        #expect(try await stream.reasoningText == "Hmm.")
        // The buffered path was used underneath.
        #expect(base.recordedCalls.allSatisfy { !$0.wasStreaming })
    }

    // MARK: Reasoning extraction

    @Test("Lifts inline reasoning out of buffered text")
    func extractsReasoningFromBufferedText() async throws {
        let base = MockLanguageModel(
            responses: [.text("<think>The sky scatters blue light.</think>\nBecause of scattering.")]
        )
        let model = wrapLanguageModel(model: base, middleware: [ExtractReasoningMiddleware()])

        let result = try await generateText(model: model, prompt: "Why is the sky blue?")
        #expect(result.text == "Because of scattering.")
        #expect(result.reasoningText == "The sky scatters blue light.")
    }

    @Test("Handles a response that begins inside the tag")
    func handlesImplicitOpeningTag() async throws {
        let base = MockLanguageModel(responses: [.text("Reasoning here.</think>\nThe answer.")])
        let model = wrapLanguageModel(
            model: base,
            middleware: [ExtractReasoningMiddleware(startsInsideReasoning: true)]
        )

        let result = try await generateText(model: model, prompt: "Why?")
        #expect(result.text == "The answer.")
        #expect(result.reasoningText == "Reasoning here.")
    }

    @Test("Leaves text without tags untouched")
    func leavesUntaggedTextAlone() async throws {
        let base = MockLanguageModel(responses: [.text("Just an answer.")])
        let model = wrapLanguageModel(model: base, middleware: [ExtractReasoningMiddleware()])

        let result = try await generateText(model: model, prompt: "Hi")
        #expect(result.text == "Just an answer.")
        #expect(result.reasoningText == nil)
    }

    @Test("Extracts reasoning from a stream even when tags straddle deltas", arguments: [1, 2, 3, 5, 8])
    func extractsReasoningFromStream(chunkSize: Int) async throws {
        // A chunk size of one splits `<think>` across seven deltas, which is exactly the case a
        // naive implementation gets wrong.
        let base = MockLanguageModel(
            responses: [.text("<think>Scattering.</think>Because of scattering.", textChunkSize: chunkSize)]
        )
        let model = wrapLanguageModel(model: base, middleware: [ExtractReasoningMiddleware()])

        let stream = streamText(model: model, prompt: "Why?")
        async let text = stream.textStream.collect()
        async let reasoning = stream.reasoningStream.collect()

        #expect(try await text.joined() == "Because of scattering.")
        #expect(try await reasoning.joined() == "Scattering.")
    }

    @Test("A partial tag at the end of a stream is still emitted as text")
    func emitsUnterminatedPartialTag() async throws {
        // `<thi` is not a tag; dropping it would silently lose output.
        let base = MockLanguageModel(responses: [.text("Answer.<thi", textChunkSize: 2)])
        let model = wrapLanguageModel(model: base, middleware: [ExtractReasoningMiddleware()])

        let stream = streamText(model: model, prompt: "Hi")
        #expect(try await stream.textStream.collect().joined() == "Answer.<thi")
    }
}

@Suite("Splitter")
struct ReasoningTagSplitterTests {
    @Test("Holds back only what could still become a tag")
    func holdsBackAmbiguousSuffix() {
        var splitter = ReasoningTagSplitter(openingTag: "<think>", closingTag: "</think>", isReasoning: false)

        // `<thi` could still become `<think>`, so it is withheld; `Hello ` cannot, so it is emitted.
        let pieces = splitter.consume("Hello <thi")
        #expect(pieces.map(\.text) == ["Hello "])

        // The tag completes, so what follows is reasoning.
        let more = splitter.consume("nk>Reasoning")
        #expect(more.map(\.text) == ["Reasoning"])
        #expect(more.map(\.isReasoning) == [true])
    }

    @Test("Flushing emits whatever remains")
    func flushEmitsRemainder() {
        var splitter = ReasoningTagSplitter(openingTag: "<think>", closingTag: "</think>", isReasoning: false)
        _ = splitter.consume("Text<thi")
        #expect(splitter.flush().map(\.text) == ["<thi"])
    }
}

@Suite("ProviderRegistry")
struct ProviderRegistryTests {
    private func registry() -> ProviderRegistry {
        ProviderRegistry([
            "mock": MockProvider(
                language: MockLanguageModel(modelID: "mock-model", text: "hi"),
                embedding: MockEmbeddingModel()
            )
        ])
    }

    @Test("Resolves a qualified identifier")
    func resolvesQualifiedIdentifier() throws {
        #expect(try registry().languageModel("mock:some-model").modelID == "mock-model")
    }

    @Test("Splits on the first separator only")
    func splitsOnFirstSeparatorOnly() throws {
        // Model identifiers containing the separator must still resolve.
        #expect(throws: Never.self) { try registry().languageModel("mock:family:version:2") }
    }

    @Test("Reports an unqualified identifier clearly")
    func reportsUnqualifiedIdentifier() {
        #expect(throws: NoSuchModelError.self) { try registry().languageModel("gpt-5") }
    }

    @Test("Lists the available providers when one is unknown")
    func listsAvailableProviders() throws {
        var caught: NoSuchProviderError?
        do {
            _ = try registry().languageModel("openai:gpt-5")
        } catch let error as NoSuchProviderError {
            caught = error
        }
        #expect(try #require(caught).message.contains("mock"))
    }

    @Test("Resolves other model kinds")
    func resolvesOtherModelKinds() throws {
        #expect(throws: Never.self) { try registry().embeddingModel("mock:embed") }
        #expect(throws: NoSuchModelError.self) { try registry().imageModel("mock:image") }
    }

    @Test("A custom provider names configured models")
    func customProviderNamesModels() throws {
        let provider = CustomProvider(
            languageModels: ["fast": MockLanguageModel(modelID: "small", text: "hi")]
        )
        #expect(try provider.languageModel("fast").modelID == "small")
        #expect(throws: NoSuchModelError.self) { try provider.languageModel("unknown") }
    }

    @Test("A custom provider falls back for names it does not define")
    func customProviderFallsBack() throws {
        let provider = CustomProvider(
            languageModels: ["fast": MockLanguageModel(modelID: "small", text: "hi")],
            fallback: MockProvider(language: MockLanguageModel(modelID: "default", text: "hi"))
        )
        #expect(try provider.languageModel("anything-else").modelID == "default")
    }
}

@Suite("Agent")
struct AgentTests {
    @Test("Applies its configuration to every call")
    func appliesConfiguration() async throws {
        let model = MockLanguageModel(text: "ok")
        let agent = Agent(
            model: model,
            system: "Be terse.",
            settings: GenerationSettings(temperature: 0.1)
        )

        _ = try await agent.generate(prompt: "Hello")
        let options = try #require(model.recordedOptions())
        #expect(options.prompt.first?.text == "Be terse.")
        #expect(options.temperature == 0.1)
    }

    @Test("Loops by default, unlike the free functions")
    func loopsByDefault() async throws {
        let recorder = Recorder()
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("Done.")]
        )
        let weather = tool("weather", description: "Look up the weather.") { (query: WeatherQuery, _) in
            recorder.record(query.city)
            return .text("Rain.")
        }

        let result = try await Agent(model: model, tools: [weather]).generate(prompt: "Weather?")
        #expect(result.steps.count == 2)
        #expect(result.text == "Done.")
    }

    @Test("Streams with the same configuration")
    func streamsWithSameConfiguration() async throws {
        let model = MockLanguageModel(responses: [.text("Streamed.", textChunkSize: 3)])
        let agent = Agent(model: model, system: "Be terse.")

        let stream = agent.stream(prompt: "Hello")
        #expect(try await stream.textStream.collect().joined() == "Streamed.")
        #expect(model.recordedOptions()?.prompt.first?.text == "Be terse.")
    }

    @Test("Copies with changes leave the original untouched")
    func copiesAreIndependent() {
        let agent = Agent(model: MockLanguageModel(text: "ok"), system: "Original.")
        let variant = agent.with(system: "Changed.")

        #expect(agent.system == "Original.")
        #expect(variant.system == "Changed.")
    }

    @Test("Adding tools extends rather than replaces")
    func addingToolsExtends() {
        let first = tool("a", description: "A.") { (_: JSONValue, _) in .text("a") }
        let second = tool("b", description: "B.") { (_: JSONValue, _) in .text("b") }

        let agent = Agent(model: MockLanguageModel(text: "ok"), tools: [first]).addingTools([second])
        #expect(agent.tools.map(\.name) == ["a", "b"])
    }
}

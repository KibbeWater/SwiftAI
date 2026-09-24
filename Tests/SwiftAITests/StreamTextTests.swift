import AITestSupport
import Foundation
import Testing

@testable import SwiftAI

@Suite("streamText")
struct StreamTextTests {
    private func weatherTool(_ recorder: Recorder = Recorder()) -> ClosureTool<WeatherQuery> {
        tool("weather", description: "Look up the weather.") { (query: WeatherQuery, _) in
            recorder.record(query.city)
            return .text("It is raining in \(query.city).")
        }
    }

    // MARK: - Text streaming

    @Test("Streams text deltas in order")
    func streamsTextDeltas() async throws {
        let model = MockLanguageModel(responses: [.text("Hello, world!", textChunkSize: 3)])
        let stream = streamText(model: model, prompt: "Greet me")

        var received: [String] = []
        for try await delta in stream.textStream {
            received.append(delta)
        }

        #expect(received == ["Hel", "lo,", " wo", "rld", "!"])
        #expect(received.joined() == "Hello, world!")
    }

    @Test("Resolves final values after the stream ends")
    func resolvesFinalValues() async throws {
        let model = MockLanguageModel(responses: [.text("Complete answer.", textChunkSize: 4)])
        let stream = streamText(model: model, prompt: "Answer")

        for try await _ in stream.textStream {}

        #expect(try await stream.text == "Complete answer.")
        #expect(try await stream.finishReason == .stop)
        #expect(try await stream.usage.inputTokens == 10)
        #expect(try await stream.steps.count == 1)
    }

    @Test("Final values are available without consuming the stream")
    func finalValuesWithoutConsuming() async throws {
        // Awaiting a promised value keeps the generation alive on its own.
        let model = MockLanguageModel(responses: [.text("Answer.", textChunkSize: 2)])
        let stream = streamText(model: model, prompt: "Answer")
        #expect(try await stream.text == "Answer.")
    }

    @Test("The stream can be consumed after it has finished")
    func lateSubscriptionReplaysEverything() async throws {
        let model = MockLanguageModel(responses: [.text("Replayed.", textChunkSize: 3)])
        let stream = streamText(model: model, prompt: "Say it")

        // Wait for completion, then subscribe.
        _ = try await stream.steps

        var received: [String] = []
        for try await delta in stream.textStream {
            received.append(delta)
        }
        #expect(received.joined() == "Replayed.")
    }

    @Test("Two consumers see identical sequences")
    func twoConsumersSeeTheSameSequence() async throws {
        let model = MockLanguageModel(responses: [.text("Shared output.", textChunkSize: 2)])
        let stream = streamText(model: model, prompt: "Say it")

        async let first = stream.textStream.collect()
        async let second = stream.fullStream.collect()

        let deltas = try await first
        let parts = try await second

        #expect(deltas.joined() == "Shared output.")
        #expect(parts.compactMap(\.textDelta) == deltas)
    }

    @Test("Reasoning is streamed separately from the answer")
    func streamsReasoningSeparately() async throws {
        let model = MockLanguageModel(
            responses: [.reasoningThenText(reasoning: "Think think.", text: "Answer.")]
        )
        let stream = streamText(model: model, prompt: "Why?")

        async let reasoning = stream.reasoningStream.collect()
        async let text = stream.textStream.collect()

        #expect(try await reasoning.joined() == "Think think.")
        #expect(try await text.joined() == "Answer.")
    }

    // MARK: - Full stream shape

    @Test("A simple generation produces a well-formed part sequence")
    func fullStreamShape() async throws {
        let model = MockLanguageModel(responses: [.text("Hi", textChunkSize: 1)])
        let stream = streamText(model: model, prompt: "Greet")

        let kinds = try await stream.fullStream.collect().map(\.kind)
        #expect(
            kinds == [
                .start, .stepStart,
                .textStart, .textDelta, .textDelta, .textEnd,
                .stepFinish, .finish,
            ]
        )
    }

    @Test("A tool round trip appears in the stream")
    func toolRoundTripAppearsInStream() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("Yes.", textChunkSize: 3)]
        )
        let stream = streamText(
            model: model,
            prompt: "Umbrella?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)]
        )

        let parts = try await stream.fullStream.collect()
        let kinds = parts.map(\.kind)

        // Arguments stream in as JSON fragments before the parsed call arrives.
        #expect(kinds.contains(.toolInputStart))
        #expect(kinds.contains(.toolInputDelta))
        #expect(kinds.contains(.toolCall))
        // Then the SDK runs the tool and publishes the outcome.
        #expect(kinds.contains(.toolWillRun))
        #expect(kinds.contains(.toolResult))
        // Two steps, so two boundaries.
        #expect(kinds.filter { $0 == .stepStart }.count == 2)
        #expect(kinds.filter { $0 == .stepFinish }.count == 2)
        #expect(kinds.last == .finish)
    }

    @Test("Tool argument deltas reassemble into the parsed call")
    func toolArgumentDeltasReassemble() async throws {
        let model = MockLanguageModel(
            responses: [
                MockLanguageModel.Response(
                    content: [.toolCall(ToolCallPart(toolCallID: "c1", toolName: "weather", input: ["city": "Lund"]))],
                    finishReason: .toolCalls,
                    textChunkSize: 2
                ),
                .text("done"),
            ]
        )
        let stream = streamText(
            model: model,
            prompt: "Weather?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)]
        )

        let parts = try await stream.fullStream.collect()
        let assembled = parts.compactMap { part -> String? in
            guard case .toolInputDelta(_, let delta) = part else { return nil }
            return delta
        }.joined()

        #expect(try JSONValue.parse(assembled) == ["city": "Lund"])
    }

    @Test("The final part reports total usage across steps")
    func finalPartReportsTotalUsage() async throws {
        let model = MockLanguageModel(
            responses: [
                .toolCall(name: "weather", input: ["city": "Malmö"], usage: Usage(inputTokens: 10, outputTokens: 5)),
                .text("done", usage: Usage(inputTokens: 20, outputTokens: 7)),
            ]
        )
        let stream = streamText(
            model: model,
            prompt: "Weather?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)]
        )

        let parts = try await stream.fullStream.collect()
        guard case .finish(let reason, let usage)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .stop)
        #expect(usage.inputTokens == 30)
        #expect(usage.outputTokens == 12)
    }

    // MARK: - Parity with buffered generation

    @Test("Produces the same result as the buffered path")
    func parityWithGenerateText() async throws {
        func responses() -> [MockLanguageModel.Response] {
            [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("Yes, take one.")]
        }

        let buffered = try await generateText(
            model: MockLanguageModel(responses: responses()),
            prompt: "Umbrella?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)]
        )
        let streamed = try await streamText(
            model: MockLanguageModel(responses: responses()),
            prompt: "Umbrella?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)]
        ).collected

        #expect(buffered.text == streamed.text)
        #expect(buffered.steps.count == streamed.steps.count)
        #expect(buffered.totalUsage == streamed.totalUsage)
        #expect(buffered.responseMessages.map(\.role) == streamed.responseMessages.map(\.role))
        #expect(buffered.finishReason == streamed.finishReason)
    }

    @Test("Response messages continue the conversation")
    func responseMessagesContinueConversation() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("Yes.")]
        )
        let stream = streamText(
            model: model,
            prompt: "Umbrella?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)]
        )
        #expect(try await stream.responseMessages.map(\.role) == [.assistant, .tool, .assistant])
    }

    // MARK: - Callbacks

    @Test("Calls the step and finish callbacks")
    func callsCallbacks() async throws {
        let observed = Recorder()
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("done")]
        )

        let stream = streamText(
            model: model,
            prompt: "Weather?",
            tools: [weatherTool()],
            stopWhen: [.stepCount(3)],
            onStepFinish: { step in observed.record("step:\(step.finishReason.rawValue)") },
            onFinish: { result in observed.record("finish:\(result.steps.count)") }
        )
        try await stream.consume()
        // `onFinish` runs after the stream closes, so give it a moment to be recorded.
        _ = try await stream.steps

        #expect(observed.recorded.contains("step:tool-calls"))
        #expect(observed.recorded.contains("step:stop"))
    }

    // MARK: - Failures

    @Test("A failure surfaces when the stream is iterated, not when it is started")
    func failureSurfacesOnIteration() async throws {
        let error = APICallError(
            message: "Invalid key.",
            url: URL(string: "https://example.com")!,
            statusCode: 401
        )
        let model = MockLanguageModel(responses: [.failure(error)])

        // Starting the generation does not throw, because nothing has been attempted yet.
        let stream = streamText(model: model, prompt: "Hi")

        await #expect(throws: APICallError.self) {
            for try await _ in stream.textStream {}
        }
    }

    @Test("A failure also fails the promised values")
    func failurePropagatesToPromisedValues() async {
        let model = MockLanguageModel(
            responses: [
                .failure(APICallError(message: "Nope.", url: URL(string: "https://example.com")!, statusCode: 400))
            ]
        )
        let stream = streamText(model: model, prompt: "Hi")

        await #expect(throws: APICallError.self) { _ = try await stream.text }
    }

    @Test("The error callback receives the failure")
    func errorCallbackReceivesFailure() async throws {
        let observed = Recorder()
        let model = MockLanguageModel(
            responses: [
                .failure(APICallError(message: "Nope.", url: URL(string: "https://example.com")!, statusCode: 400))
            ]
        )

        let stream = streamText(
            model: model,
            prompt: "Hi",
            onError: { error in observed.record(String(describing: type(of: error))) }
        )
        _ = try? await stream.text

        #expect(observed.recorded == ["APICallError"])
    }

    @Test("Retries a transient failure before any part is published")
    func retriesBeforeFirstPart() async throws {
        let model = MockLanguageModel(
            responses: [
                .failure(
                    APICallError(message: "Overloaded.", url: URL(string: "https://example.com")!, statusCode: 503)
                ),
                .text("recovered"),
            ]
        )
        var settings = GenerationSettings()
        settings.retryPolicy = RetryPolicy(maximumRetries: 2, initialDelay: .milliseconds(1))

        let stream = streamText(model: model, prompt: "Hi", settings: settings)
        #expect(try await stream.text == "recovered")
    }

    // MARK: - Cancellation

    @Test("Cancelling the result terminates every view")
    func cancellationTerminatesViews() async throws {
        let model = MockLanguageModel(responses: [.text("slow")], repeatsLastResponse: true)
        model.onCall = { _ in try await Task.sleep(for: .seconds(10)) }

        let stream = streamText(model: model, prompt: "Hi")
        // Let the pump reach the model call before cancelling.
        try await Task.sleep(for: .milliseconds(20))
        stream.cancel()

        await #expect(throws: (any Error).self) {
            for try await _ in stream.textStream {}
        }
    }

    @Test("Cancelling the consuming task does not hang")
    func cancellingConsumerDoesNotHang() async throws {
        let model = MockLanguageModel(responses: [.text("slow")], repeatsLastResponse: true)
        model.onCall = { _ in try await Task.sleep(for: .seconds(10)) }

        let stream = streamText(model: model, prompt: "Hi")
        let task = Task {
            for try await _ in stream.textStream {}
        }
        try await Task.sleep(for: .milliseconds(20))
        task.cancel()

        _ = try? await task.value
        stream.cancel()
    }

    // MARK: - Reasoning metadata

    @Test("Keeps a streamed reasoning block's metadata for the next turn")
    func keepsReasoningMetadata() async throws {
        // OpenRouter replays reasoning through `reasoning_details`, which must survive streaming
        // exactly as it survives a buffered response, or the next turn loses the model's thinking.
        let details: ProviderOptions = ["openrouter": ["reasoning_details": [["type": "reasoning.text", "text": "Hmm"]]]]
        let model = MockLanguageModel(responses: [
            .init(content: [.reasoning(ReasoningPart("Hmm", providerOptions: details)), .text(TextPart("Done."))]),
        ])

        let stream = streamText(model: model, prompt: "Think")
        let reasoning = try await stream.responseMessages.first?.assistantReasoning

        #expect(reasoning?.providerOptions == details)
    }

    @Test("Keeps an empty reasoning block that carries metadata")
    func keepsEncryptedReasoning() async throws {
        // Encrypted reasoning has no visible text, only the opaque blob the provider needs back.
        let encrypted: ProviderOptions = ["openrouter": ["reasoning_details": [["type": "reasoning.encrypted", "data": "b64"]]]]
        let model = MockLanguageModel(responses: [
            .init(content: [.reasoning(ReasoningPart("", providerOptions: encrypted)), .text(TextPart("Done."))]),
        ])

        let stream = streamText(model: model, prompt: "Think")
        let reasoning = try await stream.responseMessages.first?.assistantReasoning

        #expect(reasoning?.text == "")
        #expect(reasoning?.providerOptions == encrypted)
    }
}

private extension ModelMessage {
    /// The first reasoning part of an assistant message.
    var assistantReasoning: ReasoningPart? {
        guard case .assistant(let message) = self else { return nil }
        return message.content.lazy.compactMap { part -> ReasoningPart? in
            guard case .reasoning(let reasoning) = part else { return nil }
            return reasoning
        }.first
    }
}

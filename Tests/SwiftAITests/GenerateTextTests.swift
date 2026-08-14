import AITestSupport
import Foundation
import Testing

@testable import SwiftAI

// MARK: - Fixtures

@Structured("Where to look up the weather.")
struct WeatherQuery {
    @Guidance("The city name.")
    var city: String
}

/// A tool defined as a type, exercising the conformance path.
private struct WeatherTool: Tool {
    typealias Arguments = WeatherQuery

    var name: String { "weather" }
    var description: String { "Look up the current weather in a city." }

    /// Records the cities the model asked about, so tests can assert on decoded arguments.
    let recorder: Recorder

    func call(_ arguments: WeatherQuery, context: ToolContext) async throws -> ToolOutput {
        recorder.record(arguments.city)
        return .text("It is 14°C and raining in \(arguments.city).")
    }
}

/// A thread-safe recorder for observing tool invocations.
final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []

    var recorded: [String] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }

    func record(_ value: String) {
        lock.lock()
        values.append(value)
        lock.unlock()
    }
}

// MARK: - Tests

@Suite("generateText")
struct GenerateTextTests {
    // MARK: Basics

    @Test("Returns the model's text")
    func returnsText() async throws {
        let model = MockLanguageModel(responses: [.text("The sky is blue.")])
        let result = try await generateText(model: model, prompt: "Why is the sky blue?")

        #expect(result.text == "The sky is blue.")
        #expect(result.finishReason == .stop)
        #expect(result.steps.count == 1)
    }

    @Test("A string literal is sent as a single user message")
    func stringLiteralPrompt() async throws {
        let model = MockLanguageModel(text: "ok")
        _ = try await generateText(model: model, prompt: "Hello")

        let options = try #require(model.recordedOptions())
        #expect(options.prompt.count == 1)
        #expect(options.prompt.first?.role == .user)
        #expect(options.prompt.first?.text == "Hello")
    }

    @Test("A system message is prepended")
    func systemMessageIsPrepended() async throws {
        let model = MockLanguageModel(text: "ok")
        _ = try await generateText(model: model, system: "Be terse.", prompt: "Hello")

        let options = try #require(model.recordedOptions())
        #expect(options.prompt.map(\.role) == [.system, .user])
        #expect(options.prompt.first?.text == "Be terse.")
    }

    @Test("Rejects a system message supplied twice")
    func rejectsDuplicateSystemMessage() async {
        let model = MockLanguageModel(text: "ok")
        await #expect(throws: InvalidPromptError.self) {
            try await generateText(
                model: model,
                system: "Be terse.",
                prompt: Prompt([.system("Be verbose."), .user("Hello")])
            )
        }
    }

    @Test("Rejects a system message after the first position")
    func rejectsMisplacedSystemMessage() async {
        let model = MockLanguageModel(text: "ok")
        await #expect(throws: InvalidPromptError.self) {
            try await generateText(model: model, prompt: Prompt([.user("Hi"), .system("Now be terse.")]))
        }
    }

    @Test("Rejects an empty prompt")
    func rejectsEmptyPrompt() async {
        let model = MockLanguageModel(text: "ok")
        await #expect(throws: InvalidPromptError.self) {
            try await generateText(model: model, prompt: Prompt([ModelMessage]()))
        }
    }

    @Test("Passes generation settings through to the provider")
    func passesSettings() async throws {
        let model = MockLanguageModel(text: "ok")
        let settings = GenerationSettings(
            maxOutputTokens: 256,
            temperature: 0.2,
            topP: 0.9,
            stopSequences: ["END"],
            providerOptions: ["mock": ["custom": true]]
        )
        _ = try await generateText(model: model, prompt: "Hi", settings: settings)

        let options = try #require(model.recordedOptions())
        #expect(options.maxOutputTokens == 256)
        #expect(options.temperature == 0.2)
        #expect(options.topP == 0.9)
        #expect(options.stopSequences == ["END"])
        #expect(options.providerOptions?.value("custom", for: "mock")?.boolValue == true)
    }

    @Test("Surfaces reasoning separately from the answer")
    func surfacesReasoning() async throws {
        let model = MockLanguageModel(
            responses: [.reasoningThenText(reasoning: "Rayleigh scattering.", text: "Because of scattering.")]
        )
        let result = try await generateText(model: model, prompt: "Why?")

        #expect(result.text == "Because of scattering.")
        #expect(result.reasoningText == "Rayleigh scattering.")
        #expect(result.usage.reasoningTokens == 20)
    }

    @Test("Reports warnings from the provider")
    func reportsWarnings() async throws {
        let model = MockLanguageModel(
            responses: [
                MockLanguageModel.Response(
                    content: [.text("ok")],
                    warnings: [.unsupportedSetting(setting: "topK")]
                )
            ]
        )
        let result = try await generateText(model: model, prompt: "Hi")
        #expect(result.warnings.count == 1)
    }

    @Test("Propagates a provider failure")
    func propagatesProviderFailure() async {
        let error = APICallError(
            message: "Invalid key.",
            url: URL(string: "https://example.com")!,
            statusCode: 401
        )
        let model = MockLanguageModel(responses: [.failure(error)])

        await #expect(throws: APICallError.self) {
            try await generateText(model: model, prompt: "Hi")
        }
    }

    @Test("Retries a transient provider failure")
    func retriesTransientFailure() async throws {
        let model = MockLanguageModel(
            responses: [
                .failure(
                    APICallError(
                        message: "Overloaded.",
                        url: URL(string: "https://example.com")!,
                        statusCode: 503
                    )
                ),
                .text("recovered"),
            ]
        )
        var settings = GenerationSettings()
        settings.retryPolicy = RetryPolicy(maximumRetries: 2, initialDelay: .milliseconds(1))

        let result = try await generateText(model: model, prompt: "Hi", settings: settings)
        #expect(result.text == "recovered")
        #expect(model.recordedCalls.count == 2)
    }

    // MARK: Tools

    @Test("Sends tool definitions to the model")
    func sendsToolDefinitions() async throws {
        let model = MockLanguageModel(text: "ok")
        _ = try await generateText(
            model: model,
            prompt: "Hi",
            tools: [WeatherTool(recorder: Recorder())]
        )

        let options = try #require(model.recordedOptions())
        #expect(options.tools.count == 1)
        guard case .function(let function) = options.tools[0] else {
            Issue.record("Expected a function tool.")
            return
        }
        #expect(function.name == "weather")
        #expect(function.description == "Look up the current weather in a city.")
        #expect(function.inputSchema.jsonValue()["properties"]?["city"]?["type"]?.stringValue == "string")
        #expect(options.toolChoice == .auto)
    }

    @Test("Omits the tool choice when there are no tools")
    func omitsToolChoiceWithoutTools() async throws {
        let model = MockLanguageModel(text: "ok")
        _ = try await generateText(model: model, prompt: "Hi", toolChoice: .required)
        #expect(model.recordedOptions()?.toolChoice == nil)
    }

    @Test("A single step does not feed tool results back")
    func singleStepDoesNotLoop() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("unused")]
        )
        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())]
        )

        // The tool still runs, but the loop stops because the default budget is one step.
        #expect(model.recordedCalls.count == 1)
        #expect(result.steps.count == 1)
        #expect(result.toolResults.count == 1)
        #expect(result.finishReason == .toolCalls)
    }

    @Test("Runs the tool loop until the model answers")
    func runsToolLoop() async throws {
        let recorder = Recorder()
        let model = MockLanguageModel(
            responses: [
                .toolCall(name: "weather", input: ["city": "Malmö"]),
                .text("Yes, take an umbrella."),
            ]
        )

        let result = try await generateText(
            model: model,
            prompt: "Umbrella tomorrow?",
            tools: [WeatherTool(recorder: recorder)],
            stopWhen: [.stepCount(5)]
        )

        #expect(result.text == "Yes, take an umbrella.")
        #expect(result.steps.count == 2)
        #expect(recorder.recorded == ["Malmö"])

        // The second call must include the assistant's tool call and the tool's result.
        let secondPrompt = try #require(model.recordedOptions(at: 1)?.prompt)
        #expect(secondPrompt.map(\.role) == [.user, .assistant, .tool])
    }

    @Test("Decodes tool arguments into the declared type")
    func decodesToolArguments() async throws {
        let recorder = Recorder()
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Göteborg"]), .text("done")]
        )
        _ = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: recorder)],
            stopWhen: [.stepCount(3)]
        )
        #expect(recorder.recorded == ["Göteborg"])
    }

    @Test("Closure tools work the same as conforming types")
    func closureTools() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "echo", input: ["city": "Lund"]), .text("done")]
        )
        let echo = tool("echo", description: "Echo the city.") { (query: WeatherQuery, _) in
            .text("echo: \(query.city)")
        }

        let result = try await generateText(
            model: model,
            prompt: "Echo Lund",
            tools: [echo],
            stopWhen: [.stepCount(3)]
        )
        #expect(result.steps[0].toolResults.first?.output == .text("echo: Lund"))
    }

    @Test("Dynamic tools receive raw JSON")
    func dynamicTools() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "search", input: ["query": "swift"]), .text("done")]
        )
        let search = dynamicTool(
            "search",
            description: "Search.",
            inputSchema: .object(properties: ["query": .string()], required: ["query"])
        ) { input, _ in
            .text("found \(input["query"]?.stringValue ?? "")")
        }

        let result = try await generateText(
            model: model,
            prompt: "Search",
            tools: [search],
            stopWhen: [.stepCount(3)]
        )
        #expect(result.steps[0].toolResults.first?.output == .text("found swift"))
    }

    @Test("Runs several tool calls concurrently and preserves their order")
    func runsToolsConcurrentlyInOrder() async throws {
        let recorder = Recorder()
        let model = MockLanguageModel(
            responses: [
                .toolCalls([
                    ToolCallPart(toolCallID: "1", toolName: "weather", input: ["city": "Malmö"]),
                    ToolCallPart(toolCallID: "2", toolName: "weather", input: ["city": "Lund"]),
                    ToolCallPart(toolCallID: "3", toolName: "weather", input: ["city": "Ystad"]),
                ]),
                .text("done"),
            ]
        )

        let result = try await generateText(
            model: model,
            prompt: "Weather in three cities?",
            tools: [WeatherTool(recorder: recorder)],
            stopWhen: [.stepCount(3)]
        )

        // Results are reordered to match the order the model requested them.
        #expect(result.steps[0].toolResults.map(\.toolCallID) == ["1", "2", "3"])
        #expect(recorder.recorded.count == 3)
    }

    @Test("A thrown tool error becomes a result the model can read")
    func toolErrorBecomesResult() async throws {
        struct Unavailable: Error {}
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("Sorry, I could not check.")]
        )
        let failing = tool("weather", description: "Look up the weather.") { (_: WeatherQuery, _) in
            throw Unavailable()
        }

        // The generation succeeds: the model is told the tool failed and answers anyway.
        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [failing],
            stopWhen: [.stepCount(3)]
        )

        #expect(result.text == "Sorry, I could not check.")
        let output = try #require(result.steps[0].toolResults.first?.output)
        #expect(output.isError)
    }

    @Test("Invalid tool arguments become an error result rather than a failure")
    func invalidToolArgumentsBecomeResult() async throws {
        let model = MockLanguageModel(
            responses: [
                // `city` is required but missing.
                .toolCall(name: "weather", input: ["town": "Malmö"]),
                .text("Let me try again."),
            ]
        )

        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)]
        )

        let output = try #require(result.steps[0].toolResults.first?.output)
        #expect(output.isError)
        // The message names the property so the model can correct itself.
        if case .errorText(let text) = output {
            #expect(text.contains("city"))
        }
    }

    @Test("An unknown tool becomes an error result listing what is available")
    func unknownToolBecomesResult() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "stocks", input: [:]), .text("I cannot do that.")]
        )

        let result = try await generateText(
            model: model,
            prompt: "Stock price?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)]
        )

        let output = try #require(result.steps[0].toolResults.first?.output)
        #expect(output.isError)
        if case .errorText(let text) = output {
            #expect(text.contains("stocks"))
            #expect(text.contains("weather"))
        }
    }

    @Test("A client-executed tool stops the loop")
    func clientToolStopsLoop() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "confirm", input: ["amount": 42]), .text("never reached")]
        )
        let confirm = clientTool(
            "confirm",
            description: "Ask the user to confirm.",
            inputSchema: .object(properties: ["amount": .number()], required: ["amount"])
        )

        let result = try await generateText(
            model: model,
            prompt: "Buy it",
            tools: [confirm],
            stopWhen: [.stepCount(5)]
        )

        #expect(model.recordedCalls.count == 1)
        #expect(result.finishReason == .toolCalls)
        #expect(result.toolCalls.first?.toolName == "confirm")
        // Nothing ran locally, so there is nothing to feed back.
        #expect(result.toolResults.isEmpty)
    }

    @Test("Tools receive the conversation and the call identifier")
    func toolReceivesContext() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(id: "call_7", name: "inspect", input: [:]), .text("done")]
        )
        let captured = Recorder()
        let inspect = tool("inspect", description: "Inspect the context.") { (_: JSONValue, context) in
            captured.record(context.toolCallID)
            captured.record("messages:\(context.messages.count)")
            captured.record("step:\(context.stepNumber)")
            return .text("ok")
        }

        _ = try await generateText(model: model, prompt: "Hi", tools: [inspect], stopWhen: [.stepCount(3)])
        #expect(captured.recorded == ["call_7", "messages:1", "step:0"])
    }

    // MARK: Stop conditions

    @Test("A step budget bounds a model that keeps calling tools")
    func stepBudgetBoundsTheLoop() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"])],
            repeatsLastResponse: true
        )

        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)]
        )
        #expect(result.steps.count == 3)
    }

    @Test("Stops when a designated tool has been called")
    func stopsOnToolCall() async throws {
        let model = MockLanguageModel(
            responses: [
                .toolCall(name: "weather", input: ["city": "Malmö"]),
                .toolCall(id: "call_2", name: "submit", input: [:]),
                .text("never reached"),
            ]
        )
        let submit = tool("submit", description: "Submit the final answer.") { (_: JSONValue, _) in
            .text("submitted")
        }

        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder()), submit],
            stopWhen: [.hasToolCall("submit"), .stepCount(10)]
        )
        #expect(result.steps.count == 2)
    }

    @Test("Stops when the token budget is reached")
    func stopsOnTokenBudget() async throws {
        let model = MockLanguageModel(
            responses: [
                .toolCall(name: "weather", input: ["city": "Malmö"], usage: Usage(inputTokens: 60, outputTokens: 40))
            ],
            repeatsLastResponse: true
        )

        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.totalTokens(150), .stepCount(10)]
        )
        // Two steps reach 200 tokens, which crosses the budget.
        #expect(result.steps.count == 2)
    }

    @Test("Conditions combine with logical or")
    func conditionsCombineWithOr() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"])],
            repeatsLastResponse: true
        )
        let never = StopCondition { _ in false }

        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [never, .stepCount(2)]
        )
        #expect(result.steps.count == 2)
    }

    // MARK: Step preparation

    @Test("Restricts the tools offered in a step")
    func restrictsActiveTools() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("done")]
        )
        let other = tool("other", description: "Something else.") { (_: JSONValue, _) in .text("x") }

        _ = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder()), other],
            stopWhen: [.stepCount(3)],
            prepareStep: { context in
                context.stepNumber == 0 ? nil : PrepareStepAdjustments(activeTools: ["other"])
            }
        )

        #expect(model.recordedOptions(at: 0)?.tools.count == 2)
        // The second step only sees the tool it was restricted to.
        #expect(model.recordedOptions(at: 1)?.tools.map(\.name) == ["other"])
    }

    @Test("Replaces the conversation for a step")
    func replacesMessagesForAStep() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("done")]
        )

        _ = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)],
            prepareStep: { context in
                guard context.stepNumber == 1 else { return nil }
                // Compressing history is how a long-running agent stays inside its context window.
                return PrepareStepAdjustments(messages: [.user("Summarized history.")])
            }
        )

        let secondPrompt = try #require(model.recordedOptions(at: 1)?.prompt)
        #expect(secondPrompt.count == 1)
        #expect(secondPrompt.first?.text == "Summarized history.")
    }

    @Test("Overrides settings for a step")
    func overridesSettingsForAStep() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("done")]
        )

        _ = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            settings: GenerationSettings(temperature: 1.0),
            stopWhen: [.stepCount(3)],
            prepareStep: { context in
                context.stepNumber == 1 ? PrepareStepAdjustments(settings: GenerationSettings(temperature: 0)) : nil
            }
        )

        #expect(model.recordedOptions(at: 0)?.temperature == 1.0)
        #expect(model.recordedOptions(at: 1)?.temperature == 0)
    }

    @Test("Switches models between steps")
    func switchesModelsBetweenSteps() async throws {
        let planner = MockLanguageModel(
            modelID: "planner",
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"])]
        )
        let writer = MockLanguageModel(modelID: "writer", responses: [.text("done")])

        let result = try await generateText(
            model: planner,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)],
            prepareStep: { context in
                context.stepNumber == 1 ? PrepareStepAdjustments(model: writer) : nil
            }
        )

        #expect(planner.recordedCalls.count == 1)
        #expect(writer.recordedCalls.count == 1)
        #expect(result.text == "done")
    }

    // MARK: Results

    @Test("Reports per-step and total usage")
    func reportsUsage() async throws {
        let model = MockLanguageModel(
            responses: [
                .toolCall(name: "weather", input: ["city": "Malmö"], usage: Usage(inputTokens: 10, outputTokens: 5)),
                .text("done", usage: Usage(inputTokens: 30, outputTokens: 8)),
            ]
        )

        let result = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)]
        )

        // `usage` is the final step; `totalUsage` is what the whole call cost.
        #expect(result.usage.inputTokens == 30)
        #expect(result.totalUsage.inputTokens == 40)
        #expect(result.totalUsage.outputTokens == 13)
    }

    @Test("Response messages continue the conversation faithfully")
    func responseMessagesContinueConversation() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("Yes.")]
        )

        let result = try await generateText(
            model: model,
            prompt: "Umbrella?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)]
        )

        // Assistant tool call, tool result, then the final assistant answer.
        #expect(result.responseMessages.map(\.role) == [.assistant, .tool, .assistant])

        let followUp = MockLanguageModel(text: "Still yes.")
        var history: [ModelMessage] = [.user("Umbrella?")]
        history += result.responseMessages
        history.append(.user("And tomorrow?"))
        _ = try await generateText(model: followUp, prompt: Prompt(history))

        #expect(followUp.recordedOptions()?.prompt.count == 5)
    }

    @Test("Calls the step callback after every step")
    func callsStepCallback() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"]), .text("done")]
        )
        let observed = Recorder()

        _ = try await generateText(
            model: model,
            prompt: "Weather?",
            tools: [WeatherTool(recorder: Recorder())],
            stopWhen: [.stepCount(3)],
            onStepFinish: { step in observed.record(step.finishReason.rawValue) }
        )
        #expect(observed.recorded == ["tool-calls", "stop"])
    }

    // MARK: Cancellation

    @Test("Cancellation stops the loop")
    func cancellationStopsTheLoop() async throws {
        let model = MockLanguageModel(
            responses: [.toolCall(name: "weather", input: ["city": "Malmö"])],
            repeatsLastResponse: true
        )
        model.onCall = { _ in try await Task.sleep(for: .milliseconds(20)) }

        let task = Task {
            try await generateText(
                model: model,
                prompt: "Weather?",
                tools: [WeatherTool(recorder: Recorder())],
                stopWhen: [.stepCount(100)]
            )
        }
        try await Task.sleep(for: .milliseconds(30))
        task.cancel()

        await #expect(throws: CancellationError.self) { try await task.value }
    }
}

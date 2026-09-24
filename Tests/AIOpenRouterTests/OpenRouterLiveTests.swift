import AIProviderSpec
import Foundation
import SwiftAI
import Testing

@testable import AIOpenRouter

@Structured("Where to look up the weather.")
struct LiveWeatherQuery {
    @Guidance("The city name.")
    var city: String
}

/// End-to-end checks against the real API, for what fixtures cannot prove: that an upstream
/// *accepts* what this provider sends back — replayed reasoning signatures above all.
///
/// Off by default, since they spend credits and need the network. Run them with:
///
/// ```bash
/// OPENROUTER_LIVE_TESTS=1 OPENROUTER_API_KEY=… swift test --filter OpenRouterLiveTests
/// ```
@Suite(
    "OpenRouter live",
    .enabled(if: ProcessInfo.processInfo.environment["OPENROUTER_LIVE_TESTS"] == "1"),
    .serialized
)
struct OpenRouterLiveTests {
    private let openrouter = OpenRouterProvider(appName: "SwiftAI live tests")

    private var weather: some Tool {
        tool("weather", description: "Look up the current weather in a city.") { (query: LiveWeatherQuery, _) in
            .text("It is 14°C and raining in \(query.city).")
        }
    }

    private var reasoning: GenerationSettings {
        var settings = GenerationSettings(maxOutputTokens: 2048)
        settings.providerOptions = ["openrouter": ["reasoning": ["max_tokens": 1024]]]
        return settings
    }

    /// The details the second step's request replayed, so a test can tell "accepted the replay"
    /// apart from "accepted a request that silently dropped it".
    private func replayedDetails(_ steps: [StepResult]) throws -> [JSONValue] {
        let body = try #require(steps.dropFirst().first?.request?.body)
        let messages = try #require(try JSONValue.parse(body)["messages"]?.arrayValue)
        let assistant = try #require(messages.last { $0["role"] == "assistant" })
        return assistant["reasoning_details"]?.arrayValue ?? []
    }

    @Test("Anthropic accepts replayed signed reasoning across a buffered tool loop")
    func bufferedToolLoopWithReasoning() async throws {
        let result = try await generateText(
            model: openrouter.languageModel("anthropic/claude-haiku-4.5"),
            prompt: "What is the weather in Malmö? Use the tool.",
            tools: [weather],
            settings: reasoning,
            stopWhen: [.stepCount(3)]
        )
        #expect(result.steps.count >= 2)
        #expect(result.text.contains("14"))
        #expect(try replayedDetails(result.steps).contains { $0["signature"]?.stringValue?.isEmpty == false })
    }

    @Test("Anthropic accepts replayed signed reasoning across a streamed tool loop")
    func streamedToolLoopWithReasoning() async throws {
        let stream = streamText(
            model: openrouter.languageModel("anthropic/claude-haiku-4.5"),
            prompt: "What is the weather in Oslo? Use the tool.",
            tools: [weather],
            settings: reasoning,
            stopWhen: [.stepCount(3)]
        )
        #expect(try await stream.steps.count >= 2)
        #expect(try await stream.text.contains("14"))
        #expect(try replayedDetails(try await stream.steps).contains { $0["signature"]?.stringValue?.isEmpty == false })
    }

    @Test("OpenAI accepts replayed encrypted reasoning across a streamed tool loop")
    func streamedToolLoopWithEncryptedReasoning() async throws {
        var settings = GenerationSettings()
        settings.providerOptions = ["openrouter": ["reasoning": ["effort": "low"]]]
        let stream = streamText(
            model: openrouter.languageModel("openai/gpt-5-nano"),
            prompt: "What is the weather in Bergen? Use the tool.",
            tools: [weather],
            settings: settings,
            stopWhen: [.stepCount(3)]
        )
        #expect(try await stream.steps.count >= 2)
        #expect(try replayedDetails(try await stream.steps).contains { $0["type"] == "reasoning.encrypted" })
    }

    @Test("A decision model answers natively")
    func decisionModel() async throws {
        let result = try await evaluate(
            model: openrouter.decisionModel("typesafe/jev-1.13"),
            state: "I was charged twice. Please refund the extra charge.",
            questions: [
                "department": .choice("Which team should handle this?", options: ["billing": "Payments", "support": nil]),
                "refund": .boolean("Is the customer asking for money back?"),
            ]
        )
        #expect(result["department"]?.choice == "billing")
        #expect((result["refund"]?.probability ?? 0) > 0.5)
    }

    @Test("The language model adapter answers through OpenRouter's structured output")
    func languageModelAdapter() async throws {
        let model = LanguageModelEvaluationModel(model: openrouter.languageModel("openai/gpt-4.1-nano"))
        let result = try await evaluate(
            model: model,
            state: "I was charged twice. Please refund the extra charge.",
            questions: [
                "department": .choice("Which team should handle this?", options: ["billing": "Payments", "support": nil]),
                "severity": .score("How severe is it?", levels: ["Cosmetic", "Annoying", "Blocking"]),
                "refund": .boolean("Is the customer asking for money back?"),
            ]
        )
        #expect(result["department"]?.choice == "billing")
        #expect((result["refund"]?.probability ?? 0) > 0.5)
    }
}

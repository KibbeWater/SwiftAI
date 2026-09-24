import AIProviderSpec
import AITestSupport
import Foundation
import Testing

@testable import AIProviderUtils

@Suite("Language model evaluation adapter")
struct LanguageModelEvaluationModelTests {
    private let questions: EvaluationQuestions = [
        "department": .choice("Which team?", options: ["billing": "Payments", "support": nil]),
        "severity": .score("How severe?", levels: ["Low", "Medium", "High"]),
        "refund": .boolean("Refund requested?", whenTrue: "Asks for money back"),
    ]

    private func evaluate(
        replying text: String,
        finishReason: FinishReason = .stop,
        defaults: ProviderOptions? = nil,
        providerOptions: ProviderOptions? = nil
    ) async throws -> (EvaluationModelResponse, MockLanguageModel) {
        let language = MockLanguageModel(responses: [.text(text, finishReason: finishReason)])
        let model = LanguageModelEvaluationModel(model: language, defaultProviderOptions: defaults)
        let response = try await model.evaluate(
            EvaluationModelCallOptions(state: ["message": "Charged twice"], questions: questions, providerOptions: providerOptions)
        )
        return (response, language)
    }

    @Test("Maps positional codes back to the caller's identifiers and option names")
    func mapsCodes() async throws {
        let (response, _) = try await evaluate(replying: #"{"q0":"c0","q1":1.5,"q2":0.9}"#)

        #expect(response.answers == [
            "department": .choice("billing"),
            "severity": .score(1.5),
            "refund": .boolean(probability: 0.9),
        ])
    }

    @Test("Requests a closed schema with one property per question")
    func requestsSchema() async throws {
        let (_, language) = try await evaluate(replying: #"{"q0":"c1","q1":0,"q2":0}"#)
        let options = try #require(language.recordedOptions())

        guard case .json(let schema?, let name, _) = options.responseFormat else {
            Issue.record("Expected a JSON response format with a schema")
            return
        }
        #expect(name == "evaluation")
        let encoded = schema.jsonValue()
        #expect(encoded["required"] == ["q0", "q1", "q2"])
        #expect(encoded["additionalProperties"] == false)
        #expect(encoded["properties"]?["q0"]?["enum"] == ["c0", "c1"])
        // Bounds are described rather than declared: several providers reject `minimum`/`maximum`.
        #expect(encoded["properties"]?["q1"]?["maximum"] == nil)
    }

    @Test("Sends the state and ordered rubrics as JSON")
    func sendsRubrics() async throws {
        let (_, language) = try await evaluate(replying: #"{"q0":"c1","q1":0,"q2":0}"#)
        let prompt = try #require(language.recordedOptions()?.prompt)

        guard case .user(let user) = prompt.last, case .text(let text) = user.content.first else {
            Issue.record("Expected a user text message")
            return
        }
        let payload = try JSONValue.parse(text.text)
        #expect(payload["state"] == ["message": "Charged twice"])
        #expect(payload["questions"]?["q0"]?["id"] == "department")
        #expect(payload["questions"]?["q0"]?["criteria"]?["c0"] == ["label": "billing", "description": "Payments"])
        #expect(payload["questions"]?["q0"]?["criteria"]?["c1"]?["description"] == .null)
        #expect(payload["questions"]?["q2"]?["criteria"] == ["true": "Asks for money back"])
        #expect(text.text.range(of: "\"q0\"")!.lowerBound < text.text.range(of: "\"q2\"")!.lowerBound)
    }

    @Test("Lets the caller's provider options override the defaults")
    func mergesProviderOptions() async throws {
        let (_, language) = try await evaluate(
            replying: #"{"q0":"c1","q1":0,"q2":0}"#,
            defaults: ["openai": ["reasoning": ["effort": "none"], "store": false]],
            providerOptions: ["openai": ["reasoning": ["effort": "high"]]]
        )
        let options = try #require(language.recordedOptions()?.providerOptions)

        #expect(options.value("reasoning", for: "openai") == ["effort": "high"])
        #expect(options.value("store", for: "openai") == false)
    }

    @Test("Rejects output that is not a clean answer", arguments: [
        #"{"q0":"c2","q1":0,"q2":0}"#,  // An option code that does not exist.
        #"{"q0":"billing","q1":0,"q2":0}"#,  // The label instead of the code.
        #"{"q0":"c0","q1":3,"q2":0}"#,  // A score past the top level.
        #"{"q0":"c0","q1":0,"q2":1.5}"#,  // A probability above one.
        #"{"q0":"c0","q1":0}"#,  // A missing question.
        #"not json"#,
    ])
    func rejectsBadOutput(text: String) async throws {
        await #expect(throws: InvalidResponseDataError.self) {
            try await evaluate(replying: text)
        }
    }

    @Test("Rejects a response that did not finish cleanly")
    func rejectsTruncation() async throws {
        // Truncated JSON can still parse when the cut falls between values, so the finish reason
        // is the only reliable signal.
        await #expect(throws: InvalidResponseDataError.self) {
            try await evaluate(replying: #"{"q0":"c0","q1":0,"q2":0}"#, finishReason: .length)
        }
    }
}

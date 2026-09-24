import AITestSupport
import Foundation
import Testing

@testable import SwiftAI

@Suite("evaluate")
struct EvaluationTests {
    private let triage: EvaluationQuestions = [
        "department": .choice("Which team should handle this?", options: ["billing": "Payments", "support": nil]),
        "severity": .score("How severe is it?", levels: ["Cosmetic", "Workaround exists", "Blocking"]),
        "refund": .boolean("Is the customer asking for money back?"),
    ]

    private func evaluateTriage(
        _ answers: [String: EvaluationAnswer],
        rounding: EvaluationRounding? = nil
    ) async throws -> EvaluationResult {
        try await evaluate(
            model: MockEvaluationModel(answers: answers, rounding: rounding),
            state: "I was charged twice.",
            questions: triage,
            retryPolicy: .none
        )
    }

    private let validAnswers: [String: EvaluationAnswer] = [
        "department": .choice("billing"),
        "severity": .score(1.5),
        "refund": .boolean(probability: 0.97),
    ]

    // MARK: - Results

    @Test("Returns one answer per question")
    func returnsAnswers() async throws {
        let result = try await evaluateTriage(validAnswers)

        #expect(result["department"]?.choice == "billing")
        #expect(result["severity"]?.score == 1.5)
        #expect(result["refund"]?.probability == 0.97)
    }

    @Test("Totals usage only when both input and output are known")
    func totalsUsage() async throws {
        let known = try await evaluate(
            model: MockEvaluationModel(answers: ["refund": .boolean(probability: 1)], usage: Usage(inputTokens: 40, outputTokens: 2)),
            state: "x",
            questions: ["refund": .boolean("Refund?")]
        )
        #expect(known.usage.totalTokens == 42)

        // An unknown output count must not be read as zero, or the total would undercount.
        let partial = try await evaluate(
            model: MockEvaluationModel(answers: ["refund": .boolean(probability: 1)], usage: Usage(inputTokens: 40)),
            state: "x",
            questions: ["refund": .boolean("Refund?")]
        )
        #expect(partial.usage.totalTokens == nil)
    }

    @Test("Fills in the response timestamp and model identifier")
    func fillsResponseInfo() async throws {
        let result = try await evaluateTriage(validAnswers)
        #expect(result.response.modelID == "mock-evaluation")
        #expect(result.response.timestamp != nil)
    }

    @Test("Passes questions to the model in the order they were written")
    func preservesQuestionOrder() async throws {
        let model = MockEvaluationModel()
        _ = try await evaluate(model: model, state: "x", questions: triage)

        #expect(model.recordedCalls.first?.questions.ids == ["department", "severity", "refund"])
    }

    @Test("Answers a typed choice with a case of the enumeration")
    func typedChoice() async throws {
        enum Department: String, EvaluationChoice {
            case billing, support
            var evaluationDescription: JSONValue? { self == .billing ? "Payments" : nil }
        }

        let model = MockEvaluationModel(answers: ["department": .choice("support")])
        let result = try await evaluate(
            model: model,
            state: "The app crashes on launch.",
            questions: ["department": .choice("Which team?", options: Department.self)]
        )

        #expect(result.choice("department", as: Department.self) == .support)
        #expect(model.recordedCalls.first?.questions["department"] == .choice(
            instructions: "Which team?",
            options: [EvaluationOption("billing", description: "Payments"), EvaluationOption("support")]
        ))
    }

    // MARK: - Input validation

    @Test("Rejects a question type the model does not support before calling it")
    func rejectsUnsupportedType() async throws {
        let model = MockEvaluationModel(supportedQuestionTypes: [.choice, .score])

        await #expect(throws: UnsupportedQuestionTypeError.self) {
            try await evaluate(model: model, state: "x", questions: triage)
        }
        #expect(model.recordedCalls.isEmpty)
    }

    @Test("Rejects malformed questions", arguments: [
        EvaluationQuestions(),
        ["levels": .score("One level is not a rubric", levels: ["Only"])],
        ["options": .choice(instructions: "No options", options: [])],
        ["options": .choice(instructions: "Twice", options: [EvaluationOption("a"), EvaluationOption("a")])],
        ["instructions": .boolean(instructions: .int(3))],
        ["description": .boolean("Refund?", whenTrue: .double(.nan))],
    ])
    func rejectsMalformedQuestions(questions: EvaluationQuestions) async throws {
        let model = MockEvaluationModel()
        await #expect(throws: InvalidArgumentError.self) {
            try await evaluate(model: model, state: "x", questions: questions)
        }
        #expect(model.recordedCalls.isEmpty)
    }

    @Test("Rejects a bare number as state")
    func rejectsScalarState() async throws {
        await #expect(throws: InvalidArgumentError.self) {
            try await evaluate(model: MockEvaluationModel(), state: 42, questions: ["refund": .boolean("Refund?")])
        }
    }

    // MARK: - Answer validation

    @Test("Rejects answers that break the contract", arguments: [
        // A missing answer.
        ["department": .choice("billing"), "severity": .score(1)],
        // An extra answer.
        ["department": .choice("billing"), "severity": .score(1), "refund": .boolean(probability: 1), "other": .score(0)],
        // The wrong kind of answer.
        ["department": .score(0), "severity": .score(1), "refund": .boolean(probability: 1)],
        // An option that was not offered.
        ["department": .choice("sales"), "severity": .score(1), "refund": .boolean(probability: 1)],
        // A score off the rubric.
        ["department": .choice("billing"), "severity": .score(2.5), "refund": .boolean(probability: 1)],
        // A probability above one.
        ["department": .choice("billing"), "severity": .score(1), "refund": .boolean(probability: 1.2)],
        // A choice that is not the most likely option.
        ["department": .choice("billing", probabilities: ["billing": 0.3, "support": 0.7]), "severity": .score(1), "refund": .boolean(probability: 1)],
        // An incomplete distribution.
        ["department": .choice("billing", probabilities: ["billing": 1]), "severity": .score(1), "refund": .boolean(probability: 1)],
        // A score that disagrees with its distribution.
        ["department": .choice("billing"), "severity": .score(0, probabilities: [0: 0, 1: 0, 2: 1]), "refund": .boolean(probability: 1)],
    ] as [[String: EvaluationAnswer]])
    func rejectsInvalidAnswers(answers: [String: EvaluationAnswer]) async throws {
        await #expect(throws: InvalidResponseDataError.self) {
            try await evaluateTriage(answers)
        }
    }

    @Test("Accepts a tied distribution")
    func acceptsTies() async throws {
        var answers = validAnswers
        answers["department"] = .choice("support", probabilities: ["billing": 0.5, "support": 0.5])
        _ = try await evaluateTriage(answers)
    }

    @Test("Allows for the rounding a provider declares")
    func allowsDeclaredRounding() async throws {
        // Recorded from TypeSafe's API: probabilities and score rounded to two places. The
        // weighted mean is 0.98, not 0.97, and only the declared rounding reconciles the two.
        var answers = validAnswers
        answers["severity"] = .score(0.97, probabilities: [0: 0.13, 1: 0.76, 2: 0.11])

        _ = try await evaluateTriage(answers, rounding: EvaluationRounding(probabilityDecimals: 2, scoreDecimals: 2))
        await #expect(throws: InvalidResponseDataError.self) {
            try await evaluateTriage(answers)
        }
    }

    @Test("Allows a rounded distribution to sum slightly off one")
    func allowsRoundedSum() async throws {
        var answers = validAnswers
        answers["department"] = .choice("billing", probabilities: ["billing": 0.67, "support": 0.34])

        _ = try await evaluateTriage(answers, rounding: EvaluationRounding(probabilityDecimals: 2))
        await #expect(throws: InvalidResponseDataError.self) {
            try await evaluateTriage(answers)
        }
    }

    // MARK: - Retries

    @Test("Retries a transient provider failure")
    func retriesTransientFailure() async throws {
        let attempts = Counter()
        let model = MockEvaluationModel(respond: { _ in
            if attempts.increment() == 1 {
                throw APICallError(message: "Overloaded", url: URL(string: "https://example.com")!, statusCode: 503)
            }
            return EvaluationModelResponse(answers: ["refund": .boolean(probability: 0.1)])
        })

        let result = try await evaluate(
            model: model,
            state: "x",
            questions: ["refund": .boolean("Refund?")],
            retryPolicy: RetryPolicy(maximumRetries: 1, initialDelay: .zero)
        )
        #expect(result["refund"]?.probability == 0.1)
        #expect(attempts.value == 2)
    }

    @Test("Does not retry an answer that fails validation")
    func doesNotRetryInvalidAnswers() async throws {
        // Asking again rarely changes the format a model answers in, so it would only spend quota.
        let model = MockEvaluationModel(answers: ["refund": .boolean(probability: 7)])
        await #expect(throws: InvalidResponseDataError.self) {
            try await evaluate(
                model: model,
                state: "x",
                questions: ["refund": .boolean("Refund?")],
                retryPolicy: RetryPolicy(maximumRetries: 3, initialDelay: .zero)
            )
        }
        #expect(model.recordedCalls.count == 1)
    }

    // MARK: - Resolution

    @Test("Resolves evaluation models through a registry and a custom provider")
    func resolvesThroughRegistry() throws {
        let native = MockEvaluationModel(modelID: "native")
        let registry = ProviderRegistry([
            "app": CustomProvider(evaluationModels: ["triage": native]),
            "mock": MockProvider(evaluation: MockEvaluationModel(modelID: "fallback")),
        ])

        #expect(try registry.evaluationModel("app:triage").modelID == "native")
        #expect(try registry.evaluationModel("mock:anything").modelID == "fallback")
        #expect(throws: NoSuchModelError.self) { try registry.evaluationModel("app:missing") }
    }

    @Test("A provider without evaluation models reports the evaluation kind")
    func defaultThrowsNoSuchModel() throws {
        do {
            _ = try BareProvider().evaluationModel("jev")
            Issue.record("Expected NoSuchModelError")
        } catch let error as NoSuchModelError {
            #expect(error.modelKind == .evaluation)
        }
    }
}

/// A provider that implements nothing, to exercise the protocol's defaults.
private struct BareProvider: AIProvider {
    let name = "bare"
}

/// A thread-safe counter for asserting on attempts.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int { lock.withLock { count } }

    func increment() -> Int {
        lock.withLock {
            count += 1
            return count
        }
    }
}

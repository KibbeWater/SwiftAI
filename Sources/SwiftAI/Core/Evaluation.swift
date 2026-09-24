import AIProviderSpec
import AIProviderUtils
import Foundation

// MARK: - Result

/// The result of an evaluation.
///
/// - Important: Experimental. Evaluation may change in a minor release.
public struct EvaluationResult: Sendable {
    /// One answer per question, keyed by the question's identifier.
    public var answers: [String: EvaluationAnswer]

    /// Token usage. ``Usage/totalTokens`` is set only when both input and output are known.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// How the provider rounded its output, or `nil` for full precision.
    public var rounding: EvaluationRounding?

    /// Provider-specific values, such as OpenRouter's per-question confidence.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about the response. The timestamp and model identifier are always filled in.
    public var response: ResponseInfo

    /// The answer to a question.
    public subscript(id: String) -> EvaluationAnswer? { answers[id] }

    /// The option chosen for a choice question asked with an ``EvaluationChoice`` type.
    ///
    /// Returns `nil` when the question is unknown or is not a choice question.
    public func choice<Option: EvaluationChoice>(_ id: String, as type: Option.Type = Option.self) -> Option? {
        answers[id]?.choice.flatMap(Option.init(rawValue:))
    }
}

// MARK: - Typed choices

/// A Swift enumeration whose cases are the options of a choice question.
///
/// Asking with the type rather than with string names means the answer comes back as a case, and
/// a renamed option is a compile error instead of a silent mismatch.
///
/// ```swift
/// enum Department: String, EvaluationChoice {
///     case billing, support
///
///     var evaluationDescription: JSONValue? {
///         switch self {
///         case .billing: "Payments and refunds"
///         case .support: "Everything else"
///         }
///     }
/// }
///
/// let result = try await evaluate(
///     model: model,
///     state: "I was charged twice.",
///     questions: ["department": .choice("Which team should handle this?", options: Department.self)]
/// )
/// let department = result.choice("department", as: Department.self)  // .billing
/// ```
public protocol EvaluationChoice: RawRepresentable, CaseIterable, Sendable where RawValue == String {
    /// What this option means, shown to the model. Defaults to `nil`, leaving the name to speak for
    /// itself.
    var evaluationDescription: JSONValue? { get }
}

extension EvaluationChoice {
    public var evaluationDescription: JSONValue? { nil }
}

extension EvaluationQuestion {
    /// A choice question whose options are the cases of `options`, in declaration order.
    public static func choice<Option: EvaluationChoice>(
        _ instructions: JSONValue,
        options: Option.Type
    ) -> EvaluationQuestion {
        .choice(
            instructions: instructions,
            options: Option.allCases.map { EvaluationOption($0.rawValue, description: $0.evaluationDescription) }
        )
    }
}

// MARK: - Evaluate

/// Answers named questions about one shared state.
///
/// ```swift
/// let result = try await evaluate(
///     model: openrouter.decisionModel("openrouter/auto"),
///     state: ["message": "I was charged twice. Please refund the extra charge."],
///     questions: [
///         "department": .choice(
///             "Which team should handle this?",
///             options: ["billing": "Payments and refunds", "support": "Other requests"]
///         ),
///         "severity": .score(
///             "How severe is the issue?",
///             levels: ["Cosmetic", "Workaround exists", "Blocking; no workaround"]
///         ),
///         "refund": .boolean("Is the customer asking for money back?"),
///     ]
/// )
///
/// if let p = result["refund"]?.probability, p >= 0.8 { routeToRefunds() }
/// ```
///
/// Questions are checked before any request, and answers are checked against them afterwards:
/// every question gets exactly one answer of its own kind, choices name an offered option,
/// distributions are complete and sum to one, and scores stay on their rubric and agree with their
/// distribution. Nothing is renormalized; output that breaks these rules is rejected.
///
/// Only the provider call is retried. A response that fails validation is not, since asking again
/// is unlikely to change a model's mind about the format it returns.
///
/// - Important: Experimental. Evaluation may change in a minor release.
///
/// - Parameters:
///   - model: The evaluation model.
///   - state: What the questions are about: a string, an object, or an array.
///   - questions: The questions, keyed by identifier.
///   - retryPolicy: How transient failures are retried.
///   - headers: Extra HTTP headers.
///   - providerOptions: Provider-specific settings.
/// - Returns: One answer per question, with usage and metadata.
/// - Throws: ``InvalidArgumentError`` for malformed questions, ``UnsupportedQuestionTypeError``
///   when the model cannot answer a question's kind, ``InvalidResponseDataError`` for answers that
///   break the rules above, and ``APICallError`` for upstream failures.
public func evaluate(
    model: any EvaluationModel,
    state: JSONValue,
    questions: EvaluationQuestions,
    retryPolicy: RetryPolicy = .default,
    headers: [String: String] = [:],
    providerOptions: ProviderOptions? = nil
) async throws -> EvaluationResult {
    try EvaluationValidation.validateInput(state: state, questions: questions)

    for entry in questions where !model.supportedQuestionTypes.contains(entry.question.type) {
        throw UnsupportedQuestionTypeError(
            questionID: entry.id,
            questionType: entry.question.type,
            provider: model.provider,
            modelID: model.modelID
        )
    }

    let response = try await withRetries(policy: retryPolicy) { _ in
        try await model.evaluate(
            EvaluationModelCallOptions(
                state: state,
                questions: questions,
                headers: headers,
                providerOptions: providerOptions
            )
        )
    }
    try Task.checkCancellation()
    try EvaluationValidation.validateAnswers(response.answers, for: questions, rounding: response.rounding)

    var usage = response.usage
    if usage.totalTokens == nil, let input = usage.inputTokens, let output = usage.outputTokens {
        usage.totalTokens = input + output
    }

    var responseInfo = response.response ?? ResponseInfo()
    responseInfo.timestamp = responseInfo.timestamp ?? Date()
    responseInfo.modelID = responseInfo.modelID ?? model.modelID

    return EvaluationResult(
        answers: response.answers,
        usage: usage,
        warnings: response.warnings,
        rounding: response.rounding,
        providerMetadata: response.providerMetadata,
        response: responseInfo
    )
}

// MARK: - Validation

/// The rules evaluation inputs and answers must satisfy.
///
/// Public so that code calling an ``EvaluationModel`` directly, bypassing `evaluate`, can hold its
/// answers to the same standard.
public enum EvaluationValidation {
    /// The absolute slack allowed on sums and weighted means, before any rounding allowance.
    public static let tolerance = 1e-6

    /// Checks that the state and questions are well formed.
    ///
    /// - Throws: ``InvalidArgumentError`` naming the offending question.
    public static func validateInput(state: JSONValue, questions: EvaluationQuestions) throws {
        guard isInput(state) else {
            throw InvalidArgumentError(argument: "state", message: "The state must be a string, object, or array of finite JSON values.")
        }
        guard !questions.isEmpty else {
            throw InvalidArgumentError(argument: "questions", message: "An evaluation needs at least one question.")
        }

        for entry in questions {
            let argument = "questions.\(entry.id)"
            guard isInput(entry.question.instructions) else {
                throw InvalidArgumentError(
                    argument: argument,
                    message: "Instructions must be a string, object, or array of finite JSON values."
                )
            }

            let descriptions: [JSONValue?]
            switch entry.question {
            case .choice(_, let options):
                guard !options.isEmpty else {
                    throw InvalidArgumentError(argument: argument, message: "A choice question needs at least one option.")
                }
                guard Set(options.map(\.name)).count == options.count else {
                    throw InvalidArgumentError(argument: argument, message: "A choice question's option names must be unique.")
                }
                descriptions = options.map(\.description)

            case .score(_, let levels):
                guard levels.count >= 2 else {
                    throw InvalidArgumentError(argument: argument, message: "A score question needs at least two ordered levels.")
                }
                descriptions = levels

            case .boolean(_, let whenTrue, let whenFalse):
                descriptions = [whenTrue, whenFalse]
            }

            for description in descriptions {
                guard let description else { continue }
                guard isInput(description) else {
                    throw InvalidArgumentError(
                        argument: argument,
                        message: "Descriptions must be strings, objects, or arrays of finite JSON values, or nil."
                    )
                }
            }
        }
    }

    /// Checks a provider's answers against the questions they answer.
    ///
    /// - Throws: ``InvalidResponseDataError`` describing the first rule broken.
    public static func validateAnswers(
        _ answers: [String: EvaluationAnswer],
        for questions: EvaluationQuestions,
        rounding: EvaluationRounding?
    ) throws {
        let probabilityError = try roundingError(rounding?.probabilityDecimals)
        let scoreError = try roundingError(rounding?.scoreDecimals)

        guard answers.count == questions.count, questions.allSatisfy({ answers[$0.id] != nil }) else {
            throw InvalidResponseDataError(message: "The evaluation must return exactly one answer for every question.")
        }

        for entry in questions {
            let id = entry.id
            let answer = answers[id]!
            guard answer.type == entry.question.type else {
                throw InvalidResponseDataError(message: "Question '\(id)' returned a \(answer.type.rawValue) answer.")
            }

            switch (entry.question, answer) {
            case (.choice(_, let options), .choice(let choice, let probabilities)):
                let names = options.map(\.name)
                guard names.contains(choice) else {
                    throw InvalidResponseDataError(message: "Question '\(id)' selected '\(choice)', which is not one of its options.")
                }
                guard let probabilities else { continue }
                try validateDistribution(probabilities, keys: Set(names), id: id, roundingError: probabilityError)
                let selected = probabilities[choice]!
                // Ties are allowed: an evenly split distribution has no single highest option.
                guard probabilities.values.allSatisfy({ $0 <= selected + tolerance }) else {
                    throw InvalidResponseDataError(message: "Question '\(id)' did not select a highest-probability option.")
                }

            case (.score(_, let levels), .score(let score, let probabilities)):
                let highest = Double(levels.count - 1)
                guard score.isFinite, score >= 0, score <= highest else {
                    throw InvalidResponseDataError(message: "Question '\(id)' scored \(score), outside its rubric of 0...\(levels.count - 1).")
                }
                guard let probabilities else { continue }
                try validateDistribution(probabilities, keys: Set(levels.indices), id: id, roundingError: probabilityError)

                let mean = probabilities.reduce(0) { $0 + Double($1.key) * $1.value }
                // Each rounded probability can shift the mean by its level index times its
                // rounding error, so the allowance grows with the rubric: Σ index · error.
                let meanRoundingError = probabilityError * Double(levels.count * (levels.count - 1)) / 2
                guard abs(mean - score) <= tolerance + meanRoundingError + scoreError else {
                    throw InvalidResponseDataError(
                        message: "Question '\(id)' scored \(score), but its distribution's weighted mean is \(mean)."
                    )
                }

            case (.boolean, .boolean(let probability)):
                guard probability.isFinite, (0...1).contains(probability) else {
                    throw InvalidResponseDataError(
                        message: "Question '\(id)' must return P(true) as a finite probability in 0...1, not \(probability)."
                    )
                }

            default:
                // Unreachable: the types were compared above.
                break
            }
        }
    }

    private static func validateDistribution<Key: Hashable>(
        _ probabilities: [Key: Double],
        keys: Set<Key>,
        id: String,
        roundingError: Double
    ) throws {
        guard Set(probabilities.keys) == keys,
              probabilities.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw InvalidResponseDataError(
                message: "Question '\(id)' must have a complete distribution of finite probabilities in 0...1."
            )
        }
        let sum = probabilities.values.reduce(0, +)
        guard abs(sum - 1) <= tolerance + Double(keys.count) * roundingError else {
            throw InvalidResponseDataError(message: "Question '\(id)' has probabilities summing to \(sum), not 1.")
        }
    }

    /// Half a unit in the last decimal place: the most a correctly rounded value can be off by.
    private static func roundingError(_ decimals: Int?) throws -> Double {
        guard let decimals else { return 0 }
        guard (0...15).contains(decimals) else {
            throw InvalidResponseDataError(message: "Rounding must declare between 0 and 15 decimal places, not \(decimals).")
        }
        return 0.5 * pow(10, -Double(decimals))
    }

    /// Whether a value can serve as state, instructions, or a description.
    ///
    /// A bare number, Boolean, or null says nothing on its own, so the top level must be text or a
    /// container.
    private static func isInput(_ value: JSONValue) -> Bool {
        switch value {
        case .string: return true
        case .array, .object: return isFinite(value)
        case .null, .bool, .int, .double: return false
        }
    }

    private static func isFinite(_ value: JSONValue) -> Bool {
        switch value {
        case .double(let number): return number.isFinite
        case .array(let elements): return elements.allSatisfy(isFinite)
        case .object(let members): return members.values.allSatisfy(isFinite)
        case .null, .bool, .int, .string: return true
        }
    }
}

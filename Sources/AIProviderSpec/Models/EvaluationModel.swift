import Foundation

/// The contract every evaluation provider implements.
///
/// An evaluation model answers a set of named questions about one shared state: which of several
/// options fits it, where it sits on an ordered rubric, or how likely a statement about it is to be
/// true. Some providers answer natively and return probability distributions — OpenRouter's
/// Decisions API, TypeSafe's Jev — while others adapt a language model's structured output.
///
/// Implementations handle exactly one request. Validating the questions beforehand and the answers
/// afterwards is the core's job, so a provider only translates to and from its wire format.
///
/// - Important: Experimental. This specification may change in a minor release.
public protocol EvaluationModelV1: Sendable {
    /// The provider's short name, such as `"openrouter"`.
    var provider: String { get }

    /// The provider's identifier for this model.
    var modelID: String { get }

    /// The question types this model can answer.
    ///
    /// The core checks every question against this before making a request, so an unsupported
    /// question fails fast rather than after a round trip.
    var supportedQuestionTypes: Set<EvaluationQuestionType> { get }

    /// Answers every question against the same state.
    ///
    /// - Returns: Exactly one answer per question, keyed by the question's identifier. There is no
    ///   partial success: a model that cannot answer every question throws.
    /// - Throws: ``APICallError`` for upstream failures, ``InvalidResponseDataError`` when the
    ///   provider's answers are malformed.
    func evaluate(_ options: EvaluationModelCallOptions) async throws -> EvaluationModelResponse
}

/// The current evaluation model specification.
///
/// - Important: Experimental. This specification may change in a minor release.
public typealias EvaluationModel = EvaluationModelV1

// MARK: - Questions

/// The kinds of judgment an evaluation model can make.
public enum EvaluationQuestionType: String, Sendable, Hashable, CaseIterable {
    /// Pick one option from a named set.
    case choice
    /// Place the state on an ordered rubric.
    case score
    /// Estimate the probability that a statement is true.
    case boolean
}

/// One option of a ``EvaluationQuestion/choice(instructions:options:)`` question.
public struct EvaluationOption: Sendable, Hashable {
    /// The option's name, returned as the answer when it is chosen.
    public var name: String

    /// What the option means. `nil` leaves the name to speak for itself.
    public var description: JSONValue?

    public init(_ name: String, description: JSONValue? = nil) {
        self.name = name
        self.description = description
    }
}

/// A judgment to make about the shared state.
///
/// Instructions and descriptions are JSON rather than strings so that structured rubrics — a list
/// of examples, an object of named criteria — can be passed through without flattening. A plain
/// string literal works wherever a ``JSONValue`` is expected.
///
/// ```swift
/// let department = EvaluationQuestion.choice(
///     "Which team should handle this?",
///     options: ["billing": "Payments and refunds", "support": "Other requests"]
/// )
/// ```
public enum EvaluationQuestion: Sendable, Hashable {
    /// Pick one option. Options are ordered; some providers present them in this order.
    case choice(instructions: JSONValue, options: [EvaluationOption])

    /// Place the state on a rubric of at least two ordered levels, indexed from zero.
    ///
    /// A `nil` level has no description of its own and is defined by its neighbours.
    case score(instructions: JSONValue, levels: [JSONValue?])

    /// Estimate P(true), optionally guided by what true and false mean here.
    case boolean(instructions: JSONValue, whenTrue: JSONValue? = nil, whenFalse: JSONValue? = nil)

    /// The kind of judgment this question asks for.
    public var type: EvaluationQuestionType {
        switch self {
        case .choice: return .choice
        case .score: return .score
        case .boolean: return .boolean
        }
    }

    /// The instructions shared by every question kind.
    public var instructions: JSONValue {
        switch self {
        case .choice(let instructions, _), .score(let instructions, _), .boolean(let instructions, _, _):
            return instructions
        }
    }

    /// A choice question whose options are written as a literal, in order.
    ///
    /// `KeyValuePairs` rather than a dictionary, because a dictionary literal loses the order the
    /// options were written in.
    public static func choice(
        _ instructions: JSONValue,
        options: KeyValuePairs<String, JSONValue?>
    ) -> EvaluationQuestion {
        .choice(instructions: instructions, options: options.map { EvaluationOption($0.key, description: $0.value) })
    }

    /// A score question whose levels are written in order, lowest first.
    public static func score(_ instructions: JSONValue, levels: [JSONValue?]) -> EvaluationQuestion {
        .score(instructions: instructions, levels: levels)
    }

    /// A boolean question.
    public static func boolean(
        _ instructions: JSONValue,
        whenTrue: JSONValue? = nil,
        whenFalse: JSONValue? = nil
    ) -> EvaluationQuestion {
        .boolean(instructions: instructions, whenTrue: whenTrue, whenFalse: whenFalse)
    }
}

/// The questions of one evaluation, keyed by identifier and kept in the order they were written.
///
/// Order is preserved because language-model adapters present questions in sequence, and a stable
/// order keeps their prompts reproducible.
///
/// ```swift
/// let questions: EvaluationQuestions = [
///     "department": .choice("Which team?", options: ["billing": nil, "support": nil]),
///     "refund": .boolean("Is the customer asking for money back?"),
/// ]
/// ```
public struct EvaluationQuestions: Sendable, Hashable, ExpressibleByDictionaryLiteral, RandomAccessCollection {
    /// One identified question.
    public struct Entry: Sendable, Hashable {
        public var id: String
        public var question: EvaluationQuestion

        public init(id: String, question: EvaluationQuestion) {
            self.id = id
            self.question = question
        }
    }

    /// The questions, in order.
    public private(set) var entries: [Entry]

    /// Creates a question set. A repeated identifier keeps its last question, in its first position.
    public init(_ entries: [Entry] = []) {
        self.entries = []
        for entry in entries { self[entry.id] = entry.question }
    }

    public init(dictionaryLiteral elements: (String, EvaluationQuestion)...) {
        self.init(elements.map { Entry(id: $0.0, question: $0.1) })
    }

    /// The identifiers, in order.
    public var ids: [String] { entries.map(\.id) }

    /// The question with an identifier. Assigning adds it at the end, or replaces it in place.
    public subscript(id: String) -> EvaluationQuestion? {
        get { entries.first { $0.id == id }?.question }
        set {
            let index = entries.firstIndex { $0.id == id }
            switch (index, newValue) {
            case (let index?, let question?): entries[index].question = question
            case (let index?, nil): entries.remove(at: index)
            case (nil, let question?): entries.append(Entry(id: id, question: question))
            case (nil, nil): break
            }
        }
    }

    public var startIndex: Int { entries.startIndex }
    public var endIndex: Int { entries.endIndex }
    public subscript(position: Int) -> Entry { entries[position] }
}

// MARK: - Answers

/// A model's answer to one question. Always the same kind as the question.
public enum EvaluationAnswer: Sendable, Hashable {
    /// The selected option's name.
    ///
    /// - Parameter probabilities: A complete distribution over every option, when the provider
    ///   supplies one. The selected option then has the highest probability.
    case choice(String, probabilities: [String: Double]? = nil)

    /// A fractional position on the rubric, in `0...(levels - 1)`.
    ///
    /// - Parameter probabilities: A complete distribution over level indices, when the provider
    ///   supplies one. The score is then its probability-weighted mean.
    case score(Double, probabilities: [Int: Double]? = nil)

    /// The model's estimate of P(true), in `0...1`.
    ///
    /// This is not confidence in either outcome: `0.02` is a confident *no*.
    case boolean(probability: Double)

    /// The kind of question this answers.
    public var type: EvaluationQuestionType {
        switch self {
        case .choice: return .choice
        case .score: return .score
        case .boolean: return .boolean
        }
    }

    /// The selected option, or `nil` for other answer kinds.
    public var choice: String? {
        guard case .choice(let choice, _) = self else { return nil }
        return choice
    }

    /// The score, or `nil` for other answer kinds.
    public var score: Double? {
        guard case .score(let score, _) = self else { return nil }
        return score
    }

    /// P(true), or `nil` for other answer kinds.
    public var probability: Double? {
        guard case .boolean(let probability) = self else { return nil }
        return probability
    }
}

/// The decimal places a provider rounds its output to.
///
/// Rounded distributions rarely sum to exactly one, so validation allows half a unit in the last
/// place per rounded value. Omit a field when that value is reported at full precision.
public struct EvaluationRounding: Sendable, Hashable {
    /// Decimal places of reported probabilities, from 0 to 15.
    public var probabilityDecimals: Int?

    /// Decimal places of reported scores, from 0 to 15.
    public var scoreDecimals: Int?

    public init(probabilityDecimals: Int? = nil, scoreDecimals: Int? = nil) {
        self.probabilityDecimals = probabilityDecimals
        self.scoreDecimals = scoreDecimals
    }
}

// MARK: - Call options and response

/// A request to evaluate questions against one state.
public struct EvaluationModelCallOptions: Sendable {
    /// The shared state: a string, an object, or an array. An array is one state, not a batch.
    public var state: JSONValue

    /// The questions to answer.
    public var questions: EvaluationQuestions

    /// Extra HTTP headers to merge into the request.
    public var headers: [String: String]

    /// Provider-specific settings, namespaced by provider name.
    public var providerOptions: ProviderOptions?

    public init(
        state: JSONValue,
        questions: EvaluationQuestions,
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil
    ) {
        self.state = state
        self.questions = questions
        self.headers = headers
        self.providerOptions = providerOptions
    }
}

/// The response from an evaluation request.
public struct EvaluationModelResponse: Sendable {
    /// One answer per question, keyed by the question's identifier.
    public var answers: [String: EvaluationAnswer]

    /// How the provider rounded its output, or `nil` for full precision.
    public var rounding: EvaluationRounding?

    /// Token usage.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values, such as a native confidence statistic.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about the response.
    public var response: ResponseInfo?

    public init(
        answers: [String: EvaluationAnswer],
        rounding: EvaluationRounding? = nil,
        usage: Usage = .none,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        response: ResponseInfo? = nil
    ) {
        self.answers = answers
        self.rounding = rounding
        self.usage = usage
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.response = response
    }
}

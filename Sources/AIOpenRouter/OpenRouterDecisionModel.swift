import AIProviderSpec
import AIProviderUtils
import Foundation

/// An evaluation model served by OpenRouter's Decisions API.
///
/// Decisions are OpenRouter's name for evaluation: questions about one shared state, answered
/// natively by models such as TypeSafe's Jev. Choice and score answers come with probability
/// distributions; each also carries a confidence statistic, reported in
/// `providerMetadata["openrouter"]["confidence"]` keyed by question. That statistic is
/// OpenRouter's own and is not the selected option's probability.
///
/// Provider options under `"openrouter"` are merged into the request body — `provider` routing
/// preferences, `user`, `session_id`, `trace` — but never replace the model, state, or questions.
///
/// - Important: Experimental. The Decisions API is in alpha.
public struct OpenRouterDecisionModel: EvaluationModel {
    public let provider: String
    public let modelID: String
    public let supportedQuestionTypes = Set(EvaluationQuestionType.allCases)

    let client: ProviderHTTPClient?
    let extraBody: [String: JSONValue]

    /// The API rounds probabilities and scores to two decimal places.
    static let rounding = EvaluationRounding(probabilityDecimals: 2, scoreDecimals: 2)

    public func evaluate(_ options: EvaluationModelCallOptions) async throws -> EvaluationModelResponse {
        guard let client else {
            throw InvalidArgumentError(
                argument: "baseURL",
                message: """
                    The Decisions API URL cannot be derived from a base URL that does not end in \
                    '/v1'. Pass `decisionsBaseURL` when creating the provider.
                    """
            )
        }

        var fields = extraBody
        for (key, value) in options.providerOptions?[provider] ?? [:] { fields[key] = value }
        fields["model"] = .string(modelID)
        fields["state"] = options.state
        fields["questions"] = .object(Dictionary(uniqueKeysWithValues: try options.questions.map { entry in
            (entry.id, try Self.wireQuestion(entry.question, id: entry.id))
        }))

        let (body, head, _) = try await client.postJSON(
            path: "decisions",
            body: JSONValue.object(fields),
            additionalHeaders: options.headers,
            as: JSONValue.self
        )
        try OpenRouterResponse.throwIfError(body, url: try client.url(path: "decisions"), headers: head.headers)

        var answers: [String: EvaluationAnswer] = [:]
        var confidence: [String: JSONValue] = [:]
        for (id, answer) in body["answers"]?.objectValue ?? [:] {
            answers[id] = try Self.answer(answer, id: id)
            if let value = answer["confidence"], !value.isNull { confidence[id] = value }
        }

        var metadata: [String: JSONValue] = [:]
        if !confidence.isEmpty { metadata["confidence"] = .object(confidence) }
        if let upstream = body["provider"]?.stringValue { metadata["provider"] = .string(upstream) }
        if let cost = body["usage"]?["cost"], !cost.isNull { metadata["cost"] = cost }

        return EvaluationModelResponse(
            answers: answers,
            rounding: Self.rounding,
            usage: Usage(
                inputTokens: body["usage"]?["input_tokens"]?.intValue,
                outputTokens: body["usage"]?["output_tokens"]?.intValue
            ),
            providerMetadata: metadata.isEmpty ? nil : ProviderMetadata([provider: metadata]),
            response: ResponseInfo(
                id: body["id"]?.stringValue,
                modelID: body["model"]?.stringValue ?? modelID,
                headers: head.headers
            )
        )
    }

    /// Encodes a question in the Decisions API's form.
    ///
    /// The API calls a boolean question `noul`, and rejects two shapes the evaluation spec allows:
    /// a score level without a description, and boolean criteria describing only one side. Those
    /// fail here, before the request, with a message that says what to change.
    static func wireQuestion(_ question: EvaluationQuestion, id: String) throws -> JSONValue {
        switch question {
        case .choice(let instructions, let options):
            return .object([
                "type": "choice",
                "instructions": instructions,
                "criteria": .object(Dictionary(uniqueKeysWithValues: options.map { ($0.name, $0.description ?? .null) })),
            ])

        case .score(let instructions, let levels):
            let described = levels.compactMap { $0 }
            guard described.count == levels.count else {
                throw InvalidArgumentError(
                    argument: "questions.\(id)",
                    message: "The OpenRouter Decisions API requires a description for every score level."
                )
            }
            return .object(["type": "score", "instructions": instructions, "criteria": .array(described)])

        case .boolean(let instructions, let whenTrue, let whenFalse):
            var fields: [String: JSONValue] = ["type": "noul", "instructions": instructions]
            switch (whenTrue, whenFalse) {
            case (nil, nil):
                break
            case (let whenTrue?, let whenFalse?):
                fields["criteria"] = .object(["true": whenTrue, "false": whenFalse])
            default:
                throw InvalidArgumentError(
                    argument: "questions.\(id)",
                    message: "The OpenRouter Decisions API requires both a true and a false description, or neither."
                )
            }
            return .object(fields)
        }
    }

    /// Decodes an answer, translating `noul` back to boolean.
    ///
    /// Values are passed through unchanged; the core validates them against the questions.
    static func answer(_ answer: JSONValue, id: String) throws -> EvaluationAnswer {
        switch answer["type"]?.stringValue {
        case "choice":
            guard let choice = answer["choice"]?.stringValue else { break }
            return .choice(choice, probabilities: answer["probabilities"]?.objectValue?.compactMapValues(\.numberValue))

        case "score":
            guard let score = answer["score"]?.numberValue else { break }
            let probabilities = answer["probabilities"]?.objectValue.map { values in
                Dictionary(uniqueKeysWithValues: values.compactMap { key, value in
                    Int(key).flatMap { index in value.numberValue.map { (index, $0) } }
                })
            }
            return .score(score, probabilities: probabilities)

        case "noul", "boolean":
            guard let probability = (answer["noul"] ?? answer["probability"])?.numberValue else { break }
            return .boolean(probability: probability)

        default:
            break
        }
        throw InvalidResponseDataError(message: "Question '\(id)' returned an answer OpenRouter's schema does not describe.", data: answer)
    }
}

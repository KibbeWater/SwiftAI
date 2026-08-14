import AIProviderSpec
import AIProviderUtils
import Foundation

/// Turns Gemini's streamed chunks into spec stream parts.
///
/// Gemini streams complete parts rather than a block-structured event sequence: each chunk carries
/// whatever is new, and function calls arrive whole rather than as argument fragments. The decoder
/// therefore has to synthesize the block structure the spec guarantees — opening a text block on
/// the first text part, closing it when reasoning or a tool call interrupts, and emitting a
/// complete start/delta/end triple around each function call so consumers see the same shape they
/// would from a provider that genuinely streams arguments.
struct GoogleStreamDecoder {
    private var textBlockID: String?
    private var reasoningBlockID: String?
    private var finishReason: FinishReason = .unknown
    private var usage: Usage = .none
    private var hasReportedMetadata = false
    private var sawFunctionCall = false

    mutating func consume(_ chunk: GoogleGenerateContentResponse) -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []

        if !hasReportedMetadata, chunk.responseId != nil || chunk.modelVersion != nil {
            hasReportedMetadata = true
            parts.append(.responseMetadata(id: chunk.responseId, modelID: chunk.modelVersion))
        }
        if let usage = chunk.usageMetadata {
            self.usage = usage.normalized
        }

        guard let candidate = chunk.candidates?.first else { return parts }
        if let reason = candidate.finishReason {
            finishReason = FinishReason.fromGoogle(reason, hasFunctionCall: sawFunctionCall)
        }

        for part in candidate.content?.parts ?? [] {
            if let call = part.functionCall, let name = call.name {
                sawFunctionCall = true
                parts.append(contentsOf: closeOpenBlocks())

                // Gemini delivers arguments whole. Emitting the start/delta/end triple anyway
                // means a consumer never has to special-case this provider.
                let id = IdentifierGenerator.generate(prefix: "call")
                let input = call.args ?? .object([:])
                parts.append(.toolInputStart(id: id, toolName: name))
                parts.append(.toolInputDelta(id: id, delta: input.serialized()))
                parts.append(.toolInputEnd(id: id))
                parts.append(.toolCall(ToolCallPart(toolCallID: id, toolName: name, input: input)))
                continue
            }

            guard let text = part.text, !text.isEmpty else { continue }

            if part.thought == true {
                if let id = textBlockID {
                    parts.append(.textEnd(id: id))
                    textBlockID = nil
                }
                if reasoningBlockID == nil {
                    let id = IdentifierGenerator.generate(prefix: "reasoning")
                    reasoningBlockID = id
                    parts.append(.reasoningStart(id: id))
                }
                parts.append(.reasoningDelta(id: reasoningBlockID!, delta: text))
            } else {
                if let id = reasoningBlockID {
                    parts.append(.reasoningEnd(id: id))
                    reasoningBlockID = nil
                }
                if textBlockID == nil {
                    let id = IdentifierGenerator.generate(prefix: "text")
                    textBlockID = id
                    parts.append(.textStart(id: id))
                }
                parts.append(.textDelta(id: textBlockID!, delta: text))
            }
        }

        for source in candidate.citationMetadata?.citationSources ?? [] {
            guard let uri = source.uri, let url = URL(string: uri) else { continue }
            parts.append(
                .source(
                    SourcePart(
                        id: IdentifierGenerator.generate(prefix: "source"),
                        kind: .url(url),
                        title: source.title
                    )
                )
            )
        }
        return parts
    }

    /// Closes any open text or reasoning block and emits the finish part.
    mutating func finish() -> [LanguageModelStreamPart] {
        var parts = closeOpenBlocks()
        parts.append(
            .finish(
                finishReason: sawFunctionCall ? .toolCalls : finishReason,
                usage: usage
            )
        )
        return parts
    }

    private mutating func closeOpenBlocks() -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []
        if let id = reasoningBlockID {
            parts.append(.reasoningEnd(id: id))
            reasoningBlockID = nil
        }
        if let id = textBlockID {
            parts.append(.textEnd(id: id))
            textBlockID = nil
        }
        return parts
    }
}

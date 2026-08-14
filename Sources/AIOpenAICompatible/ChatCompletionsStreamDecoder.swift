import AIProviderSpec
import AIProviderUtils
import Foundation

/// Turns a sequence of `chat.completion.chunk` events into spec stream parts.
///
/// Chat Completions streams a flat sequence of deltas with no block structure: text just arrives,
/// and tool arguments arrive as fragments correlated by an integer index. The spec's stream is
/// block-structured, so this decoder synthesizes the missing boundaries — opening a text block on
/// the first content delta, opening a tool-input block on the first fragment of each call, and
/// closing everything when the stream ends.
///
/// Getting this right matters for consumers: it means a UI can rely on the same start/delta/end
/// shape regardless of whether the provider underneath has one.
struct ChatCompletionsStreamDecoder {
    /// State for one in-flight tool call, keyed by the wire's index.
    private struct PendingToolCall {
        var id: String
        var name: String
        var arguments: String = ""
        var hasStarted = false
    }

    private var textBlockID: String?
    private var reasoningBlockID: String?
    private var toolCalls: [Int: PendingToolCall] = [:]
    private var finishReason: FinishReason = .unknown
    private var usage: Usage = .none
    private var hasReportedMetadata = false

    /// Consumes one chunk and returns the parts it produced.
    mutating func consume(_ chunk: ChatCompletionChunk) throws -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []

        if !hasReportedMetadata, chunk.id != nil || chunk.model != nil {
            hasReportedMetadata = true
            parts.append(
                .responseMetadata(
                    id: chunk.id,
                    modelID: chunk.model,
                    timestamp: chunk.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }
                )
            )
        }

        // The final chunk of an OpenAI stream carries usage and no choices.
        if let usage = chunk.usage {
            self.usage = usage.normalized
        }

        for choice in chunk.choices ?? [] {
            if let reason = choice.finishReason {
                finishReason = FinishReason.fromOpenAI(reason)
            }
            guard let delta = choice.delta else { continue }

            if let reasoning = delta.reasoningContent ?? delta.reasoning, !reasoning.isEmpty {
                if reasoningBlockID == nil {
                    let id = IdentifierGenerator.generate(prefix: "reasoning")
                    reasoningBlockID = id
                    parts.append(.reasoningStart(id: id))
                }
                parts.append(.reasoningDelta(id: reasoningBlockID!, delta: reasoning))
            }

            if let text = delta.content, !text.isEmpty {
                // Reasoning always precedes the answer, so a first content delta closes it.
                if let reasoningID = reasoningBlockID {
                    parts.append(.reasoningEnd(id: reasoningID))
                    reasoningBlockID = nil
                }
                if textBlockID == nil {
                    let id = IdentifierGenerator.generate(prefix: "text")
                    textBlockID = id
                    parts.append(.textStart(id: id))
                }
                parts.append(.textDelta(id: textBlockID!, delta: text))
            }

            if let refusal = delta.refusal, !refusal.isEmpty {
                if textBlockID == nil {
                    let id = IdentifierGenerator.generate(prefix: "text")
                    textBlockID = id
                    parts.append(.textStart(id: id))
                }
                parts.append(.textDelta(id: textBlockID!, delta: refusal))
            }

            for fragment in delta.toolCalls ?? [] {
                parts.append(contentsOf: consume(fragment))
            }
        }
        return parts
    }

    private mutating func consume(_ fragment: ChatCompletionChunk.ToolCallDelta) -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []
        let index = fragment.index ?? 0

        var pending = toolCalls[index] ?? PendingToolCall(
            id: fragment.id ?? IdentifierGenerator.generate(prefix: "call"),
            name: fragment.function?.name ?? ""
        )
        // Identifier and name arrive on the first fragment, but not always the very first one.
        if let id = fragment.id { pending.id = id }
        if let name = fragment.function?.name, !name.isEmpty { pending.name = name }

        if !pending.hasStarted, !pending.name.isEmpty {
            pending.hasStarted = true
            parts.append(.toolInputStart(id: pending.id, toolName: pending.name))
        }

        if let arguments = fragment.function?.arguments, !arguments.isEmpty {
            pending.arguments += arguments
            if pending.hasStarted {
                parts.append(.toolInputDelta(id: pending.id, delta: arguments))
            }
        }

        toolCalls[index] = pending
        return parts
    }

    /// Closes any open blocks and emits the completed tool calls and the finish part.
    ///
    /// Tool calls are emitted here rather than as they stream because their arguments are only
    /// parseable once complete.
    mutating func finish() -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []

        if let id = reasoningBlockID {
            parts.append(.reasoningEnd(id: id))
            reasoningBlockID = nil
        }
        if let id = textBlockID {
            parts.append(.textEnd(id: id))
            textBlockID = nil
        }

        for index in toolCalls.keys.sorted() {
            guard let pending = toolCalls[index], !pending.name.isEmpty else { continue }

            if !pending.hasStarted {
                // A server that sent no name until the end still gets a well-formed block.
                parts.append(.toolInputStart(id: pending.id, toolName: pending.name))
                parts.append(.toolInputDelta(id: pending.id, delta: pending.arguments))
            }
            parts.append(.toolInputEnd(id: pending.id))

            do {
                let input = try ProviderJSON.parseEmbeddedJSON(
                    pending.arguments,
                    context: "the arguments for '\(pending.name)'"
                )
                parts.append(
                    .toolCall(
                        ToolCallPart(toolCallID: pending.id, toolName: pending.name, input: input)
                    )
                )
            } catch {
                // Unparseable arguments are reported as a tool error the model can react to,
                // rather than failing the whole stream.
                parts.append(
                    .toolResult(
                        ToolResultPart(
                            toolCallID: pending.id,
                            toolName: pending.name,
                            output: .errorText(
                                "The model produced arguments that are not valid JSON: \(pending.arguments)"
                            )
                        )
                    )
                )
            }
        }
        toolCalls.removeAll()

        // A stream that produced tool calls but reported "stop" would stall the loop.
        let reason = finishReason
        parts.append(.finish(finishReason: reason, usage: usage))
        return parts
    }
}

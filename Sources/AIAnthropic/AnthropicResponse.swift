import AIProviderSpec
import AIProviderUtils
import Foundation

/// A Messages API response.
struct AnthropicMessage: Decodable {
    var id: String?
    var model: String?
    var role: String?
    var content: [ContentBlock]
    var stopReason: String?
    var stopSequence: String?
    var usage: TokenUsage?

    struct ContentBlock: Decodable {
        var type: String
        var text: String?
        var thinking: String?
        var signature: String?
        var data: String?

        // Present on `tool_use` blocks.
        var id: String?
        var name: String?
        var input: JSONValue?
    }

    struct TokenUsage: Decodable {
        var inputTokens: Int?
        var outputTokens: Int?
        var cacheCreationInputTokens: Int?
        var cacheReadInputTokens: Int?

        /// Normalizes to the shared usage type.
        ///
        /// Anthropic reports cache writes and cache reads separately, and counts neither in
        /// `input_tokens`. The total therefore has to be assembled here rather than taken from the
        /// response, or a cached conversation would appear to cost almost nothing.
        var normalized: Usage {
            let input = (inputTokens ?? 0) + (cacheCreationInputTokens ?? 0) + (cacheReadInputTokens ?? 0)
            return Usage(
                inputTokens: inputTokens == nil ? nil : input,
                outputTokens: outputTokens,
                totalTokens: outputTokens.map { input + $0 },
                cachedInputTokens: cacheReadInputTokens
            )
        }

        /// The provider-specific counts, which have no home on ``Usage``.
        var metadata: [String: JSONValue] {
            var values: [String: JSONValue] = [:]
            if let cacheCreationInputTokens {
                values["cacheCreationInputTokens"] = .int(cacheCreationInputTokens)
            }
            if let cacheReadInputTokens {
                values["cacheReadInputTokens"] = .int(cacheReadInputTokens)
            }
            return values
        }
    }
}

extension AnthropicMessage {
    /// The content blocks, converted to the shared representation.
    ///
    /// A forced structured-output tool call is republished as text, so that a caller who asked for
    /// JSON receives JSON rather than a tool call they never registered.
    func modelContent() -> [ModelContent] {
        content.compactMap { block in
            switch block.type {
            case "text":
                return block.text.map { .text(TextPart($0)) }

            case "thinking":
                return block.thinking.map { .reasoning(ReasoningPart($0, signature: block.signature)) }

            case "redacted_thinking":
                // The content is encrypted, but the block still has to be replayed to continue.
                return .reasoning(
                    ReasoningPart(
                        "",
                        signature: block.data,
                        providerOptions: ["anthropic": ["redacted": .bool(true)]]
                    )
                )

            case "tool_use":
                guard let id = block.id, let name = block.name else { return nil }
                if name == AnthropicStructuredOutput.toolName {
                    return .text(TextPart((block.input ?? .object([:])).serialized()))
                }
                return .toolCall(
                    ToolCallPart(toolCallID: id, toolName: name, input: block.input ?? .object([:]))
                )

            default:
                return nil
            }
        }
    }

    var normalizedFinishReason: FinishReason {
        // A forced structured-output call reports `tool_use`, but from the caller's point of view
        // the model answered and stopped.
        let isStructuredOutput = content.contains {
            $0.type == "tool_use" && $0.name == AnthropicStructuredOutput.toolName
        }
        if isStructuredOutput { return .stop }
        return FinishReason.fromAnthropic(stopReason)
    }

    var providerMetadata: ProviderMetadata? {
        let values = usage?.metadata ?? [:]
        return values.isEmpty ? nil : ProviderMetadata(["anthropic": values])
    }
}

extension FinishReason {
    /// Maps an Anthropic stop reason.
    static func fromAnthropic(_ raw: String?) -> FinishReason {
        switch raw {
        case "end_turn", "stop_sequence": return .stop
        case "max_tokens": return .length
        case "tool_use": return .toolCalls
        case "refusal": return .contentFilter
        case "pause_turn": return .other
        case nil: return .unknown
        default: return .other
        }
    }
}

// MARK: - Streaming events

/// One event from a streamed Messages response.
///
/// The stream is a sequence of typed events rather than repeated snapshots, which maps almost
/// directly onto this SDK's block-structured stream — `content_block_start` and
/// `content_block_stop` are the boundaries other providers have to have synthesized for them.
struct AnthropicStreamEvent: Decodable {
    var type: String
    var index: Int?
    var message: AnthropicMessage?
    var contentBlock: AnthropicMessage.ContentBlock?
    var delta: Delta?
    var usage: AnthropicMessage.TokenUsage?
    var error: ErrorPayload?

    struct Delta: Decodable {
        var type: String?
        var text: String?
        var thinking: String?
        var signature: String?
        var partialJson: String?
        var stopReason: String?
        var stopSequence: String?
    }

    struct ErrorPayload: Decodable {
        var type: String?
        var message: String?
    }
}

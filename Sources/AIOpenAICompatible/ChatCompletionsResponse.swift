import AIProviderSpec
import AIProviderUtils
import Foundation

/// The Chat Completions response payload.
///
/// Decoded with ``ProviderJSON/snakeCaseDecoder``, so the wire's `finish_reason` arrives as
/// `finishReason` without a hand-written `CodingKeys` for every field. Every property is optional
/// because compatible servers vary in what they populate, and a missing field should degrade
/// gracefully rather than fail the whole response.
struct ChatCompletionResponse: Decodable {
    var id: String?
    var model: String?
    var created: Int?
    var choices: [Choice]
    var usage: TokenUsage?

    struct Choice: Decodable {
        var index: Int?
        var message: Message
        var finishReason: String?
    }

    struct Message: Decodable {
        var content: String?
        var toolCalls: [ToolCall]?

        /// The reasoning channel, spelled differently by different servers.
        ///
        /// DeepSeek and vLLM use `reasoning_content`; some gateways use `reasoning`.
        var reasoningContent: String?
        var reasoning: String?

        /// A refusal, which OpenAI returns instead of content when it declines.
        var refusal: String?
    }

    struct ToolCall: Decodable {
        var id: String?
        var type: String?
        var function: Function

        struct Function: Decodable {
            var name: String?
            var arguments: String?
        }
    }

    struct TokenUsage: Decodable {
        var promptTokens: Int?
        var completionTokens: Int?
        var totalTokens: Int?
        var promptTokensDetails: PromptDetails?
        var completionTokensDetails: CompletionDetails?

        struct PromptDetails: Decodable {
            var cachedTokens: Int?
        }

        struct CompletionDetails: Decodable {
            var reasoningTokens: Int?
        }

        var normalized: Usage {
            Usage(
                inputTokens: promptTokens,
                outputTokens: completionTokens,
                totalTokens: totalTokens,
                reasoningTokens: completionTokensDetails?.reasoningTokens,
                cachedInputTokens: promptTokensDetails?.cachedTokens
            )
        }
    }
}

extension ChatCompletionResponse {
    /// The content parts, in the order a reader would expect them.
    ///
    /// Reasoning comes first because it precedes the answer chronologically, even though the wire
    /// format carries both as sibling fields with no ordering of their own.
    func modelContent() throws -> [ModelContent] {
        guard let choice = choices.first else { return [] }
        var content: [ModelContent] = []

        if let reasoning = choice.message.reasoningContent ?? choice.message.reasoning,
           !reasoning.isEmpty {
            content.append(.reasoning(ReasoningPart(reasoning)))
        }
        if let text = choice.message.content, !text.isEmpty {
            content.append(.text(TextPart(text)))
        }
        // A refusal is the model's answer, so it belongs in the text rather than being dropped.
        if let refusal = choice.message.refusal, !refusal.isEmpty {
            content.append(.text(TextPart(refusal)))
        }

        for call in choice.message.toolCalls ?? [] {
            guard let name = call.function.name else { continue }
            content.append(
                .toolCall(
                    ToolCallPart(
                        toolCallID: call.id ?? IdentifierGenerator.generate(prefix: "call"),
                        toolName: name,
                        input: try ProviderJSON.parseEmbeddedJSON(
                            call.function.arguments ?? "",
                            context: "the arguments for '\(name)'"
                        )
                    )
                )
            )
        }
        return content
    }

    var normalizedFinishReason: FinishReason {
        FinishReason.fromOpenAI(choices.first?.finishReason)
    }

    var responseInfo: ResponseInfo {
        ResponseInfo(
            id: id,
            modelID: model,
            timestamp: created.map { Date(timeIntervalSince1970: TimeInterval($0)) }
        )
    }
}

extension FinishReason {
    /// Maps an OpenAI-style finish reason.
    static func fromOpenAI(_ raw: String?) -> FinishReason {
        switch raw {
        case "stop": return .stop
        case "length", "max_tokens": return .length
        case "tool_calls", "function_call": return .toolCalls
        case "content_filter": return .contentFilter
        case nil: return .unknown
        default: return .other
        }
    }
}

// MARK: - Streaming chunks

/// One `chat.completion.chunk` event.
struct ChatCompletionChunk: Decodable {
    var id: String?
    var model: String?
    var created: Int?
    var choices: [Choice]?
    var usage: ChatCompletionResponse.TokenUsage?

    struct Choice: Decodable {
        var index: Int?
        var delta: Delta?
        var finishReason: String?
    }

    struct Delta: Decodable {
        var content: String?
        var reasoningContent: String?
        var reasoning: String?
        var refusal: String?
        var toolCalls: [ToolCallDelta]?
    }

    /// A fragment of a tool call.
    ///
    /// Only the first fragment carries `id` and `function.name`; the rest carry successive slices
    /// of the argument JSON. Fragments are correlated by ``index``, not by identifier, which is why
    /// the decoder keys its state on the index.
    struct ToolCallDelta: Decodable {
        var index: Int?
        var id: String?
        var type: String?
        var function: FunctionDelta?

        struct FunctionDelta: Decodable {
            var name: String?
            var arguments: String?
        }
    }
}

// MARK: - Embeddings

struct EmbeddingsResponse: Decodable {
    var model: String?
    var data: [Item]
    var usage: TokenUsage?

    struct Item: Decodable {
        var index: Int?
        var embedding: [Double]
    }

    struct TokenUsage: Decodable {
        var promptTokens: Int?
        var totalTokens: Int?
    }
}

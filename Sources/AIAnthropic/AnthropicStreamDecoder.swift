import AIProviderSpec
import AIProviderUtils
import Foundation

/// Turns Anthropic's stream events into spec stream parts.
///
/// The mapping is close to one-to-one, because Anthropic's stream is already block-structured:
/// `content_block_start` and `content_block_stop` bracket each block, and deltas name their type.
/// The work here is correlating blocks by their integer index — the wire's only identifier for a
/// text or thinking block — and holding tool arguments until they are complete enough to parse.
struct AnthropicStreamDecoder {
    /// What an open content block is, and what has accumulated in it.
    private struct OpenBlock {
        enum Kind {
            case text
            case thinking
            case toolUse(id: String, name: String)
        }

        var kind: Kind
        var blockID: String
        var accumulated = ""
        var signature: String?
    }

    private var blocks: [Int: OpenBlock] = [:]
    private var finishReason: FinishReason = .unknown
    private var inputUsage: AnthropicMessage.TokenUsage?
    private var outputUsage: AnthropicMessage.TokenUsage?
    private var isStructuredOutput = false

    /// Consumes one event and returns the parts it produced.
    mutating func consume(_ event: AnthropicStreamEvent) throws -> [LanguageModelStreamPart] {
        switch event.type {
        case "message_start":
            inputUsage = event.message?.usage
            return [
                .responseMetadata(id: event.message?.id, modelID: event.message?.model)
            ]

        case "content_block_start":
            return startBlock(index: event.index ?? 0, block: event.contentBlock)

        case "content_block_delta":
            return deltaBlock(index: event.index ?? 0, delta: event.delta)

        case "content_block_stop":
            return try stopBlock(index: event.index ?? 0)

        case "message_delta":
            if let reason = event.delta?.stopReason {
                finishReason = FinishReason.fromAnthropic(reason)
            }
            outputUsage = event.usage
            return []

        case "message_stop":
            return []

        case "error":
            throw APICallError(
                message: event.error?.message ?? "The Anthropic stream reported an error.",
                url: URL(string: "https://api.anthropic.com/v1/messages")!,
                // Overloaded is the one error that routinely appears mid-stream and is worth
                // another attempt.
                isRetryable: event.error?.type == "overloaded_error"
            )

        default:
            // `ping` and any event type added after this was written.
            return []
        }
    }

    private mutating func startBlock(
        index: Int,
        block: AnthropicMessage.ContentBlock?
    ) -> [LanguageModelStreamPart] {
        guard let block else { return [] }

        switch block.type {
        case "text":
            let id = "text_\(index)"
            blocks[index] = OpenBlock(kind: .text, blockID: id)
            var parts: [LanguageModelStreamPart] = [.textStart(id: id)]
            if let text = block.text, !text.isEmpty {
                blocks[index]?.accumulated = text
                parts.append(.textDelta(id: id, delta: text))
            }
            return parts

        case "thinking", "redacted_thinking":
            let id = "thinking_\(index)"
            blocks[index] = OpenBlock(kind: .thinking, blockID: id)
            return [.reasoningStart(id: id)]

        case "tool_use":
            guard let id = block.id, let name = block.name else { return [] }
            if name == AnthropicStructuredOutput.toolName {
                // The forced structured-output call is republished as text, so its arguments
                // stream as the JSON the caller asked for.
                isStructuredOutput = true
                blocks[index] = OpenBlock(kind: .text, blockID: id)
                return [.textStart(id: id)]
            }
            blocks[index] = OpenBlock(kind: .toolUse(id: id, name: name), blockID: id)
            return [.toolInputStart(id: id, toolName: name)]

        default:
            return []
        }
    }

    private mutating func deltaBlock(
        index: Int,
        delta: AnthropicStreamEvent.Delta?
    ) -> [LanguageModelStreamPart] {
        guard let delta, var block = blocks[index] else { return [] }
        defer { blocks[index] = block }

        switch delta.type {
        case "text_delta":
            guard let text = delta.text else { return [] }
            block.accumulated += text
            return [.textDelta(id: block.blockID, delta: text)]

        case "thinking_delta":
            guard let thinking = delta.thinking else { return [] }
            block.accumulated += thinking
            return [.reasoningDelta(id: block.blockID, delta: thinking)]

        case "signature_delta":
            // The signature is metadata, not content; it is attached when the block closes.
            block.signature = (block.signature ?? "") + (delta.signature ?? "")
            return []

        case "input_json_delta":
            guard let fragment = delta.partialJson else { return [] }
            block.accumulated += fragment
            // A structured-output block is text from the caller's point of view.
            if case .text = block.kind {
                return [.textDelta(id: block.blockID, delta: fragment)]
            }
            return [.toolInputDelta(id: block.blockID, delta: fragment)]

        default:
            return []
        }
    }

    private mutating func stopBlock(index: Int) throws -> [LanguageModelStreamPart] {
        guard let block = blocks.removeValue(forKey: index) else { return [] }

        switch block.kind {
        case .text:
            return [.textEnd(id: block.blockID)]

        case .thinking:
            var metadata: ProviderMetadata?
            if let signature = block.signature {
                metadata = ProviderMetadata(["anthropic": ["signature": .string(signature)]])
            }
            return [.reasoningEnd(id: block.blockID, providerMetadata: metadata)]

        case .toolUse(let id, let name):
            var parts: [LanguageModelStreamPart] = [.toolInputEnd(id: id)]
            do {
                let input = try ProviderJSON.parseEmbeddedJSON(
                    block.accumulated,
                    context: "the arguments for '\(name)'"
                )
                parts.append(.toolCall(ToolCallPart(toolCallID: id, toolName: name, input: input)))
            } catch {
                parts.append(
                    .toolResult(
                        ToolResultPart(
                            toolCallID: id,
                            toolName: name,
                            output: .errorText(
                                "The model produced arguments that are not valid JSON: \(block.accumulated)"
                            )
                        )
                    )
                )
            }
            return parts
        }
    }

    /// Closes anything still open and emits the finish part.
    mutating func finish() -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []

        // A stream cut short leaves blocks open; closing them keeps the stream well formed.
        for index in blocks.keys.sorted() {
            parts.append(contentsOf: (try? stopBlock(index: index)) ?? [])
        }

        var usage = Usage(
            inputTokens: inputUsage?.inputTokens,
            outputTokens: outputUsage?.outputTokens ?? inputUsage?.outputTokens,
            cachedInputTokens: inputUsage?.cacheReadInputTokens
        )
        let cacheWrites = inputUsage?.cacheCreationInputTokens ?? 0
        let cacheReads = inputUsage?.cacheReadInputTokens ?? 0
        if let input = usage.inputTokens {
            // Cached tokens are billed but excluded from `input_tokens`.
            usage.inputTokens = input + cacheWrites + cacheReads
        }
        usage.totalTokens = usage.resolvedTotalTokens

        var metadata: ProviderMetadata?
        let values = inputUsage?.metadata ?? [:]
        if !values.isEmpty { metadata = ProviderMetadata(["anthropic": values]) }

        parts.append(
            .finish(
                finishReason: isStructuredOutput && finishReason == .toolCalls ? .stop : finishReason,
                usage: usage,
                providerMetadata: metadata
            )
        )
        return parts
    }
}

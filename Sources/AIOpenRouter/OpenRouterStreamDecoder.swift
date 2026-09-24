import AIProviderSpec
import AIProviderUtils
import Foundation

/// Turns OpenRouter's streamed chunks into spec stream parts.
///
/// On top of Chat Completions' usual block synthesis, this assembles `reasoning_details` from
/// fragments. The assembled details close the reasoning block, so the step keeps them for the next
/// turn, and ride on the first tool call, for the reasons given at
/// ``OpenRouterResponse/toolCallOptions(isFirst:details:providerName:)``.
struct OpenRouterStreamDecoder {
    let providerName: String

    private struct PendingToolCall {
        var id: String
        var name: String
        var arguments = ""
        var hasStarted = false
    }

    private var reasoningBlockID: String?
    private var textBlockID: String?
    private var hasStartedText = false
    private var reasoningDetails: [JSONValue] = []
    private var toolCalls: [Int: PendingToolCall] = [:]
    private var toolCallIDs: Set<String> = []
    private var finishReason: String?
    private var usage: JSONValue?
    private var upstreamProvider: String?
    private var fileAnnotations: [JSONValue] = []
    private var hasReportedMetadata = false

    init(providerName: String) {
        self.providerName = providerName
    }

    mutating func consume(_ chunk: JSONValue) throws -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []

        if !hasReportedMetadata, chunk["id"] != nil || chunk["model"] != nil {
            hasReportedMetadata = true
            parts.append(.responseMetadata(
                id: chunk["id"]?.stringValue,
                modelID: chunk["model"]?.stringValue,
                timestamp: chunk["created"]?.numberValue.map { Date(timeIntervalSince1970: $0) }
            ))
        }
        if let provider = chunk["provider"]?.stringValue { upstreamProvider = provider }
        // Usage can arrive on any chunk; the last report is the complete one.
        if let usage = chunk["usage"], !usage.isNull { self.usage = usage }

        guard let choice = chunk["choices"]?[0] else { return parts }
        if let reason = choice["finish_reason"]?.stringValue { finishReason = reason }
        guard let delta = choice["delta"] else { return parts }

        // Reasoning.
        let details = ReasoningDetails.entries(delta["reasoning_details"])
        for detail in details {
            ReasoningDetails.accumulate(detail, into: &reasoningDetails)
        }
        if !hasStartedText {
            if !details.isEmpty {
                // Opened even for a detail with no visible text: an encrypted-only block still has
                // to close with its details for the next turn to receive them.
                openReasoning(&parts)
                for text in details.compactMap(ReasoningDetails.visibleText) where !text.isEmpty {
                    parts.append(.reasoningDelta(id: reasoningBlockID!, delta: text))
                }
            } else if let reasoning = delta["reasoning"]?.stringValue, !reasoning.isEmpty {
                // The legacy plain-text channel, used only when no structured details arrive.
                openReasoning(&parts)
                parts.append(.reasoningDelta(id: reasoningBlockID!, delta: reasoning))
            }
        }
        // Details that arrive after the answer has begun — a late signature, typically — are still
        // accumulated. They reach the next turn through the first tool call and the finish
        // metadata, without reopening a reasoning block the reader has already seen close.

        // Text.
        for text in [delta["content"]?.stringValue, delta["refusal"]?.stringValue] {
            guard let text, !text.isEmpty else { continue }
            closeReasoning(&parts)
            hasStartedText = true
            if textBlockID == nil {
                let id = IdentifierGenerator.generate(prefix: "text")
                textBlockID = id
                parts.append(.textStart(id: id))
            }
            parts.append(.textDelta(id: textBlockID!, delta: text))
        }

        // Tool calls.
        for fragment in delta["tool_calls"]?.arrayValue ?? [] {
            closeReasoning(&parts)
            parts += consumeToolCall(fragment)
        }

        // Generated images and citations.
        parts += OpenRouterResponse.images(delta["images"]).map(LanguageModelStreamPart.file)
        parts += OpenRouterResponse.sources(delta["annotations"], providerName: providerName).map(LanguageModelStreamPart.source)
        fileAnnotations += OpenRouterResponse.fileAnnotations(delta["annotations"])

        return parts
    }

    private mutating func openReasoning(_ parts: inout [LanguageModelStreamPart]) {
        guard reasoningBlockID == nil else { return }
        let id = IdentifierGenerator.generate(prefix: "reasoning")
        reasoningBlockID = id
        parts.append(.reasoningStart(id: id))
    }

    private mutating func closeReasoning(_ parts: inout [LanguageModelStreamPart]) {
        guard let id = reasoningBlockID else { return }
        reasoningBlockID = nil
        parts.append(.reasoningEnd(id: id, providerMetadata: detailsMetadata()))
    }

    private func detailsMetadata() -> ProviderMetadata? {
        reasoningDetails.isEmpty ? nil : [providerName: [ReasoningDetails.optionsKey: .array(reasoningDetails)]]
    }

    private mutating func consumeToolCall(_ fragment: JSONValue) -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []
        // A fragment without an index continues the most recent call.
        let index = fragment["index"]?.intValue ?? toolCalls.keys.max() ?? 0
        let name = fragment["function"]?["name"]?.stringValue ?? ""

        var pending = toolCalls[index] ?? PendingToolCall(
            id: OpenRouterResponse.uniqueToolCallID(fragment["id"]?.stringValue, seen: &toolCallIDs),
            name: name
        )
        if !name.isEmpty { pending.name = name }

        if !pending.hasStarted, !pending.name.isEmpty {
            pending.hasStarted = true
            parts.append(.toolInputStart(id: pending.id, toolName: pending.name))
        }
        if let arguments = fragment["function"]?["arguments"]?.stringValue, !arguments.isEmpty {
            pending.arguments += arguments
            if pending.hasStarted { parts.append(.toolInputDelta(id: pending.id, delta: arguments)) }
        }
        toolCalls[index] = pending
        return parts
    }

    /// Closes open blocks, emits the completed tool calls, and finishes.
    ///
    /// Tool calls are emitted here because their arguments only parse once complete.
    mutating func finish() -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []
        closeReasoning(&parts)
        if let id = textBlockID {
            parts.append(.textEnd(id: id))
            textBlockID = nil
        }

        let calls = toolCalls.keys.sorted().compactMap { toolCalls[$0] }.filter { !$0.name.isEmpty }
        for (position, pending) in calls.enumerated() {
            if !pending.hasStarted {
                parts.append(.toolInputStart(id: pending.id, toolName: pending.name))
                parts.append(.toolInputDelta(id: pending.id, delta: pending.arguments))
            }
            parts.append(.toolInputEnd(id: pending.id))

            let arguments = pending.arguments.isEmpty ? "{}" : pending.arguments
            if let input = try? ProviderJSON.parseEmbeddedJSON(arguments, context: "the arguments for '\(pending.name)'") {
                parts.append(.toolCall(ToolCallPart(
                    toolCallID: pending.id,
                    toolName: pending.name,
                    input: input,
                    providerOptions: OpenRouterResponse.toolCallOptions(
                        isFirst: position == 0,
                        details: reasoningDetails,
                        providerName: providerName
                    )
                )))
            } else {
                // Reported as an error the model can see and correct, rather than failing the
                // whole stream.
                parts.append(.toolResult(ToolResultPart(
                    toolCallID: pending.id,
                    toolName: pending.name,
                    output: .errorText("The model produced arguments that are not valid JSON: \(pending.arguments)")
                )))
            }
        }
        toolCalls.removeAll()

        parts.append(.finish(
            finishReason: OpenRouterResponse.finishReason(finishReason, hasToolCalls: !calls.isEmpty),
            usage: OpenRouterResponse.usage(usage),
            providerMetadata: OpenRouterResponse.metadata(
                provider: upstreamProvider,
                usage: usage,
                reasoningDetails: reasoningDetails.isEmpty ? nil : reasoningDetails,
                fileAnnotations: fileAnnotations,
                providerName: providerName
            )
        ))
        return parts
    }
}

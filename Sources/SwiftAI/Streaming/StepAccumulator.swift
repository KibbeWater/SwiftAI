import AIProviderSpec

/// Reassembles a provider's stream parts into the content list a completed step produces.
///
/// A streaming response arrives as identifier-correlated blocks that providers are free to
/// interleave, so the order parts *finish* in is not the order they *started* in. The accumulator
/// reserves a slot when a block opens and fills it when the block closes, which means the final
/// content list reads in the order the model actually produced it — the order a reader would
/// expect — rather than in completion order.
struct StepAccumulator {
    /// A slot reserved for a block, filled when the block closes.
    private var slots: [ModelContent?] = []
    /// Maps an open block's identifier to its reserved slot and accumulated text.
    private var openBlocks: [String: (slot: Int, text: String, kind: BlockKind)] = [:]

    private enum BlockKind {
        case text
        case reasoning
    }

    private(set) var finishReason: FinishReason = .unknown
    private(set) var usage: Usage = .none
    private(set) var warnings: [CallWarning] = []
    private(set) var providerMetadata: ProviderMetadata?
    private(set) var responseID: String?
    private(set) var responseModelID: String?

    /// Consumes one provider part.
    ///
    /// - Returns: The public stream part to publish, or `nil` for parts that carry no
    ///   consumer-visible information.
    mutating func consume(_ part: LanguageModelStreamPart) -> TextStreamPart? {
        switch part {
        case .streamStart(let warnings):
            self.warnings.append(contentsOf: warnings)
            return nil

        case .responseMetadata(let id, let modelID, _):
            responseID = id ?? responseID
            responseModelID = modelID ?? responseModelID
            return nil

        case .textStart(let id, _):
            openBlock(id: id, kind: .text)
            return .textStart(id: id)

        case .textDelta(let id, let delta):
            append(delta, to: id)
            return .textDelta(id: id, delta: delta)

        case .textEnd(let id, _):
            closeBlock(id: id) { .text(TextPart($0)) }
            return .textEnd(id: id)

        case .reasoningStart(let id, _):
            openBlock(id: id, kind: .reasoning)
            return .reasoningStart(id: id)

        case .reasoningDelta(let id, let delta):
            append(delta, to: id)
            return .reasoningDelta(id: id, delta: delta)

        case .reasoningEnd(let id, let metadata):
            // Providers that sign their reasoning attach the signature here, and require it to be
            // replayed verbatim on the next turn.
            let signature = metadata?.namespaces.values
                .compactMap { $0["signature"]?.stringValue }
                .first
            closeBlock(id: id) { .reasoning(ReasoningPart($0, signature: signature, providerOptions: nil)) }
            return .reasoningEnd(id: id)

        case .toolInputStart(let id, let toolName, _, _):
            return .toolInputStart(id: id, toolName: toolName)

        case .toolInputDelta(let id, let delta):
            return .toolInputDelta(id: id, delta: delta)

        case .toolInputEnd(let id):
            return .toolInputEnd(id: id)

        case .toolCall(let call):
            slots.append(.toolCall(call))
            return .toolCall(call)

        case .toolResult(let result):
            slots.append(.toolResult(result))
            return .toolResult(result)

        case .file(let file):
            slots.append(.file(file))
            return .file(file)

        case .source(let source):
            slots.append(.source(source))
            return .source(source)

        case .finish(let reason, let usage, let metadata):
            finishReason = reason
            self.usage = usage
            providerMetadata = ProviderMetadata.merging(providerMetadata, metadata)
            return nil

        case .raw(let value):
            return .raw(value)
        }
    }

    /// The step assembled from everything consumed so far.
    ///
    /// A block left open — because the provider ended the stream without closing it — is still
    /// emitted with whatever text arrived, rather than being dropped.
    func finalize(request: RequestInfo?, response: ResponseInfo?) -> RawStep {
        var content = slots
        for (_, block) in openBlocks where !block.text.isEmpty {
            content[block.slot] = block.kind == .text
                ? .text(TextPart(block.text))
                : .reasoning(ReasoningPart(block.text))
        }

        var responseInfo = response ?? ResponseInfo()
        responseInfo.id = responseInfo.id ?? responseID
        responseInfo.modelID = responseInfo.modelID ?? responseModelID

        return RawStep(
            content: content.compactMap { $0 },
            finishReason: finishReason,
            usage: usage,
            warnings: warnings,
            providerMetadata: providerMetadata,
            request: request,
            response: responseInfo
        )
    }

    // MARK: - Block bookkeeping

    private mutating func openBlock(id: String, kind: BlockKind) {
        slots.append(nil)
        openBlocks[id] = (slot: slots.count - 1, text: "", kind: kind)
    }

    private mutating func append(_ delta: String, to id: String) {
        guard openBlocks[id] != nil else {
            // A delta with no matching start: reserve a slot so nothing is lost.
            openBlock(id: id, kind: .text)
            openBlocks[id]?.text = delta
            return
        }
        openBlocks[id]?.text += delta
    }

    private mutating func closeBlock(id: String, makeContent: (String) -> ModelContent) {
        guard let block = openBlocks.removeValue(forKey: id) else { return }
        // An empty block carries no information and would render as a stray blank part.
        guard !block.text.isEmpty else { return }
        slots[block.slot] = makeContent(block.text)
    }
}

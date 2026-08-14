import AIProviderSpec

/// An event from a streaming generation.
///
/// This is the provider's stream vocabulary plus what the loop adds around it: step boundaries,
/// the results of tools the SDK ran, and a final summary. Consumers that only want the answer can
/// use ``StreamTextResult/textStream`` instead; this is for interfaces that show reasoning, tool
/// activity, or progress through a multi-step run.
///
/// ```swift
/// for try await part in stream.fullStream {
///     switch part {
///     case .textDelta(_, let delta): transcript.append(delta)
///     case .toolCall(let call): status = "Looking up \(call.toolName)…"
///     case .toolResult: status = "Thinking…"
///     case .finish(let reason, let usage): log(reason, usage)
///     default: break
///     }
/// }
/// ```
///
/// Text, reasoning, and tool arguments arrive as identifier-correlated blocks: a `…Start`, any
/// number of deltas, then a matching `…End`. Providers may interleave several blocks, so use the
/// identifier rather than assuming deltas are contiguous.
public enum TextStreamPart: Sendable {
    /// The generation has begun. Always the first part.
    case start

    /// A step is beginning, counting from zero.
    case stepStart(stepNumber: Int)

    /// A block of visible text begins.
    case textStart(id: String)
    /// More text for an open block.
    case textDelta(id: String, delta: String)
    /// A block of visible text ends.
    case textEnd(id: String)

    /// A block of reasoning begins.
    case reasoningStart(id: String)
    /// More reasoning for an open block.
    case reasoningDelta(id: String, delta: String)
    /// A block of reasoning ends.
    case reasoningEnd(id: String)

    /// The model begins assembling a tool call.
    case toolInputStart(id: String, toolName: String)
    /// More argument text for an open tool call. Fragments of JSON, rarely valid alone.
    case toolInputDelta(id: String, delta: String)
    /// The model finishes assembling a tool call's arguments.
    case toolInputEnd(id: String)

    /// A complete, parsed tool call.
    case toolCall(ToolCallPart)

    /// A tool is about to run. Emitted for tools the SDK executes.
    case toolWillRun(ToolCallPart)

    /// A tool finished. Check ``ToolResultOutput/isError`` to see whether it succeeded.
    case toolResult(ToolResultPart)

    /// A file the model produced.
    case file(FilePart)

    /// A source the model cited.
    case source(SourcePart)

    /// A step finished, including any tools it ran.
    case stepFinish(StepResult)

    /// The generation finished. Always the last part of a successful stream.
    case finish(finishReason: FinishReason, totalUsage: Usage)

    /// An undecoded provider event, when ``GenerationSettings/includesRawChunks`` is set.
    case raw(JSONValue)
}

extension TextStreamPart {
    /// The text carried by a ``textDelta(id:delta:)``, or `nil` for every other part.
    public var textDelta: String? {
        guard case .textDelta(_, let delta) = self else { return nil }
        return delta
    }

    /// The text carried by a ``reasoningDelta(id:delta:)``, or `nil` for every other part.
    public var reasoningDelta: String? {
        guard case .reasoningDelta(_, let delta) = self else { return nil }
        return delta
    }

    /// A stable discriminator, independent of payload. Useful for asserting on a stream's shape.
    public enum Kind: String, Sendable, Hashable, CaseIterable {
        case start, stepStart
        case textStart, textDelta, textEnd
        case reasoningStart, reasoningDelta, reasoningEnd
        case toolInputStart, toolInputDelta, toolInputEnd
        case toolCall, toolWillRun, toolResult
        case file, source
        case stepFinish, finish, raw
    }

    public var kind: Kind {
        switch self {
        case .start: return .start
        case .stepStart: return .stepStart
        case .textStart: return .textStart
        case .textDelta: return .textDelta
        case .textEnd: return .textEnd
        case .reasoningStart: return .reasoningStart
        case .reasoningDelta: return .reasoningDelta
        case .reasoningEnd: return .reasoningEnd
        case .toolInputStart: return .toolInputStart
        case .toolInputDelta: return .toolInputDelta
        case .toolInputEnd: return .toolInputEnd
        case .toolCall: return .toolCall
        case .toolWillRun: return .toolWillRun
        case .toolResult: return .toolResult
        case .file: return .file
        case .source: return .source
        case .stepFinish: return .stepFinish
        case .finish: return .finish
        case .raw: return .raw
        }
    }
}

extension TextStreamPart: CustomStringConvertible {
    public var description: String {
        switch self {
        case .start: return "start"
        case .stepStart(let stepNumber): return "stepStart(\(stepNumber))"
        case .textStart(let id): return "textStart(\(id))"
        case .textDelta(let id, let delta): return "textDelta(\(id), \(delta.debugDescription))"
        case .textEnd(let id): return "textEnd(\(id))"
        case .reasoningStart(let id): return "reasoningStart(\(id))"
        case .reasoningDelta(let id, let delta): return "reasoningDelta(\(id), \(delta.debugDescription))"
        case .reasoningEnd(let id): return "reasoningEnd(\(id))"
        case .toolInputStart(let id, let toolName): return "toolInputStart(\(id), \(toolName))"
        case .toolInputDelta(let id, let delta): return "toolInputDelta(\(id), \(delta.debugDescription))"
        case .toolInputEnd(let id): return "toolInputEnd(\(id))"
        case .toolCall(let call): return "toolCall(\(call.toolName))"
        case .toolWillRun(let call): return "toolWillRun(\(call.toolName))"
        case .toolResult(let result): return "toolResult(\(result.toolName))"
        case .file(let file): return "file(\(file.mediaType))"
        case .source(let source): return "source(\(source.id))"
        case .stepFinish(let step): return "stepFinish(\(step.finishReason.rawValue))"
        case .finish(let reason, let usage): return "finish(\(reason.rawValue), \(usage))"
        case .raw: return "raw"
        }
    }
}

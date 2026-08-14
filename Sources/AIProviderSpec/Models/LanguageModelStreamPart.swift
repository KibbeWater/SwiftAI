import Foundation

/// A single event in a streaming generation, as produced by a provider.
///
/// The vocabulary is block-structured rather than a flat sequence of deltas. Text, reasoning, and
/// tool arguments each open with a `…Start` part carrying an identifier, emit any number of
/// deltas under that identifier, and close with a matching `…End`. That structure is what allows
/// providers to interleave several blocks — a model that reasons, answers, and assembles a tool
/// call concurrently produces three interleaved runs of deltas that consumers can still
/// reassemble correctly.
///
/// A well-formed stream:
///
/// 1. begins with ``streamStart(warnings:)``,
/// 2. optionally reports ``responseMetadata(id:modelID:timestamp:)``,
/// 3. emits any number of content parts,
/// 4. ends with ``finish(finishReason:usage:providerMetadata:)``.
///
/// A stream that fails after it has begun terminates by throwing from the underlying
/// `AsyncThrowingStream` rather than emitting a final part.
public enum LanguageModelStreamPart: Sendable {
    /// The first part of every stream, carrying any settings the provider could not honor.
    ///
    /// Warnings arrive here rather than on the response because a streaming call returns before
    /// the provider has finished validating the request.
    case streamStart(warnings: [CallWarning])

    /// Identifying information about the response, once the provider reports it.
    case responseMetadata(id: String? = nil, modelID: String? = nil, timestamp: Date? = nil)

    /// A block of visible text begins.
    case textStart(id: String, providerMetadata: ProviderMetadata? = nil)

    /// More text for an open block.
    case textDelta(id: String, delta: String)

    /// A block of visible text ends.
    case textEnd(id: String, providerMetadata: ProviderMetadata? = nil)

    /// A block of reasoning begins.
    case reasoningStart(id: String, providerMetadata: ProviderMetadata? = nil)

    /// More reasoning for an open block.
    case reasoningDelta(id: String, delta: String)

    /// A block of reasoning ends.
    ///
    /// - Parameter providerMetadata: Where a provider returns a signature attesting to the
    ///   reasoning, it appears here so it can be replayed on the next turn.
    case reasoningEnd(id: String, providerMetadata: ProviderMetadata? = nil)

    /// The model begins assembling a tool call.
    ///
    /// - Parameters:
    ///   - id: The tool call identifier, matching the eventual ``toolCall(_:)``.
    ///   - toolName: The tool being called.
    ///   - providerExecuted: Whether the provider will run the tool itself.
    ///   - isDynamic: Whether the tool was registered without a compile-time Swift type.
    case toolInputStart(
        id: String,
        toolName: String,
        providerExecuted: Bool = false,
        isDynamic: Bool = false
    )

    /// More argument text for an open tool call.
    ///
    /// Deltas are fragments of JSON and are rarely valid on their own. Consumers that want to
    /// show arguments as they arrive should accumulate them and parse in
    /// ``JSONValue/ParsingMode/partial`` mode.
    case toolInputDelta(id: String, delta: String)

    /// The model finishes assembling a tool call's arguments.
    case toolInputEnd(id: String)

    /// A complete, parsed tool call.
    ///
    /// Providers emit this once the arguments are complete and valid. Providers that do not
    /// stream arguments emit a matching start/delta/end triple immediately before it, so that
    /// consumers can rely on one consistent shape.
    case toolCall(ToolCallPart)

    /// The result of a tool the provider executed itself.
    case toolResult(ToolResultPart)

    /// A file the model produced, such as a generated image.
    case file(FilePart)

    /// A source the model cited.
    case source(SourcePart)

    /// The final part of a successful stream.
    case finish(
        finishReason: FinishReason,
        usage: Usage = .none,
        providerMetadata: ProviderMetadata? = nil
    )

    /// An undecoded provider event.
    ///
    /// Emitted only when ``LanguageModelCallOptions/includesRawChunks`` is set. Use it to reach
    /// provider features this SDK does not model yet, but prefer the typed parts where they
    /// exist, since raw payloads change without notice.
    case raw(JSONValue)
}

// MARK: - Inspection

extension LanguageModelStreamPart {
    /// A stable discriminator for the part, independent of its payload.
    ///
    /// Useful for asserting on the shape of a stream in tests and for metrics, where comparing
    /// whole parts would be noisy.
    public enum Kind: String, Sendable, Hashable, CaseIterable {
        case streamStart, responseMetadata
        case textStart, textDelta, textEnd
        case reasoningStart, reasoningDelta, reasoningEnd
        case toolInputStart, toolInputDelta, toolInputEnd
        case toolCall, toolResult
        case file, source
        case finish, raw
    }

    public var kind: Kind {
        switch self {
        case .streamStart: return .streamStart
        case .responseMetadata: return .responseMetadata
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
        case .toolResult: return .toolResult
        case .file: return .file
        case .source: return .source
        case .finish: return .finish
        case .raw: return .raw
        }
    }

    /// The identifier of the block this part belongs to, for the parts that belong to one.
    public var blockID: String? {
        switch self {
        case .textStart(let id, _), .textDelta(let id, _), .textEnd(let id, _),
             .reasoningStart(let id, _), .reasoningDelta(let id, _), .reasoningEnd(let id, _),
             .toolInputStart(let id, _, _, _), .toolInputDelta(let id, _), .toolInputEnd(let id):
            return id
        case .toolCall(let call): return call.toolCallID
        case .toolResult(let result): return result.toolCallID
        case .streamStart, .responseMetadata, .file, .source, .finish, .raw:
            return nil
        }
    }

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
}

extension LanguageModelStreamPart: CustomStringConvertible {
    public var description: String {
        switch self {
        case .streamStart(let warnings):
            return warnings.isEmpty ? "streamStart" : "streamStart(\(warnings.count) warning(s))"
        case .responseMetadata(let id, let modelID, _):
            return "responseMetadata(id: \(id ?? "-"), model: \(modelID ?? "-"))"
        case .textStart(let id, _): return "textStart(\(id))"
        case .textDelta(let id, let delta): return "textDelta(\(id), \(delta.debugDescription))"
        case .textEnd(let id, _): return "textEnd(\(id))"
        case .reasoningStart(let id, _): return "reasoningStart(\(id))"
        case .reasoningDelta(let id, let delta): return "reasoningDelta(\(id), \(delta.debugDescription))"
        case .reasoningEnd(let id, _): return "reasoningEnd(\(id))"
        case .toolInputStart(let id, let toolName, _, _): return "toolInputStart(\(id), \(toolName))"
        case .toolInputDelta(let id, let delta): return "toolInputDelta(\(id), \(delta.debugDescription))"
        case .toolInputEnd(let id): return "toolInputEnd(\(id))"
        case .toolCall(let call): return "toolCall(\(call.toolName), \(call.input))"
        case .toolResult(let result): return "toolResult(\(result.toolName))"
        case .file(let file): return "file(\(file.mediaType))"
        case .source(let source): return "source(\(source.id))"
        case .finish(let reason, let usage, _): return "finish(\(reason.rawValue), \(usage))"
        case .raw: return "raw"
        }
    }
}

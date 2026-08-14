/// Why a model stopped producing output.
public enum FinishReason: String, Sendable, Hashable, CaseIterable, Codable {
    /// The model reached a natural stopping point, or emitted a configured stop sequence.
    case stop

    /// Generation hit the output token limit. The response is truncated.
    case length

    /// The provider's safety systems interrupted generation.
    case contentFilter = "content-filter"

    /// The model asked to call one or more tools and is waiting for their results.
    ///
    /// This is the signal the tool loop uses to decide whether another step is warranted.
    case toolCalls = "tool-calls"

    /// Generation failed partway through.
    case error

    /// The model stopped for a provider-specific reason with no equivalent here.
    case other

    /// The provider did not report a reason, or reported one this SDK does not recognize.
    case unknown

    /// Whether the response is incomplete and continuing generation might be appropriate.
    public var isIncomplete: Bool {
        switch self {
        case .length, .toolCalls: return true
        case .stop, .contentFilter, .error, .other, .unknown: return false
        }
    }
}

/// A setting or capability the provider could not honor.
///
/// Warnings exist so that a call using a setting a provider lacks still succeeds. Providers throw
/// ``UnsupportedFunctionalityError`` only when a request cannot be served at all; anything that
/// can be approximated or ignored produces a warning instead. Warnings surface on every result
/// type and, when streaming, arrive as the first stream part.
public enum CallWarning: Sendable, Hashable {
    /// A generation setting has no equivalent on this provider and was dropped.
    case unsupportedSetting(setting: String, details: String? = nil)

    /// A supplied tool cannot be represented and was not sent to the model.
    case unsupportedTool(toolName: String, details: String? = nil)

    /// Anything else the provider wants the caller to know.
    case other(message: String)
}

extension CallWarning: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unsupportedSetting(let setting, let details):
            let base = "The '\(setting)' setting is not supported by this model and was ignored."
            return details.map { "\(base) \($0)" } ?? base
        case .unsupportedTool(let toolName, let details):
            let base = "The tool '\(toolName)' is not supported by this model and was not sent."
            return details.map { "\(base) \($0)" } ?? base
        case .other(let message):
            return message
        }
    }
}

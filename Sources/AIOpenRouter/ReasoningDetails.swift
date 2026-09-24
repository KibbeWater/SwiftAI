import AIProviderSpec

/// OpenRouter's `reasoning_details`: the structured, replayable form of a model's reasoning.
///
/// OpenRouter normalizes every upstream's reasoning into one array of entries — visible text
/// (`reasoning.text`, signed by Anthropic and Gemini), summaries (`reasoning.summary`, from
/// OpenAI), and opaque blobs (`reasoning.encrypted`). The array has to be sent back on the next
/// assistant turn for the upstream to continue its reasoning rather than start over, and several
/// upstreams reject a continuation whose reasoning is missing, altered, or duplicated.
///
/// Entries are kept as raw JSON rather than decoded into a struct. They are round-tripped, not
/// interpreted, and a field this type did not know about would otherwise be lost on the way back.
enum ReasoningDetails {
    static let optionsKey = "reasoning_details"

    /// The entries of a `reasoning_details` value, skipping any that are not objects with a type.
    ///
    /// A malformed entry is dropped rather than failing the response, because one bad entry should
    /// not cost the caller the answer.
    static func entries(_ value: JSONValue?) -> [JSONValue] {
        (value?.arrayValue ?? []).filter { $0["type"]?.stringValue != nil }
    }

    /// The text a detail contributes to the visible reasoning.
    static func visibleText(_ detail: JSONValue) -> String? {
        switch detail["type"]?.stringValue {
        case "reasoning.text": return detail["text"]?.stringValue ?? ""
        case "reasoning.summary": return detail["summary"]?.stringValue
        default: return nil
        }
    }

    // MARK: - Streaming accumulation

    /// Folds streamed deltas into complete entries.
    ///
    /// Each delta carries a fragment of one entry. Consecutive text fragments belong to the same
    /// entry, as do consecutive summary fragments, so they are merged by *type change* rather than
    /// by `index`: OpenAI's upstream sends every delta with `index: 0`, which would otherwise fold
    /// separate summaries into one. Encrypted entries arrive whole and are never merged.
    static func accumulate(_ delta: JSONValue, into accumulated: inout [JSONValue]) {
        guard let type = delta["type"]?.stringValue,
              case .object(var incoming) = delta else { return }

        guard type == "reasoning.text" || type == "reasoning.summary",
              case .object(var last)? = accumulated.last,
              last["type"]?.stringValue == type else {
            accumulated.append(delta)
            return
        }

        let textKey = type == "reasoning.text" ? "text" : "summary"
        let joined = (last[textKey]?.stringValue ?? "") + (incoming[textKey]?.stringValue ?? "")
        last[textKey] = .string(joined)
        incoming.removeValue(forKey: textKey)
        // The first non-empty value of every other field wins. A signature typically arrives on
        // the final fragment, after the text, and must not be lost.
        for (key, value) in incoming where last[key] == nil || last[key]?.stringValue == "" {
            last[key] = value
        }
        accumulated[accumulated.count - 1] = .object(last)
    }

    // MARK: - Replay

    /// Removes entries an upstream would reject on replay.
    ///
    /// Anthropic and Gemini verify their reasoning text against a signature, and reject a
    /// continuation carrying unsigned text with "Invalid signature in thinking block" or "Corrupted
    /// thought signature". An unsigned entry appears when a stream was cut short before its
    /// signature arrived; dropping it costs only that entry. An entry without a format is treated
    /// as Anthropic's, which is how OpenRouter labels it when the format is omitted.
    static func replayable(_ details: [JSONValue]) -> [JSONValue] {
        details.filter { detail in
            guard detail["type"]?.stringValue == "reasoning.text" else { return true }
            let format = detail["format"]?.stringValue ?? "anthropic-claude-v1"
            guard format == "anthropic-claude-v1" || format == "google-gemini-v1" else { return true }
            return !(detail["signature"]?.stringValue ?? "").isEmpty
        }
    }

    /// Tracks entries already sent in a prompt, so none is sent twice.
    ///
    /// The same reasoning reaches a prompt more than once — on a reasoning part and again on the
    /// tool call it preceded, or across turns when history is rebuilt — and OpenRouter rejects a
    /// request with "Duplicate item found with id". The keys mirror OpenRouter's own
    /// deduplication and share one key space, as its do.
    struct DuplicateTracker {
        private var seen: Set<String> = []

        /// Records a detail, returning `false` if it was already sent or can never be identified.
        mutating func admit(_ detail: JSONValue) -> Bool {
            let key: String?
            switch detail["type"]?.stringValue {
            case "reasoning.summary":
                key = detail["summary"]?.stringValue
            case "reasoning.encrypted":
                key = detail["id"]?.stringValue ?? detail["data"]?.stringValue
            case "reasoning.text":
                key = nonEmpty(detail["text"]?.stringValue) ?? nonEmpty(detail["signature"]?.stringValue)
            default:
                // An unknown type is passed through; OpenRouter decides what to do with it.
                return true
            }
            // A text entry with neither text nor a signature carries nothing and cannot be
            // deduplicated, so it is dropped.
            guard let key else { return false }
            return seen.insert(key).inserted
        }

        private func nonEmpty(_ value: String?) -> String? {
            guard let value, !value.isEmpty else { return nil }
            return value
        }
    }
}

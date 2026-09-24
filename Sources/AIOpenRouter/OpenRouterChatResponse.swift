import AIProviderSpec
import AIProviderUtils
import Foundation

/// Reads OpenRouter responses.
///
/// Responses are read as raw JSON rather than decoded into `Codable` structs. Reasoning details
/// and annotations are replayed verbatim on later turns, and a snake-case decoding strategy would
/// silently rename the keys inside them.
enum OpenRouterResponse {
    // MARK: - Errors

    /// Reads OpenRouter's error shape, preferring the upstream provider's own words.
    ///
    /// When a routed provider rejects a request, OpenRouter's top-level message is a generic
    /// "Provider returned error"; the useful explanation is the upstream's raw body, tucked into
    /// `error.metadata.raw` as a JSON string. The upstream's name is prefixed so the reader knows
    /// which of OpenRouter's providers said it.
    static let errorMapper = ProviderErrorMapper { body in
        guard let error = body["error"] else { return ProviderErrorMapper.standard.extractMessage(body) }
        let message = error["message"]?.stringValue ?? error.stringValue
        let upstream = error["metadata"]?["raw"].flatMap(rawMessage)
        let detail = upstream.flatMap { $0 == message ? nil : $0 } ?? message
        guard let detail else { return nil }
        guard let providerName = error["metadata"]?["provider_name"]?.stringValue else { return detail }
        return "[\(providerName)] \(detail)"
    }

    /// Digs a message out of an upstream error body, which may be an object, JSON in a string, or
    /// plain text.
    private static func rawMessage(_ raw: JSONValue) -> String? {
        switch raw {
        case .string(let text):
            if let parsed = try? JSONValue.parse(text), parsed.objectValue != nil {
                return rawMessage(parsed)
            }
            return text.isEmpty ? nil : text
        case .object(let members):
            for key in ["message", "error", "detail", "details", "msg"] {
                if let value = members[key], let message = rawMessage(value) { return message }
            }
            return nil
        default:
            return nil
        }
    }

    /// Throws if a successful response carries an error instead of a completion.
    ///
    /// OpenRouter can answer `200 OK` with `{"error": …}` when an upstream fails after the
    /// response has begun, which would otherwise surface as a confusing "no choices" failure.
    static func throwIfError(_ body: JSONValue, url: URL, headers: [String: String], statusCode: Int = 200) throws {
        guard let error = body["error"], error.objectValue != nil || error.stringValue != nil else { return }
        let code = error["code"]?.intValue
        throw APICallError(
            message: "openrouter returned an error: \(errorMapper.extractMessage(body) ?? "unknown error")",
            url: url,
            statusCode: statusCode,
            responseHeaders: headers,
            responseBody: body.serialized(),
            isRetryable: code.map { APICallError.defaultIsRetryable(statusCode: $0) } ?? false,
            data: body
        )
    }

    // MARK: - Content

    /// Converts a buffered completion's message into content parts.
    static func content(of message: JSONValue, providerName: String) throws -> [ModelContent] {
        var content: [ModelContent] = []

        let details = ReasoningDetails.entries(message["reasoning_details"])
        let reasoningText = details.isEmpty
            ? (message["reasoning"]?.stringValue ?? "")
            : details.compactMap(ReasoningDetails.visibleText).joined()
        if !details.isEmpty || !reasoningText.isEmpty {
            content.append(.reasoning(ReasoningPart(
                reasoningText,
                providerOptions: details.isEmpty ? nil : [providerName: [ReasoningDetails.optionsKey: .array(details)]]
            )))
        }

        if let text = message["content"]?.stringValue, !text.isEmpty {
            content.append(.text(TextPart(text)))
        }
        if let refusal = message["refusal"]?.stringValue, !refusal.isEmpty {
            content.append(.text(TextPart(refusal)))
        }

        var seenIDs: Set<String> = []
        for call in message["tool_calls"]?.arrayValue ?? [] {
            guard let name = call["function"]?["name"]?.stringValue else { continue }
            let id = uniqueToolCallID(call["id"]?.stringValue, seen: &seenIDs)
            let arguments = call["function"]?["arguments"]?.stringValue ?? ""
            content.append(.toolCall(ToolCallPart(
                toolCallID: id,
                toolName: name,
                input: try ProviderJSON.parseEmbeddedJSON(arguments.isEmpty ? "{}" : arguments, context: "the arguments for '\(name)'"),
                providerOptions: toolCallOptions(isFirst: seenIDs.count == 1, details: details, providerName: providerName)
            )))
        }

        content += images(message["images"]).map(ModelContent.file)
        content += sources(message["annotations"], providerName: providerName).map(ModelContent.source)
        return content
    }

    /// The full reasoning details ride on the first tool call as well as on the reasoning part.
    ///
    /// A tool loop replays the call even when an application drops reasoning parts from its
    /// history, and Anthropic and Gemini refuse a tool continuation whose signed reasoning is
    /// missing. Only the first call carries them, so parallel calls do not replay one thinking
    /// block several times.
    static func toolCallOptions(isFirst: Bool, details: [JSONValue], providerName: String) -> ProviderOptions? {
        guard isFirst, !details.isEmpty else { return nil }
        return [providerName: [ReasoningDetails.optionsKey: .array(details)]]
    }

    /// Returns the wire identifier, or a fresh one when it is missing or repeated.
    ///
    /// Some upstreams return empty or duplicate identifiers for parallel calls, and matching
    /// results to calls by identifier breaks on either.
    static func uniqueToolCallID(_ id: String?, seen: inout Set<String>) -> String {
        var resolved = id ?? ""
        if resolved.isEmpty || seen.contains(resolved) {
            resolved = IdentifierGenerator.generate(prefix: "call")
        }
        seen.insert(resolved)
        return resolved
    }

    /// Images the model generated, which arrive as data URLs.
    static func images(_ value: JSONValue?) -> [FilePart] {
        (value?.arrayValue ?? []).compactMap { image in
            guard let url = image["image_url"]?["url"]?.stringValue,
                  url.hasPrefix("data:"),
                  let comma = url.firstIndex(of: ",") else { return nil }
            let header = url[url.index(url.startIndex, offsetBy: 5)..<comma]
            let mediaType = header.split(separator: ";").first.map(String.init) ?? "image/jpeg"
            guard let data = Data(base64Encoded: String(url[url.index(after: comma)...])) else { return nil }
            return FilePart.data(data, mediaType: mediaType.isEmpty ? "image/jpeg" : mediaType)
        }
    }

    /// Web search citations, as sources.
    static func sources(_ value: JSONValue?, providerName: String) -> [SourcePart] {
        (value?.arrayValue ?? []).compactMap { annotation in
            guard annotation["type"]?.stringValue == "url_citation",
                  let citation = annotation["url_citation"],
                  let urlString = citation["url"]?.stringValue,
                  let url = URL(string: urlString) else { return nil }
            var metadata: [String: JSONValue] = [:]
            if let content = citation["content"] { metadata["content"] = content }
            if let start = citation["start_index"] { metadata["startIndex"] = start }
            if let end = citation["end_index"] { metadata["endIndex"] = end }
            return SourcePart(
                id: urlString,
                kind: .url(url),
                title: citation["title"]?.stringValue,
                providerMetadata: metadata.isEmpty ? nil : ProviderMetadata([providerName: metadata])
            )
        }
    }

    /// File annotations, which record how OpenRouter parsed an attached document.
    ///
    /// Returned in metadata so they can be sent back on the next turn, sparing a re-parse.
    static func fileAnnotations(_ value: JSONValue?) -> [JSONValue] {
        (value?.arrayValue ?? []).filter {
            let type = $0["type"]?.stringValue
            return type == "file" || type == "file_annotation"
        }
    }

    // MARK: - Finish reason and usage

    /// Maps OpenRouter's finish reason, reporting tool calls whenever there are any.
    ///
    /// Upstreams disagree about how a turn with tool calls ends: Gemini 3 reports `stop` when its
    /// calls come with encrypted reasoning, and several others report a reason OpenRouter does not
    /// recognize. Taken at face value, either would end a tool loop that should continue.
    static func finishReason(_ raw: String?, hasToolCalls: Bool) -> FinishReason {
        let reason: FinishReason = switch raw {
        case "stop": .stop
        case "length": .length
        case "content_filter": .contentFilter
        case "tool_calls", "function_call": .toolCalls
        case "error": .error
        case nil: .unknown
        default: .other
        }
        guard hasToolCalls else { return reason }
        return reason == .stop || reason == .other || reason == .unknown ? .toolCalls : reason
    }

    static func usage(_ value: JSONValue?) -> Usage {
        guard let value else { return .none }
        return Usage(
            inputTokens: value["prompt_tokens"]?.intValue,
            outputTokens: value["completion_tokens"]?.intValue,
            totalTokens: value["total_tokens"]?.intValue,
            reasoningTokens: value["completion_tokens_details"]?["reasoning_tokens"]?.intValue,
            cachedInputTokens: value["prompt_tokens_details"]?["cached_tokens"]?.intValue
        )
    }

    /// The values OpenRouter reports that have no place on ``Usage``.
    static func metadata(
        provider: String?,
        usage: JSONValue?,
        reasoningDetails: [JSONValue]?,
        fileAnnotations: [JSONValue],
        providerName: String
    ) -> ProviderMetadata? {
        var values: [String: JSONValue] = [:]
        if let provider { values["provider"] = .string(provider) }
        if let cost = usage?["cost"], !cost.isNull { values["cost"] = cost }
        if let upstream = usage?["cost_details"]?["upstream_inference_cost"], !upstream.isNull {
            values["upstreamInferenceCost"] = upstream
        }
        if let cacheWrites = usage?["prompt_tokens_details"]?["cache_write_tokens"], !cacheWrites.isNull {
            values["cacheWriteTokens"] = cacheWrites
        }
        if let reasoningDetails { values[ReasoningDetails.optionsKey] = .array(reasoningDetails) }
        if !fileAnnotations.isEmpty { values["annotations"] = .array(fileAnnotations) }
        return values.isEmpty ? nil : ProviderMetadata([providerName: values])
    }
}

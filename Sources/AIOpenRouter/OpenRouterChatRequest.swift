import AIProviderSpec
import AIProviderUtils
import Foundation

/// Builds an OpenRouter chat completions request from the normalized call options.
///
/// OpenRouter extends Chat Completions in ways that touch nearly every part of the request —
/// system messages as content arrays so they can carry `cache_control`, video parts, a `name` on
/// tool messages, multimodal tool results, and `reasoning_details` on assistant turns — which is
/// why it has its own builder rather than a set of quirks on the shared one.
struct OpenRouterChatRequestBuilder {
    let modelID: String
    let providerName: String
    let extraBody: [String: JSONValue]

    struct Built {
        var body: JSONValue
        var warnings: [CallWarning]
    }

    func build(_ options: LanguageModelCallOptions, stream: Bool) throws -> Built {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = [
            "model": .string(modelID),
            "messages": .array(try convertMessages(options.prompt)),
        ]

        if let value = options.maxOutputTokens { fields["max_tokens"] = .int(value) }
        if let value = options.temperature { fields["temperature"] = .double(value) }
        if let value = options.topP { fields["top_p"] = .double(value) }
        // OpenRouter forwards `top_k` to the upstreams that support it, unlike OpenAI.
        if let value = options.topK { fields["top_k"] = .int(value) }
        if let value = options.presencePenalty { fields["presence_penalty"] = .double(value) }
        if let value = options.frequencyPenalty { fields["frequency_penalty"] = .double(value) }
        if let value = options.seed { fields["seed"] = .int(value) }
        if !options.stopSequences.isEmpty {
            fields["stop"] = .array(options.stopSequences.map(JSONValue.string))
        }

        if stream {
            fields["stream"] = .bool(true)
            // Always requested. The upstream AI SDK provider omits this by default because some
            // routed providers once rejected it; OpenRouter itself now accepts it everywhere, and
            // without it a streamed response reports neither tokens nor cost.
            fields["stream_options"] = .object(["include_usage": .bool(true)])
        }

        switch options.responseFormat {
        case nil, .text:
            break
        case .json(let schema?, let name, let description):
            var jsonSchema: [String: JSONValue] = [
                "name": .string(name ?? "response"),
                "schema": schema.strictified().jsonValue(options: .openAIStrict),
                "strict": .bool(true),
            ]
            if let description { jsonSchema["description"] = .string(description) }
            fields["response_format"] = .object(["type": "json_schema", "json_schema": .object(jsonSchema)])
        case .json(nil, _, _):
            fields["response_format"] = .object(["type": "json_object"])
        }

        let tools = convertTools(options.tools, warnings: &warnings)
        if !tools.isEmpty {
            fields["tools"] = .array(tools)
            if let choice = toolChoice(options.toolChoice) { fields["tool_choice"] = choice }
        }

        // Provider-level extra fields first, then per-call provider options, so the more specific
        // setting wins. Both are merged verbatim: OpenRouter's request surface — `models`,
        // `provider`, `reasoning`, `plugins`, `usage`, `transforms` — grows faster than any typed
        // wrapper could track, and the wire names are the documented ones.
        for (key, value) in extraBody { fields[key] = value }
        var callOptions = options.providerOptions?[providerName] ?? [:]
        // `cacheControl` is accepted in camel case, matching the other providers' spelling.
        if let cacheControl = callOptions.removeValue(forKey: "cacheControl"), callOptions["cache_control"] == nil {
            callOptions["cache_control"] = cacheControl
        }
        for (key, value) in callOptions { fields[key] = value }

        return Built(body: .object(fields), warnings: warnings)
    }

    // MARK: - Messages

    private func convertMessages(_ messages: [ModelMessage]) throws -> [JSONValue] {
        var converted: [JSONValue] = []
        // One tracker for the whole prompt: the same reasoning can reach it from several turns.
        var tracker = ReasoningDetails.DuplicateTracker()

        for message in messages {
            switch message {
            case .system(let system):
                // Always a content array, so the message can carry a cache breakpoint.
                var part: [String: JSONValue] = ["type": "text", "text": .string(system.content)]
                if let cacheControl = cacheControl(system.providerOptions) { part["cache_control"] = cacheControl }
                converted.append(.object(["role": "system", "content": .array([.object(part)])]))

            case .user(let user):
                converted.append(.object([
                    "role": "user",
                    "content": try userContent(user.content, messageCacheControl: cacheControl(user.providerOptions)),
                ]))

            case .assistant(let assistant):
                converted.append(assistantMessage(assistant, tracker: &tracker))

            case .tool(let toolMessage):
                // One message per result, as Chat Completions requires.
                for result in toolMessage.content {
                    var fields: [String: JSONValue] = [
                        "role": "tool",
                        "tool_call_id": .string(result.toolCallID),
                        // OpenRouter forwards the tool name to upstreams, such as Gemini, that
                        // match results to calls by name rather than identifier.
                        "name": .string(result.toolName),
                        "content": try toolResultContent(result.output),
                    ]
                    let cache = cacheControl(result.providerOptions) ?? cacheControl(toolMessage.providerOptions)
                    if let cache { fields["cache_control"] = cache }
                    converted.append(.object(fields))
                }
            }
        }
        return converted
    }

    /// Renders user content, using the plain string form when it is a single uncached text part.
    private func userContent(_ content: [UserContent], messageCacheControl: JSONValue?) throws -> JSONValue {
        if content.count == 1, case .text(let text) = content[0],
           messageCacheControl == nil, cacheControl(text.providerOptions) == nil {
            return .string(text.text)
        }

        // A message-level breakpoint goes on the last text part: a breakpoint marks the end of the
        // cached prefix, and the root of a message cannot carry one.
        let lastTextIndex = content.lastIndex { if case .text = $0 { return true } else { return false } }

        var parts: [JSONValue] = []
        for (index, part) in content.enumerated() {
            switch part {
            case .text(let text):
                var fields: [String: JSONValue] = ["type": "text", "text": .string(text.text)]
                let cache = cacheControl(text.providerOptions) ?? (index == lastTextIndex ? messageCacheControl : nil)
                if let cache { fields["cache_control"] = cache }
                parts.append(.object(fields))
            case .file(let file):
                parts.append(try filePart(file))
            }
        }
        return .array(parts)
    }

    private func filePart(_ file: FilePart) throws -> JSONValue {
        let reference = file.url?.absoluteString ?? file.dataURI ?? ""
        var part: [String: JSONValue]

        if file.isImage {
            part = ["type": "image_url", "image_url": .object(["url": .string(reference)])]
        } else if file.mediaType.hasPrefix("video/") {
            part = ["type": "video_url", "video_url": .object(["url": .string(reference)])]
        } else if file.mediaType.hasPrefix("audio/") {
            guard let base64 = file.base64EncodedString else {
                throw UnsupportedFunctionalityError(
                    functionality: "audio input by URL",
                    message: "OpenRouter accepts audio only as data. Download it and pass the bytes."
                )
            }
            part = ["type": "input_audio", "input_audio": .object(["data": .string(base64), "format": .string(try audioFormat(file.mediaType))])]
        } else {
            let filename = file.providerOptions?.value("filename", for: providerName)?.stringValue ?? file.filename ?? ""
            part = ["type": "file", "file": .object(["filename": .string(filename), "file_data": .string(reference)])]
            // OpenRouter ignores a breakpoint on a file it fetches by URL, and some upstreams
            // reject it outright.
            if file.url != nil { return .object(part) }
        }

        if let cache = cacheControl(file.providerOptions) { part["cache_control"] = cache }
        return .object(part)
    }

    /// Maps a MIME type to the audio format name OpenRouter expects.
    private func audioFormat(_ mediaType: String) throws -> String {
        let subtype = mediaType.split(separator: "/").last.map { $0.lowercased() } ?? ""
        switch subtype {
        case "mpeg", "mp3": return "mp3"
        case "wav", "x-wav", "wave": return "wav"
        case "ogg", "vorbis": return "ogg"
        case "aac", "x-aac": return "aac"
        // `audio/mp4` is an M4A container in practice.
        case "m4a", "x-m4a", "mp4": return "m4a"
        case "aiff", "x-aiff": return "aiff"
        case "flac", "x-flac": return "flac"
        case "pcm16", "pcm24": return subtype
        default:
            throw UnsupportedFunctionalityError(
                functionality: "audio format '\(mediaType)'",
                message: "OpenRouter does not accept '\(mediaType)' audio. Use mp3, wav, ogg, aac, m4a, aiff, flac, or pcm."
            )
        }
    }

    private func assistantMessage(_ assistant: AssistantMessage, tracker: inout ReasoningDetails.DuplicateTracker) -> JSONValue {
        var text = ""
        var reasoningText = ""
        var toolCalls: [JSONValue] = []
        var toolCallDetails: [JSONValue]?
        var reasoningPartDetails: [JSONValue] = []

        for part in assistant.content {
            switch part {
            case .text(let part):
                text += part.text
            case .reasoning(let reasoning):
                reasoningText += reasoning.text
                reasoningPartDetails += ReasoningDetails.entries(
                    reasoning.providerOptions?.value(ReasoningDetails.optionsKey, for: providerName)
                )
            case .toolCall(let call):
                if toolCallDetails == nil {
                    let details = ReasoningDetails.entries(call.providerOptions?.value(ReasoningDetails.optionsKey, for: providerName))
                    if !details.isEmpty { toolCallDetails = details }
                }
                toolCalls.append(.object([
                    "id": .string(call.toolCallID),
                    "type": "function",
                    "function": .object([
                        "name": .string(call.toolName),
                        // Sorted keys keep the serialized prompt identical across turns, which is
                        // what prompt caching keys on.
                        "arguments": .string(call.input.serialized(sortedKeys: true)),
                    ]),
                ]))
            case .file, .source, .toolResult:
                // Chat Completions has no way to replay these on an assistant turn.
                continue
            }
        }

        // Where the details come from, most explicit first. An explicit empty array on the
        // message is honored as-is: DeepSeek requires `reasoning_details: []` to be echoed back.
        let explicit = assistant.providerOptions?.value(ReasoningDetails.optionsKey, for: providerName)
        let source = explicit.map(ReasoningDetails.entries) ?? toolCallDetails ?? reasoningPartDetails
        let details = ReasoningDetails.replayable(source).filter { tracker.admit($0) }

        var fields: [String: JSONValue] = ["role": "assistant"]
        // A turn that is only tool calls must send `content: null`, not an empty string.
        fields["content"] = text.isEmpty ? .null : .string(text)
        if !toolCalls.isEmpty { fields["tool_calls"] = .array(toolCalls) }
        if !details.isEmpty || explicit != nil {
            fields["reasoning_details"] = .array(details)
            if !details.isEmpty, !reasoningText.isEmpty { fields["reasoning"] = .string(reasoningText) }
        }
        // File annotations from an earlier turn, so a parsed PDF is not parsed (and billed) again.
        if let annotations = assistant.providerOptions?.value("annotations", for: providerName) {
            fields["annotations"] = annotations
        }
        if let cache = cacheControl(assistant.providerOptions) { fields["cache_control"] = cache }
        return .object(fields)
    }

    private func toolResultContent(_ output: ToolResultOutput) throws -> JSONValue {
        switch output {
        case .text(let text), .errorText(let text):
            return .string(text)
        case .json(let value), .errorJSON(let value):
            return .string(value.serialized())
        case .content(let parts):
            // OpenRouter accepts multimodal tool results, so a screenshot a tool returns reaches
            // the model as an image rather than being flattened away.
            return .array(try parts.map { part in
                switch part {
                case .text(let text): return .object(["type": "text", "text": .string(text.text)])
                case .file(let file): return try filePart(file)
                }
            })
        }
    }

    /// Reads a cache breakpoint from this provider's options, falling back to Anthropic's.
    ///
    /// Anthropic's spelling is honored because prompts are often written for Anthropic first and
    /// then routed through OpenRouter to the same model.
    private func cacheControl(_ options: ProviderOptions?) -> JSONValue? {
        options?.value("cacheControl", for: providerName)
            ?? options?.value("cache_control", for: providerName)
            ?? options?.value("cacheControl", for: "anthropic")
            ?? options?.value("cache_control", for: "anthropic")
    }

    // MARK: - Tools

    private func convertTools(_ tools: [LanguageModelTool], warnings: inout [CallWarning]) -> [JSONValue] {
        var converted: [JSONValue] = []
        for tool in tools {
            switch tool {
            case .function(let function):
                var entry: [String: JSONValue] = [
                    "type": "function",
                    "function": .object([
                        "name": .string(function.name),
                        "description": .string(function.description),
                        "parameters": function.inputSchema.jsonValue(),
                    ]),
                ]
                // Streams the arguments as they are generated, on upstreams that buffer them by
                // default. It sits beside `function`, not inside it.
                if let eager = function.providerOptions?.value("eager_input_streaming", for: providerName) {
                    entry["eager_input_streaming"] = eager
                }
                converted.append(.object(entry))

            case .providerDefined(let definition) where definition.providerNamespace == providerName:
                // `openrouter.web_search` becomes `openrouter:web_search`, with its arguments in
                // snake case directly on the tool object.
                var entry: [String: JSONValue] = [
                    "type": .string(definition.id.replacingOccurrences(of: ".", with: ":")),
                ]
                for (key, value) in definition.arguments { entry[Self.snakeCase(key)] = value }
                converted.append(.object(entry))

            case .providerDefined(let definition):
                warnings.append(.unsupportedTool(
                    toolName: definition.name,
                    details: "'\(definition.id)' belongs to another provider; OpenRouter only runs its own server tools."
                ))
            }
        }
        return converted
    }

    private func toolChoice(_ choice: ToolChoice?) -> JSONValue? {
        switch choice {
        case nil: return nil
        case .auto: return "auto"
        case .never: return "none"
        case .required: return "required"
        case .tool(let name): return .object(["type": "function", "function": .object(["name": .string(name)])])
        }
    }

    static func snakeCase(_ key: String) -> String {
        var result = ""
        for character in key {
            if character.isUppercase {
                result += "_" + character.lowercased()
            } else {
                result.append(character)
            }
        }
        return result
    }
}

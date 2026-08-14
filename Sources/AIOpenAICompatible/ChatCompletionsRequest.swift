import AIProviderSpec
import AIProviderUtils
import Foundation

/// Builds a Chat Completions request body from the normalized call options.
///
/// The body is assembled as a ``JSONValue`` rather than through `Codable` structs. Chat
/// Completions is full of polymorphic fields — `content` is a string *or* an array, `tool_choice`
/// is a string *or* an object — and expressing those with `Encodable` types means a custom
/// `encode(to:)` for each. Building the JSON directly is both shorter and easier to read against
/// the API documentation.
struct ChatCompletionsRequestBuilder {
    let modelID: String
    let quirks: OpenAICompatibleQuirks

    /// The request body, plus anything the provider could not honor.
    struct Built {
        var body: JSONValue
        var warnings: [CallWarning]
    }

    func build(_ options: LanguageModelCallOptions, stream: Bool) throws -> Built {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = [
            "model": .string(modelID),
        ]

        var messages = try convertMessages(options.prompt, warnings: &warnings)

        // A provider without native structured output gets the schema as an instruction instead,
        // which is markedly better than sending nothing.
        if case .json(let schema?, _, _) = options.responseFormat,
           !quirks.supportsStructuredOutputs {
            messages.insert(
                .object([
                    "role": .string(quirks.systemRole),
                    "content": .string(Self.schemaInstruction(for: schema)),
                ]),
                at: 0
            )
        }
        fields["messages"] = .array(messages)

        if let value = options.maxOutputTokens {
            fields[quirks.usesMaxCompletionTokens ? "max_completion_tokens" : "max_tokens"] = .int(value)
        }
        if let value = options.temperature { fields["temperature"] = .double(value) }
        if let value = options.topP { fields["top_p"] = .double(value) }
        if let value = options.presencePenalty { fields["presence_penalty"] = .double(value) }
        if let value = options.frequencyPenalty { fields["frequency_penalty"] = .double(value) }
        if let value = options.seed { fields["seed"] = .int(value) }
        if !options.stopSequences.isEmpty {
            fields["stop"] = .array(options.stopSequences.map(JSONValue.string))
        }
        if options.topK != nil {
            warnings.append(
                .unsupportedSetting(
                    setting: "topK",
                    details: "The Chat Completions API has no equivalent; use topP instead."
                )
            )
        }

        if stream {
            fields["stream"] = .bool(true)
            // Without this, streamed responses report no usage at all on OpenAI.
            if quirks.supportsStreamUsage {
                fields["stream_options"] = .object(["include_usage": .bool(true)])
            }
        }

        if let responseFormat = try responseFormat(for: options.responseFormat, warnings: &warnings) {
            fields["response_format"] = responseFormat
        }

        let tools = try convertTools(options.tools, warnings: &warnings)
        if !tools.isEmpty {
            fields["tools"] = .array(tools)
            if let choice = toolChoice(options.toolChoice) {
                fields["tool_choice"] = choice
            }
        }

        // Provider options are merged last so callers can reach settings this type does not model.
        for (key, value) in options.providerOptions?[quirks.providerOptionsNamespace] ?? [:] {
            fields[key] = value
        }

        return Built(body: .object(fields), warnings: warnings)
    }

    // MARK: - Messages

    private func convertMessages(
        _ messages: [ModelMessage],
        warnings: inout [CallWarning]
    ) throws -> [JSONValue] {
        var converted: [JSONValue] = []

        for message in messages {
            switch message {
            case .system(let system):
                converted.append(
                    .object(["role": .string(quirks.systemRole), "content": .string(system.content)])
                )

            case .user(let user):
                converted.append(.object(["role": "user", "content": try userContent(user.content, warnings: &warnings)]))

            case .assistant(let assistant):
                converted.append(assistantMessage(assistant.content, warnings: &warnings))

            case .tool(let toolMessage):
                // Chat Completions takes one message per result rather than a list.
                for result in toolMessage.content {
                    converted.append(
                        .object([
                            "role": "tool",
                            "tool_call_id": .string(result.toolCallID),
                            "content": .string(Self.toolResultText(result.output, warnings: &warnings)),
                        ])
                    )
                }
            }
        }
        return converted
    }

    /// Renders user content, using the compact string form when there is nothing but text.
    ///
    /// Some compatible servers only accept a string, so the simple case stays simple.
    private func userContent(
        _ content: [UserContent],
        warnings: inout [CallWarning]
    ) throws -> JSONValue {
        let isPlainText = content.allSatisfy { part in
            if case .text = part { return true }
            return false
        }
        if isPlainText {
            let text = content.compactMap { part -> String? in
                guard case .text(let text) = part else { return nil }
                return text.text
            }.joined()
            return .string(text)
        }

        var parts: [JSONValue] = []
        for part in content {
            switch part {
            case .text(let text):
                parts.append(.object(["type": "text", "text": .string(text.text)]))

            case .file(let file):
                parts.append(try filePart(file, warnings: &warnings))
            }
        }
        return .array(parts)
    }

    private func filePart(_ file: FilePart, warnings: inout [CallWarning]) throws -> JSONValue {
        let reference: String
        switch file.source {
        case .url(let url): reference = url.absoluteString
        case .data: reference = file.dataURI ?? ""
        }

        if file.isImage {
            return .object(["type": "image_url", "image_url": .object(["url": .string(reference)])])
        }
        if file.mediaType.hasPrefix("audio/") {
            guard let base64 = file.base64EncodedString else {
                throw UnsupportedFunctionalityError(
                    functionality: "audio input by URL",
                    message: "Audio must be supplied as data, not a URL."
                )
            }
            let format = file.mediaType.split(separator: "/").last.map(String.init) ?? "mp3"
            return .object([
                "type": "input_audio",
                "input_audio": .object(["data": .string(base64), "format": .string(format)]),
            ])
        }
        // Everything else is offered as a file attachment; servers that do not support it reject
        // the request with a clear message of their own.
        return .object([
            "type": "file",
            "file": .object([
                "filename": .string(file.filename ?? "document"),
                "file_data": .string(reference),
            ]),
        ])
    }

    private func assistantMessage(
        _ content: [ModelContent],
        warnings: inout [CallWarning]
    ) -> JSONValue {
        var text = ""
        var toolCalls: [JSONValue] = []

        for part in content {
            switch part {
            case .text(let part):
                text += part.text
            case .toolCall(let call):
                toolCalls.append(
                    .object([
                        "id": .string(call.toolCallID),
                        "type": "function",
                        "function": .object([
                            "name": .string(call.toolName),
                            "arguments": .string(call.input.serialized()),
                        ]),
                    ])
                )
            case .reasoning, .file, .source, .toolResult:
                // Chat Completions does not accept reasoning, files, or citations as assistant
                // input. Replaying them would be rejected, so they are dropped silently — they
                // carry no instruction the model needs on the next turn.
                continue
            }
        }

        var fields: [String: JSONValue] = ["role": "assistant"]
        // An assistant turn that is only a tool call must send `content: null`, not `""`.
        fields["content"] = text.isEmpty ? .null : .string(text)
        if !toolCalls.isEmpty { fields["tool_calls"] = .array(toolCalls) }
        return .object(fields)
    }

    /// Renders a tool result as the text the model will read.
    static func toolResultText(_ output: ToolResultOutput, warnings: inout [CallWarning]) -> String {
        switch output {
        case .text(let text), .errorText(let text):
            return text
        case .json(let value), .errorJSON(let value):
            return value.serialized()
        case .content(let parts):
            warnings.append(
                .other(
                    message: """
                        A multimodal tool result was flattened to text: the Chat Completions API \
                        accepts only text in tool messages.
                        """
                )
            )
            return parts.compactMap { part -> String? in
                guard case .text(let text) = part else { return nil }
                return text.text
            }.joined(separator: "\n")
        }
    }

    // MARK: - Tools

    private func convertTools(
        _ tools: [LanguageModelTool],
        warnings: inout [CallWarning]
    ) throws -> [JSONValue] {
        var converted: [JSONValue] = []

        for tool in tools {
            switch tool {
            case .function(let function):
                var definition: [String: JSONValue] = [
                    "name": .string(function.name),
                    "description": .string(function.description),
                ]
                if quirks.supportsStrictSchemas {
                    definition["strict"] = .bool(true)
                    definition["parameters"] = function.inputSchema.strictified()
                        .jsonValue(options: .openAIStrict)
                } else {
                    definition["parameters"] = function.inputSchema.jsonValue()
                }
                converted.append(.object(["type": "function", "function": .object(definition)]))

            case .providerDefined(let definition):
                warnings.append(
                    .unsupportedTool(
                        toolName: definition.name,
                        details: "The Chat Completions API has no provider-executed tools."
                    )
                )
            }
        }
        return converted
    }

    private func toolChoice(_ choice: ToolChoice?) -> JSONValue? {
        switch choice {
        case nil, .auto: return .string("auto")
        case .never: return .string("none")
        case .required: return .string("required")
        case .tool(let name):
            return .object(["type": "function", "function": .object(["name": .string(name)])])
        }
    }

    // MARK: - Response format

    private func responseFormat(
        for format: ResponseFormat?,
        warnings: inout [CallWarning]
    ) throws -> JSONValue? {
        switch format {
        case nil, .text:
            return nil

        case .json(let schema, let name, let description):
            guard let schema, quirks.supportsStructuredOutputs else {
                if schema != nil {
                    warnings.append(
                        .other(
                            message: """
                                This model does not support schema-constrained output. The schema \
                                was supplied as an instruction instead, so the response is not \
                                guaranteed to conform.
                                """
                        )
                    )
                }
                return .object(["type": "json_object"])
            }

            var jsonSchema: [String: JSONValue] = [
                "name": .string(name ?? "response"),
                "schema": schema.strictified().jsonValue(options: .openAIStrict),
                "strict": .bool(quirks.supportsStrictSchemas),
            ]
            if let description { jsonSchema["description"] = .string(description) }
            return .object(["type": "json_schema", "json_schema": .object(jsonSchema)])
        }
    }

    /// The instruction used when a provider cannot constrain output to a schema.
    static func schemaInstruction(for schema: JSONSchema) -> String {
        """
        Respond with a single JSON object and nothing else. No prose, no code fences. \
        The object must conform to this JSON Schema:

        \(schema.jsonValue().serialized(sortedKeys: true, prettyPrinted: true))
        """
    }
}

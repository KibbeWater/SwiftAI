import AIProviderSpec
import AIProviderUtils
import Foundation

/// Builds a Messages API request body.
///
/// Anthropic's shape differs from OpenAI's in ways that matter to the conversion:
///
/// - `max_tokens` is required, so a default has to exist.
/// - System instructions are a top-level field, not a message.
/// - Tool results are content blocks inside a **user** message; there is no tool role.
/// - Extended thinking arrives as signed `thinking` blocks that must be replayed verbatim on the
///   next turn, signature included, or the request is rejected.
struct AnthropicRequestBuilder {
    let modelID: String
    let defaultMaxTokens: Int

    struct Built {
        var body: JSONValue
        var warnings: [CallWarning]
    }

    func build(_ options: LanguageModelCallOptions, stream: Bool) throws -> Built {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = [
            "model": .string(modelID),
            // Required by the API, unlike every other provider this package supports.
            "max_tokens": .int(options.maxOutputTokens ?? defaultMaxTokens),
        ]

        let converted = try convertMessages(options.prompt, warnings: &warnings)
        if let system = converted.system { fields["system"] = system }
        fields["messages"] = .array(converted.messages)

        if let value = options.temperature { fields["temperature"] = .double(value) }
        if let value = options.topP { fields["top_p"] = .double(value) }
        if let value = options.topK { fields["top_k"] = .int(value) }
        if !options.stopSequences.isEmpty {
            fields["stop_sequences"] = .array(options.stopSequences.map(JSONValue.string))
        }
        if stream { fields["stream"] = .bool(true) }

        for unsupported in ["presencePenalty", "frequencyPenalty", "seed"] {
            let isSet: Bool
            switch unsupported {
            case "presencePenalty": isSet = options.presencePenalty != nil
            case "frequencyPenalty": isSet = options.frequencyPenalty != nil
            default: isSet = options.seed != nil
            }
            if isSet {
                warnings.append(
                    .unsupportedSetting(
                        setting: unsupported,
                        details: "The Anthropic Messages API has no equivalent."
                    )
                )
            }
        }

        // Anthropic constrains output through a forced tool call rather than a response format.
        if case .json(let schema, let name, let description) = options.responseFormat {
            try applyStructuredOutput(
                schema: schema,
                name: name,
                description: description,
                to: &fields,
                warnings: &warnings
            )
        } else {
            let tools = convertTools(options.tools, warnings: &warnings)
            if !tools.isEmpty {
                fields["tools"] = .array(tools)
                if let choice = toolChoice(options.toolChoice) { fields["tool_choice"] = choice }
            }
        }

        for (key, value) in options.providerOptions?["anthropic"] ?? [:] {
            // `cacheControl` is applied to individual blocks, not the request body.
            guard key != "cacheControl" else { continue }
            fields[key] = value
        }

        return Built(body: .object(fields), warnings: warnings)
    }

    // MARK: - Structured output

    /// Constrains output to a schema by forcing a single tool call.
    ///
    /// The Messages API has no response format, but a tool whose parameters *are* the schema,
    /// combined with `tool_choice`, achieves the same guarantee. The response decoder recognizes
    /// this tool by name and republishes its input as text, so callers see ordinary JSON.
    private func applyStructuredOutput(
        schema: JSONSchema?,
        name: String?,
        description: String?,
        to fields: inout [String: JSONValue],
        warnings: inout [CallWarning]
    ) throws {
        guard let schema else {
            warnings.append(
                .other(
                    message: """
                        Anthropic cannot request JSON without a schema. Supply one, or expect \
                        ordinary text.
                        """
                )
            )
            return
        }

        var tool: [String: JSONValue] = [
            "name": .string(AnthropicStructuredOutput.toolName),
            "description": .string(
                description ?? "Return the result as a structured value matching the schema."
            ),
            "input_schema": schema.jsonValue(),
        ]
        if let name { tool["description"] = .string(description ?? "Return a '\(name)' value.") }

        fields["tools"] = .array([.object(tool)])
        fields["tool_choice"] = .object([
            "type": "tool",
            "name": .string(AnthropicStructuredOutput.toolName),
        ])
    }

    // MARK: - Messages

    private struct ConvertedMessages {
        var system: JSONValue?
        var messages: [JSONValue]
    }

    private func convertMessages(
        _ messages: [ModelMessage],
        warnings: inout [CallWarning]
    ) throws -> ConvertedMessages {
        var system: JSONValue?
        var converted: [JSONValue] = []

        for message in messages {
            switch message {
            case .system(let systemMessage):
                var block: [String: JSONValue] = ["type": "text", "text": .string(systemMessage.content)]
                if let cacheControl = cacheControl(systemMessage.providerOptions) {
                    block["cache_control"] = cacheControl
                }
                system = .array([.object(block)])

            case .user(let user):
                var blocks: [JSONValue] = []
                for part in user.content {
                    blocks.append(try userBlock(part, warnings: &warnings))
                }
                applyCacheControl(user.providerOptions, toLastOf: &blocks)
                converted.append(.object(["role": "user", "content": .array(blocks)]))

            case .assistant(let assistant):
                var blocks: [JSONValue] = []
                for part in assistant.content {
                    if let block = assistantBlock(part) { blocks.append(block) }
                }
                applyCacheControl(assistant.providerOptions, toLastOf: &blocks)
                // An assistant turn with nothing in it is rejected by the API.
                guard !blocks.isEmpty else { continue }
                converted.append(.object(["role": "assistant", "content": .array(blocks)]))

            case .tool(let toolMessage):
                // Tool results are user content here; there is no tool role.
                let blocks = toolMessage.content.map(toolResultBlock)
                converted.append(.object(["role": "user", "content": .array(blocks)]))
            }
        }

        // Consecutive same-role messages are rejected, and tool results become user messages, so
        // a tool round trip would otherwise produce two user turns in a row.
        return ConvertedMessages(system: system, messages: mergeAdjacentRoles(converted))
    }

    private func userBlock(_ part: UserContent, warnings: inout [CallWarning]) throws -> JSONValue {
        switch part {
        case .text(let text):
            var block: [String: JSONValue] = ["type": "text", "text": .string(text.text)]
            if let cacheControl = cacheControl(text.providerOptions) {
                block["cache_control"] = cacheControl
            }
            return .object(block)

        case .file(let file):
            let source: JSONValue
            switch file.source {
            case .url(let url):
                source = .object(["type": "url", "url": .string(url.absoluteString)])
            case .data(let data):
                source = .object([
                    "type": "base64",
                    "media_type": .string(file.mediaType),
                    "data": .string(data.base64EncodedString()),
                ])
            }

            if file.isImage {
                return .object(["type": "image", "source": source])
            }
            if file.mediaType == "application/pdf" || file.mediaType.hasPrefix("text/") {
                return .object(["type": "document", "source": source])
            }
            warnings.append(
                .other(message: "Anthropic does not accept '\(file.mediaType)' attachments; it was dropped.")
            )
            return .object(["type": "text", "text": .string("[unsupported attachment: \(file.mediaType)]")])
        }
    }

    private func assistantBlock(_ part: ModelContent) -> JSONValue? {
        switch part {
        case .text(let text):
            guard !text.text.isEmpty else { return nil }
            return .object(["type": "text", "text": .string(text.text)])

        case .reasoning(let reasoning):
            // A thinking block is only accepted back if its signature travels with it.
            guard let signature = reasoning.signature else { return nil }
            return .object([
                "type": "thinking",
                "thinking": .string(reasoning.text),
                "signature": .string(signature),
            ])

        case .toolCall(let call):
            return .object([
                "type": "tool_use",
                "id": .string(call.toolCallID),
                "name": .string(call.toolName),
                "input": call.input,
            ])

        case .file, .source, .toolResult:
            return nil
        }
    }

    private func toolResultBlock(_ result: ToolResultPart) -> JSONValue {
        var block: [String: JSONValue] = [
            "type": "tool_result",
            "tool_use_id": .string(result.toolCallID),
        ]
        switch result.output {
        case .text(let text), .errorText(let text):
            block["content"] = .string(text)
        case .json(let value), .errorJSON(let value):
            block["content"] = .string(value.serialized())
        case .content(let parts):
            block["content"] = .array(
                parts.compactMap { part in
                    guard case .text(let text) = part else { return nil }
                    return JSONValue.object(["type": "text", "text": .string(text.text)])
                }
            )
        }
        if result.output.isError { block["is_error"] = .bool(true) }
        return .object(block)
    }

    /// Merges runs of same-role messages, which the API rejects.
    private func mergeAdjacentRoles(_ messages: [JSONValue]) -> [JSONValue] {
        var merged: [JSONValue] = []
        for message in messages {
            guard
                let role = message["role"]?.stringValue,
                let content = message["content"]?.arrayValue,
                let previous = merged.last,
                previous["role"]?.stringValue == role,
                let previousContent = previous["content"]?.arrayValue
            else {
                merged.append(message)
                continue
            }
            merged[merged.count - 1] = .object([
                "role": .string(role),
                "content": .array(previousContent + content),
            ])
        }
        return merged
    }

    // MARK: - Cache control

    /// The `cache_control` value from provider options, if any.
    ///
    /// Anthropic charges less for a cached prefix, but only where an explicit breakpoint is
    /// placed. Callers set one with
    /// `providerOptions: ["anthropic": ["cacheControl": ["type": "ephemeral"]]]` on the message or
    /// part that ends the reusable prefix.
    private func cacheControl(_ options: ProviderOptions?) -> JSONValue? {
        options?.value("cacheControl", for: "anthropic")
    }

    private func applyCacheControl(_ options: ProviderOptions?, toLastOf blocks: inout [JSONValue]) {
        guard let cacheControl = cacheControl(options),
              let last = blocks.last,
              case .object(var fields) = last
        else { return }
        fields["cache_control"] = cacheControl
        blocks[blocks.count - 1] = .object(fields)
    }

    // MARK: - Tools

    private func convertTools(
        _ tools: [LanguageModelTool],
        warnings: inout [CallWarning]
    ) -> [JSONValue] {
        tools.compactMap { tool in
            switch tool {
            case .function(let function):
                return .object([
                    "name": .string(function.name),
                    "description": .string(function.description),
                    "input_schema": function.inputSchema.jsonValue(),
                ])

            case .providerDefined(let definition):
                guard definition.providerNamespace == "anthropic" else {
                    warnings.append(
                        .unsupportedTool(
                            toolName: definition.name,
                            details: "'\(definition.id)' belongs to another provider."
                        )
                    )
                    return nil
                }
                // A provider-defined tool is passed through as the API documents it.
                var fields: [String: JSONValue] = ["name": .string(definition.name)]
                for (key, value) in definition.arguments { fields[key] = value }
                return .object(fields)
            }
        }
    }

    private func toolChoice(_ choice: ToolChoice?) -> JSONValue? {
        switch choice {
        case nil, .auto: return .object(["type": "auto"])
        case .never: return .object(["type": "none"])
        case .required: return .object(["type": "any"])
        case .tool(let name): return .object(["type": "tool", "name": .string(name)])
        }
    }
}

/// Constants for the synthetic tool used to constrain output to a schema.
enum AnthropicStructuredOutput {
    /// The name of the tool the model is forced to call.
    ///
    /// Chosen to be descriptive to the model, since it appears in the prompt.
    static let toolName = "respond_with_structured_output"
}

import AIProviderSpec
import AIProviderUtils
import Foundation

/// Builds a request for OpenAI's Responses API.
///
/// The Responses API replaces Chat Completions' `messages` array with a flat `input` list in which
/// assistant turns, tool calls, tool outputs, and reasoning are each their own item rather than
/// fields on a message. That flattening is most of what this builder does. The rest is the two
/// features Chat Completions lacks: reasoning items that must be replayed to preserve a model's
/// chain of thought across turns, and provider-executed tools.
struct ResponsesRequestBuilder {
    let modelID: String

    struct Built {
        var body: JSONValue
        var warnings: [CallWarning]
    }

    func build(_ options: LanguageModelCallOptions, stream: Bool) throws -> Built {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = ["model": .string(modelID)]

        let converted = try convertPrompt(options.prompt, warnings: &warnings)
        if let instructions = converted.instructions { fields["instructions"] = .string(instructions) }
        fields["input"] = .array(converted.input)

        if let value = options.maxOutputTokens { fields["max_output_tokens"] = .int(value) }
        if let value = options.temperature { fields["temperature"] = .double(value) }
        if let value = options.topP { fields["top_p"] = .double(value) }
        if stream { fields["stream"] = .bool(true) }

        for (name, isSet) in [
            ("topK", options.topK != nil),
            ("presencePenalty", options.presencePenalty != nil),
            ("frequencyPenalty", options.frequencyPenalty != nil),
            ("seed", options.seed != nil),
            ("stopSequences", !options.stopSequences.isEmpty),
        ] where isSet {
            warnings.append(
                .unsupportedSetting(
                    setting: name,
                    details: "The Responses API has no equivalent. Use the chat model for it."
                )
            )
        }

        if case .json(let schema, let name, let description) = options.responseFormat {
            var format: [String: JSONValue] = ["type": "json_schema", "name": .string(name ?? "response")]
            if let schema {
                format["schema"] = schema.strictified().jsonValue(options: .openAIStrict)
                format["strict"] = .bool(true)
            } else {
                format = ["type": "json_object"]
            }
            if let description { format["description"] = .string(description) }
            fields["text"] = .object(["format": .object(format)])
        }

        let tools = convertTools(options.tools, warnings: &warnings)
        if !tools.isEmpty {
            fields["tools"] = .array(tools)
            if let choice = toolChoice(options.toolChoice) { fields["tool_choice"] = choice }
        }

        for (key, value) in options.providerOptions?["openai"] ?? [:] {
            fields[key] = value
        }

        return Built(body: .object(fields), warnings: warnings)
    }

    // MARK: - Prompt

    private struct ConvertedPrompt {
        var instructions: String?
        var input: [JSONValue]
    }

    private func convertPrompt(
        _ messages: [ModelMessage],
        warnings: inout [CallWarning]
    ) throws -> ConvertedPrompt {
        var instructions: String?
        var input: [JSONValue] = []

        for message in messages {
            switch message {
            case .system(let system):
                instructions = system.content

            case .user(let user):
                let content = user.content.map { userContentPart($0, warnings: &warnings) }
                input.append(.object(["role": "user", "content": .array(content)]))

            case .assistant(let assistant):
                input.append(contentsOf: assistantItems(assistant.content))

            case .tool(let toolMessage):
                for result in toolMessage.content {
                    input.append(
                        .object([
                            "type": "function_call_output",
                            "call_id": .string(result.toolCallID),
                            "output": .string(toolOutputText(result.output)),
                        ])
                    )
                }
            }
        }
        return ConvertedPrompt(instructions: instructions, input: input)
    }

    private func userContentPart(_ part: UserContent, warnings: inout [CallWarning]) -> JSONValue {
        switch part {
        case .text(let text):
            return .object(["type": "input_text", "text": .string(text.text)])

        case .file(let file):
            let reference: String
            switch file.source {
            case .url(let url): reference = url.absoluteString
            case .data: reference = file.dataURI ?? ""
            }

            if file.isImage {
                return .object(["type": "input_image", "image_url": .string(reference)])
            }
            return .object([
                "type": "input_file",
                "filename": .string(file.filename ?? "document"),
                "file_data": .string(reference),
            ])
        }
    }

    /// Flattens an assistant turn into the separate items the API expects.
    ///
    /// Reasoning is replayed with its provider-issued identifier and encrypted payload. Dropping
    /// those would make the model start its reasoning over on every turn, which costs both tokens
    /// and coherence.
    private func assistantItems(_ content: [ModelContent]) -> [JSONValue] {
        var items: [JSONValue] = []
        var messageContent: [JSONValue] = []

        for part in content {
            switch part {
            case .text(let text):
                guard !text.text.isEmpty else { continue }
                messageContent.append(.object(["type": "output_text", "text": .string(text.text)]))

            case .reasoning(let reasoning):
                guard let id = reasoning.providerOptions?.value("itemId", for: "openai")?.stringValue
                else { continue }
                var item: [String: JSONValue] = ["type": "reasoning", "id": .string(id)]
                if let encrypted = reasoning.providerOptions?
                    .value("encryptedContent", for: "openai")?.stringValue {
                    item["encrypted_content"] = .string(encrypted)
                }
                if !reasoning.text.isEmpty {
                    item["summary"] = .array([
                        .object(["type": "summary_text", "text": .string(reasoning.text)])
                    ])
                }
                items.append(.object(item))

            case .toolCall(let call):
                items.append(
                    .object([
                        "type": "function_call",
                        "call_id": .string(call.toolCallID),
                        "name": .string(call.toolName),
                        "arguments": .string(call.input.serialized()),
                    ])
                )

            case .file, .source, .toolResult:
                continue
            }
        }

        if !messageContent.isEmpty {
            items.append(.object(["role": "assistant", "content": .array(messageContent)]))
        }
        return items
    }

    private func toolOutputText(_ output: ToolResultOutput) -> String {
        switch output {
        case .text(let text), .errorText(let text): return text
        case .json(let value), .errorJSON(let value): return value.serialized()
        case .content(let parts):
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
    ) -> [JSONValue] {
        tools.compactMap { tool in
            switch tool {
            case .function(let function):
                return .object([
                    "type": "function",
                    "name": .string(function.name),
                    "description": .string(function.description),
                    "parameters": function.inputSchema.strictified().jsonValue(options: .openAIStrict),
                    "strict": .bool(true),
                ])

            case .providerDefined(let definition):
                guard definition.providerNamespace == "openai" else {
                    warnings.append(
                        .unsupportedTool(
                            toolName: definition.name,
                            details: "'\(definition.id)' belongs to another provider."
                        )
                    )
                    return nil
                }
                // Provider-executed tools are declared by type and configured verbatim.
                var fields: [String: JSONValue] = ["type": .string(definition.name)]
                for (key, value) in definition.arguments { fields[key] = value }
                return .object(fields)
            }
        }
    }

    private func toolChoice(_ choice: ToolChoice?) -> JSONValue? {
        switch choice {
        case nil, .auto: return .string("auto")
        case .never: return .string("none")
        case .required: return .string("required")
        case .tool(let name): return .object(["type": "function", "name": .string(name)])
        }
    }
}

import AIProviderSpec
import AIProviderUtils
import Foundation

/// Builds a `generateContent` request body.
///
/// Gemini's shape differs from the others in several ways that the conversion has to absorb:
///
/// - Messages are `contents`, the assistant role is called `model`, and system instructions are a
///   separate top-level field.
/// - Sampling settings live inside `generationConfig` rather than at the top level.
/// - Tool results are `functionResponse` parts inside a user turn, and the response must be a
///   JSON object even when the tool returned a bare string.
/// - Schemas follow an OpenAPI 3.0 subset, so several standard JSON Schema keywords have to be
///   removed rather than passed through.
struct GoogleRequestBuilder {
    let modelID: String

    struct Built {
        var body: JSONValue
        var warnings: [CallWarning]
    }

    func build(_ options: LanguageModelCallOptions) throws -> Built {
        var warnings: [CallWarning] = []
        var fields: [String: JSONValue] = [:]

        let converted = convertMessages(options.prompt, warnings: &warnings)
        if let system = converted.systemInstruction { fields["systemInstruction"] = system }
        fields["contents"] = .array(converted.contents)

        var generationConfig: [String: JSONValue] = [:]
        if let value = options.maxOutputTokens { generationConfig["maxOutputTokens"] = .int(value) }
        if let value = options.temperature { generationConfig["temperature"] = .double(value) }
        if let value = options.topP { generationConfig["topP"] = .double(value) }
        if let value = options.topK { generationConfig["topK"] = .int(value) }
        if let value = options.presencePenalty { generationConfig["presencePenalty"] = .double(value) }
        if let value = options.frequencyPenalty { generationConfig["frequencyPenalty"] = .double(value) }
        if let value = options.seed { generationConfig["seed"] = .int(value) }
        if !options.stopSequences.isEmpty {
            generationConfig["stopSequences"] = .array(options.stopSequences.map(JSONValue.string))
        }

        if case .json(let schema, _, _) = options.responseFormat {
            generationConfig["responseMimeType"] = .string("application/json")
            if let schema {
                generationConfig["responseSchema"] = googleSchema(schema, warnings: &warnings)
            }
        }

        // Provider options can add or override anything inside `generationConfig` — thinking
        // budgets and response modalities both live there.
        let providerOptions = options.providerOptions?["google"] ?? [:]
        if case .object(let overrides)? = providerOptions["generationConfig"] {
            for (key, value) in overrides { generationConfig[key] = value }
        }
        if !generationConfig.isEmpty { fields["generationConfig"] = .object(generationConfig) }

        let declarations = convertTools(options.tools, warnings: &warnings)
        if !declarations.isEmpty {
            fields["tools"] = .array([.object(["functionDeclarations": .array(declarations)])])
            if let config = toolConfig(options.toolChoice) { fields["toolConfig"] = config }
        }

        for (key, value) in providerOptions where key != "generationConfig" {
            fields[key] = value
        }

        return Built(body: .object(fields), warnings: warnings)
    }

    // MARK: - Messages

    private struct ConvertedContents {
        var systemInstruction: JSONValue?
        var contents: [JSONValue]
    }

    private func convertMessages(
        _ messages: [ModelMessage],
        warnings: inout [CallWarning]
    ) -> ConvertedContents {
        var systemInstruction: JSONValue?
        var contents: [JSONValue] = []

        for message in messages {
            switch message {
            case .system(let system):
                systemInstruction = .object(["parts": .array([.object(["text": .string(system.content)])])])

            case .user(let user):
                let parts = user.content.map { userPart($0, warnings: &warnings) }
                contents.append(.object(["role": "user", "parts": .array(parts)]))

            case .assistant(let assistant):
                let parts = assistant.content.compactMap(modelPart)
                guard !parts.isEmpty else { continue }
                contents.append(.object(["role": "model", "parts": .array(parts)]))

            case .tool(let toolMessage):
                let parts = toolMessage.content.map(functionResponsePart)
                contents.append(.object(["role": "user", "parts": .array(parts)]))
            }
        }
        return ConvertedContents(systemInstruction: systemInstruction, contents: contents)
    }

    private func userPart(_ part: UserContent, warnings: inout [CallWarning]) -> JSONValue {
        switch part {
        case .text(let text):
            return .object(["text": .string(text.text)])

        case .file(let file):
            switch file.source {
            case .data(let data):
                return .object([
                    "inlineData": .object([
                        "mimeType": .string(file.mediaType),
                        "data": .string(data.base64EncodedString()),
                    ])
                ])
            case .url(let url):
                // Only files already uploaded to the Files API can be referenced by URI; anything
                // else has to arrive inline, which the core arranges by downloading it first.
                return .object([
                    "fileData": .object([
                        "mimeType": .string(file.mediaType),
                        "fileUri": .string(url.absoluteString),
                    ])
                ])
            }
        }
    }

    private func modelPart(_ part: ModelContent) -> JSONValue? {
        switch part {
        case .text(let text):
            guard !text.text.isEmpty else { return nil }
            return .object(["text": .string(text.text)])

        case .reasoning(let reasoning):
            guard !reasoning.text.isEmpty else { return nil }
            return .object(["text": .string(reasoning.text), "thought": .bool(true)])

        case .toolCall(let call):
            return .object([
                "functionCall": .object([
                    "name": .string(call.toolName),
                    "args": call.input,
                ])
            ])

        case .file, .source, .toolResult:
            return nil
        }
    }

    private func functionResponsePart(_ result: ToolResultPart) -> JSONValue {
        // The API requires an object here, so a scalar result is wrapped.
        let response: JSONValue
        switch result.output {
        case .json(let value), .errorJSON(let value):
            response = value.objectValue == nil ? .object(["output": value]) : value
        case .text(let text), .errorText(let text):
            response = .object(["output": .string(text)])
        case .content(let parts):
            let text = parts.compactMap { part -> String? in
                guard case .text(let text) = part else { return nil }
                return text.text
            }.joined(separator: "\n")
            response = .object(["output": .string(text)])
        }

        return .object([
            "functionResponse": .object([
                "name": .string(result.toolName),
                "response": response,
            ])
        ])
    }

    // MARK: - Tools and schemas

    private func convertTools(
        _ tools: [LanguageModelTool],
        warnings: inout [CallWarning]
    ) -> [JSONValue] {
        tools.compactMap { tool in
            switch tool {
            case .function(let function):
                var declaration: [String: JSONValue] = [
                    "name": .string(function.name),
                    "description": .string(function.description),
                ]
                let parameters = googleSchema(function.inputSchema, warnings: &warnings)
                // An empty parameter object is rejected; omitting the field is the correct way to
                // declare a tool that takes no arguments.
                if parameters["properties"]?.objectValue?.isEmpty == false {
                    declaration["parameters"] = parameters
                }
                return .object(declaration)

            case .providerDefined(let definition):
                warnings.append(
                    .unsupportedTool(
                        toolName: definition.name,
                        details: "Provider-defined tools are configured through provider options for Gemini."
                    )
                )
                return nil
            }
        }
    }

    private func toolConfig(_ choice: ToolChoice?) -> JSONValue? {
        switch choice {
        case nil, .auto:
            return .object(["functionCallingConfig": .object(["mode": "AUTO"])])
        case .never:
            return .object(["functionCallingConfig": .object(["mode": "NONE"])])
        case .required:
            return .object(["functionCallingConfig": .object(["mode": "ANY"])])
        case .tool(let name):
            return .object([
                "functionCallingConfig": .object([
                    "mode": "ANY",
                    "allowedFunctionNames": .array([.string(name)]),
                ])
            ])
        }
    }

    /// Renders a schema in Gemini's OpenAPI 3.0 subset.
    ///
    /// Keywords the API rejects are removed rather than passed through, and a warning is reported
    /// when something meaningful is lost — silently dropping a constraint the caller wrote would
    /// be worse than telling them it did not survive.
    private func googleSchema(_ schema: JSONSchema, warnings: inout [CallWarning]) -> JSONValue {
        var lostKeywords: Set<String> = []
        let simplified = schema.transformed { node in
            // `anyOf` has no equivalent in the subset; the first branch is the best approximation.
            if case .anyOf(let constraints) = node, let first = constraints.subschemas.first {
                lostKeywords.insert("anyOf")
                return first
            }
            if case .oneOf(let constraints) = node, let first = constraints.subschemas.first {
                lostKeywords.insert("oneOf")
                return first
            }
            return node
        }

        if !lostKeywords.isEmpty {
            warnings.append(
                .other(
                    message: """
                        Gemini's schema dialect does not support \
                        \(lostKeywords.sorted().joined(separator: ", ")); the first branch was used.
                        """
                )
            )
        }
        return simplified.jsonValue(options: .googleGenerativeAI)
    }
}

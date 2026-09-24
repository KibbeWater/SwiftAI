import AIProviderSpec
import AIProviderUtils
import AITestSupport
import Foundation
import Testing

@testable import AIGoogle

@Suite("Google Gemini")
struct GoogleTests {
    private func fixture(_ name: String) throws -> String {
        try Fixtures.text(name, bundle: Bundle.module)
    }

    private func model(_ transport: MockHTTPTransport) -> any LanguageModel {
        GoogleProvider(apiKey: "test-key", transport: transport).languageModel("gemini-2.5-flash")
    }

    private func options(
        _ prompt: [ModelMessage] = [.user("Why is the sky blue?")],
        tools: [LanguageModelTool] = [],
        toolChoice: ToolChoice? = nil,
        responseFormat: ResponseFormat? = nil,
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        providerOptions: ProviderOptions? = nil
    ) -> LanguageModelCallOptions {
        LanguageModelCallOptions(
            prompt: prompt,
            maxOutputTokens: maxOutputTokens,
            temperature: temperature,
            responseFormat: responseFormat,
            tools: tools,
            toolChoice: toolChoice,
            providerOptions: providerOptions
        )
    }

    // MARK: - Buffered responses

    @Test("Decodes a response")
    func decodesResponse() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let response = try await model(transport).generate(options())

        #expect(response.content.text == "Blue light scatters more in the atmosphere.")
        #expect(response.finishReason == .stop)
        #expect(response.response?.id == "rIdY6Ku3Ic-K")
        #expect(response.response?.modelID == "gemini-2.5-flash")
    }

    @Test("Includes reasoning tokens in the output count")
    func includesReasoningTokensInOutput() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let response = try await model(transport).generate(options())

        // Gemini reports thoughts separately and excludes them from `candidatesTokenCount`;
        // reporting only the visible tokens would understate what a thinking model costs.
        #expect(response.usage.inputTokens == 8)
        #expect(response.usage.outputTokens == 33)
        #expect(response.usage.reasoningTokens == 24)
        #expect(response.usage.cachedInputTokens == 4)
    }

    @Test("Reports a tool call even though the wire says STOP")
    func reportsToolCallDespiteStopReason() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content-tools.json")))
        let response = try await model(transport).generate(options())

        let call = try #require(response.content.toolCalls.first)
        #expect(call.toolName == "weather")
        #expect(call.input == ["city": "Malmö"])
        // Taking the wire's "STOP" at face value would stall the tool loop.
        #expect(response.finishReason == .toolCalls)
        // Gemini issues no call identifiers, so one is generated to correlate the result.
        #expect(!call.toolCallID.isEmpty)
    }

    @Test("Maps finish reasons", arguments: [
        ("STOP", FinishReason.stop),
        ("MAX_TOKENS", .length),
        ("SAFETY", .contentFilter),
        ("RECITATION", .contentFilter),
        ("MALFORMED_FUNCTION_CALL", .error),
    ])
    func mapsFinishReasons(raw: String, expected: FinishReason) {
        #expect(FinishReason.fromGoogle(raw, hasFunctionCall: false) == expected)
    }

    @Test("Maps an error response")
    func mapsErrorResponse() async throws {
        let transport = MockHTTPTransport(
            exchange: .failure(statusCode: 400, body: try fixture("error-400.json"))
        )

        var caught: APICallError?
        do {
            _ = try await model(transport).generate(options())
        } catch let error as APICallError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.message.contains("Invalid JSON payload"))
        #expect(error.isRetryable == false)
    }

    // MARK: - Request construction

    @Test("Sends the key as a header and the model in the path")
    func sendsKeyAndPath() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        _ = try await model(transport).generate(options())

        let request = try #require(transport.recordedRequests.first)
        // A header keeps the key out of URLs, and therefore out of logs.
        #expect(request.headers["x-goog-api-key"] == "test-key")
        #expect(request.url.absoluteString.hasSuffix("/models/gemini-2.5-flash:generateContent"))
        #expect(request.url.query == nil)
    }

    @Test("Accepts a model identifier that is already qualified")
    func acceptsQualifiedModelID() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let model = GoogleProvider(apiKey: "k", transport: transport).languageModel("models/gemini-2.5-pro")
        _ = try await model.generate(options())

        let url = try #require(transport.recordedRequests.first?.url.absoluteString)
        #expect(url.hasSuffix("/models/gemini-2.5-pro:generateContent"))
        #expect(!url.contains("models/models/"))
    }

    @Test("Sends system instructions in their own field")
    func sendsSystemInstruction() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        _ = try await model(transport).generate(options([.system("Be terse."), .user("Hi")]))

        let body = try transport.recordedRequestBody()
        #expect(body["systemInstruction"]?["parts"]?[0]?["text"]?.stringValue == "Be terse.")
        #expect(body["contents"]?.arrayValue?.count == 1)
    }

    @Test("Uses the model role for assistant turns")
    func usesModelRole() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        _ = try await model(transport).generate(options([.user("Hi"), .assistant("Hello"), .user("Again")]))

        let roles = try #require(try transport.recordedRequestBody()["contents"]?.arrayValue)
            .map { $0["role"]?.stringValue }
        #expect(roles == ["user", "model", "user"])
    }

    @Test("Puts sampling settings inside generationConfig")
    func nestsGenerationConfig() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        _ = try await model(transport).generate(options(maxOutputTokens: 500, temperature: 0.4))

        let config = try #require(try transport.recordedRequestBody()["generationConfig"])
        #expect(config["maxOutputTokens"]?.intValue == 500)
        #expect(config["temperature"]?.numberValue == 0.4)
    }

    @Test("Wraps a scalar tool result in an object")
    func wrapsScalarToolResult() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "c1", toolName: "weather", input: [:]))]),
            .tool([ToolResultPart(toolCallID: "c1", toolName: "weather", output: .text("Rain."))]),
        ]
        _ = try await model(transport).generate(options(prompt))

        // The API requires an object for `response`, so a bare string has to be wrapped.
        let part = try #require(try transport.recordedRequestBody()["contents"]?[2]?["parts"]?[0])
        #expect(part["functionResponse"]?["name"]?.stringValue == "weather")
        #expect(part["functionResponse"]?["response"]?["output"]?.stringValue == "Rain.")
    }

    @Test("Passes an object tool result through unchanged")
    func passesObjectToolResultThrough() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "c1", toolName: "weather", input: [:]))]),
            .tool([ToolResultPart(toolCallID: "c1", toolName: "weather", output: .json(["temp": 14]))]),
        ]
        _ = try await model(transport).generate(options(prompt))

        let response = try #require(
            try transport.recordedRequestBody()["contents"]?[2]?["parts"]?[0]?["functionResponse"]?["response"]
        )
        #expect(response["temp"]?.intValue == 14)
    }

    @Test("Sends files as inline data")
    func sendsInlineData() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let prompt: [ModelMessage] = [
            .user([.text("What is this?"), .file(.data(Data([0x89, 0x50]), mediaType: "image/png"))])
        ]
        _ = try await model(transport).generate(options(prompt))

        let part = try #require(try transport.recordedRequestBody()["contents"]?[0]?["parts"]?[1])
        #expect(part["inlineData"]?["mimeType"]?.stringValue == "image/png")
        #expect(part["inlineData"]?["data"]?.stringValue == Data([0x89, 0x50]).base64EncodedString())
    }

    @Test("Renders schemas in Gemini's dialect")
    func rendersSchemasInGoogleDialect() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let schema = JSONSchema.object(
            properties: ["name": .string(minLength: 1), "age": .nullable(.integer())],
            required: ["name"]
        )
        _ = try await model(transport).generate(
            options(responseFormat: .json(schema: schema, name: nil, description: nil))
        )

        let config = try #require(try transport.recordedRequestBody()["generationConfig"])
        #expect(config["responseMimeType"]?.stringValue == "application/json")

        let responseSchema = try #require(config["responseSchema"])
        // Keywords Gemini rejects must not appear.
        #expect(responseSchema["additionalProperties"] == nil)
        #expect(responseSchema["properties"]?["name"]?["minLength"] == nil)
        // Optionality is expressed with the OpenAPI flag rather than a type array.
        #expect(responseSchema["properties"]?["age"]?["nullable"]?.boolValue == true)
    }

    @Test("Omits parameters for a tool that takes none")
    func omitsEmptyToolParameters() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(name: "now", description: "The current time.", inputSchema: .object(properties: [:], required: []))
        )
        _ = try await model(transport).generate(options(tools: [tool]))

        // An empty parameter object is rejected by the API.
        let declaration = try #require(try transport.recordedRequestBody()["tools"]?[0]?["functionDeclarations"]?[0])
        #expect(declaration["parameters"] == nil)
        #expect(declaration["name"]?.stringValue == "now")
    }

    @Test("Maps tool choices", arguments: [
        (ToolChoice.auto, "AUTO"),
        (.never, "NONE"),
        (.required, "ANY"),
    ])
    func mapsToolChoice(choice: ToolChoice, expected: String) async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(name: "t", description: "d", inputSchema: .object(properties: ["a": .string()], required: []))
        )
        _ = try await model(transport).generate(options(tools: [tool], toolChoice: choice))

        let mode = try #require(
            try transport.recordedRequestBody()["toolConfig"]?["functionCallingConfig"]?["mode"]?.stringValue
        )
        #expect(mode == expected)
    }

    @Test("Restricts to a named tool")
    func restrictsToNamedTool() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(name: "weather", description: "d", inputSchema: .object(properties: ["a": .string()], required: []))
        )
        _ = try await model(transport).generate(options(tools: [tool], toolChoice: .tool(named: "weather")))

        let config = try #require(try transport.recordedRequestBody()["toolConfig"]?["functionCallingConfig"])
        #expect(config["mode"]?.stringValue == "ANY")
        #expect(config["allowedFunctionNames"] == .array(["weather"]))
    }

    @Test("Merges provider options into generationConfig")
    func mergesGenerationConfigOptions() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let options = options(
            temperature: 0.5,
            providerOptions: [
                "google": ["generationConfig": ["thinkingConfig": ["thinkingBudget": 2048]]]
            ]
        )
        _ = try await model(transport).generate(options)

        let config = try #require(try transport.recordedRequestBody()["generationConfig"])
        #expect(config["thinkingConfig"]?["thinkingBudget"]?.intValue == 2048)
        // A setting the caller passed normally still survives the merge.
        #expect(config["temperature"]?.numberValue == 0.5)
    }

    @Test("Puts other provider options at the top level")
    func putsOtherProviderOptionsAtTopLevel() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("generate-content.json")))
        let options = options(
            providerOptions: [
                "google": [
                    "safetySettings": .array([
                        .object(["category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"])
                    ])
                ]
            ]
        )
        _ = try await model(transport).generate(options)

        let settings = try #require(try transport.recordedRequestBody()["safetySettings"]?[0])
        #expect(settings["threshold"]?.stringValue == "BLOCK_ONLY_HIGH")
    }

    // MARK: - Streaming

    @Test("Streams text", arguments: [1, 13, 4096])
    func streamsText(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("stream.sse"), chunkSize: chunkSize)
        )
        let parts = try await model(transport).stream(options()).stream.collect()

        #expect(parts.compactMap(\.textDelta).joined() == "Blue light scatters.")
        // The block structure is synthesized: the wire has no start or end events.
        #expect(parts.filter { $0.kind == .textStart }.count == 1)
        #expect(parts.filter { $0.kind == .textEnd }.count == 1)

        guard case .finish(let reason, let usage, _)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .stop)
        #expect(usage.inputTokens == 8)
    }

    @Test("Requests the server-sent event encoding when streaming")
    func requestsSSEEncoding() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("stream.sse"), chunkSize: 512)
        )
        _ = try await model(transport).stream(options())

        let url = try #require(transport.recordedRequests.first?.url.absoluteString)
        #expect(url.contains(":streamGenerateContent"))
        #expect(url.contains("alt=sse"))
    }

    @Test("Separates thought parts from the answer")
    func separatesThoughtParts() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("stream-thinking.sse"), chunkSize: 17)
        )
        let parts = try await model(transport).stream(options()).stream.collect()
        let kinds = parts.map(\.kind)

        #expect(parts.compactMap(\.reasoningDelta).joined() == "Shorter wavelengths scatter more.")
        #expect(parts.compactMap(\.textDelta).joined() == "Rayleigh scattering.")

        let reasoningEnd = try #require(kinds.firstIndex(of: .reasoningEnd))
        let textStart = try #require(kinds.firstIndex(of: .textStart))
        #expect(reasoningEnd < textStart)
    }

    @Test("Synthesizes a complete block around a function call", arguments: [1, 19, 2048])
    func synthesizesToolCallBlock(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("stream-tools.sse"), chunkSize: chunkSize)
        )
        let parts = try await model(transport).stream(options()).stream.collect()

        #expect(parts.compactMap(\.textDelta).joined() == "Checking.")

        // Gemini delivers arguments whole; the triple is emitted anyway so consumers see one
        // consistent shape across providers.
        #expect(parts.contains { $0.kind == .toolInputStart })
        #expect(parts.contains { $0.kind == .toolInputDelta })
        #expect(parts.contains { $0.kind == .toolInputEnd })

        guard let call = parts.compactMap({ part -> ToolCallPart? in
            guard case .toolCall(let call) = part else { return nil }
            return call
        }).first else {
            Issue.record("Expected a parsed tool call.")
            return
        }
        #expect(call.toolName == "weather")
        #expect(call.input == ["city": "Malmö"])

        guard case .finish(let reason, _, _)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .toolCalls)
    }
}

@Suite("Google evaluation")
struct GoogleEvaluationTests {
    @Test("Turns thinking down as far as each model allows", arguments: [
        ("gemini-3.5-flash-lite", ["thinkingLevel": "minimal"]),
        ("gemini-3-pro-preview", ["thinkingLevel": "minimal"]),
        // Newer non-lite Flash models no longer accept `minimal`.
        ("gemini-3.7-flash", ["thinkingLevel": "low"]),
        ("gemini-flash-latest", ["thinkingLevel": "low"]),
        ("gemini-2.5-flash", ["thinkingBudget": 0]),
        // 2.5 Pro cannot turn thinking off; 128 is its floor.
        ("models/gemini-2.5-pro", ["thinkingBudget": 128]),
    ] as [(String, JSONValue)])
    func minimalThinking(modelID: String, config: JSONValue) {
        #expect(GoogleProvider.minimalThinking(for: modelID) == config)
    }

    @Test("Leaves models without thinking alone")
    func noThinkingForOlderModels() {
        #expect(GoogleProvider.minimalThinking(for: "gemini-2.0-flash") == nil)
    }
}

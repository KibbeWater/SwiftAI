import AIProviderSpec
import AIProviderUtils
import AITestSupport
import Foundation
import Testing

@testable import AIOpenAICompatible

/// Tests driven by payloads recorded from real servers.
///
/// The fixtures are checked in verbatim, so the parsing code meets exactly the bytes a provider
/// sends — including the fields this SDK ignores, which is where changes in a provider's output
/// tend to break things first.
@Suite("Chat Completions")
struct ChatCompletionsTests {
    private func fixture(_ name: String) throws -> String {
        try Fixtures.text(name, bundle: Bundle.module)
    }

    private func provider(
        _ transport: MockHTTPTransport,
        quirks: OpenAICompatibleQuirks = .default
    ) -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            name: "compatible",
            baseURL: URL(string: "https://api.example.com/v1")!,
            apiKey: "test-key",
            transport: transport,
            quirks: quirks
        )
    }

    private func options(
        _ prompt: [ModelMessage] = [.user("Why is the sky blue?")],
        tools: [LanguageModelTool] = [],
        responseFormat: ResponseFormat? = nil,
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        topK: Int? = nil,
        toolChoice: ToolChoice? = nil,
        providerOptions: ProviderOptions? = nil
    ) -> LanguageModelCallOptions {
        LanguageModelCallOptions(
            prompt: prompt,
            maxOutputTokens: maxOutputTokens,
            temperature: temperature,
            topK: topK,
            responseFormat: responseFormat,
            tools: tools,
            toolChoice: toolChoice,
            providerOptions: providerOptions
        )
    }

    // MARK: - Buffered responses

    @Test("Decodes a completion")
    func decodesCompletion() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let model = provider(transport).languageModel("gpt-4o-mini")

        let response = try await model.generate(options())

        #expect(response.content.text.hasPrefix("The sky looks blue"))
        #expect(response.finishReason == .stop)
        #expect(response.usage.inputTokens == 24)
        #expect(response.usage.outputTokens == 17)
        #expect(response.usage.cachedInputTokens == 8)
        #expect(response.response?.id == "chatcmpl-BxYz1a2b3c")
        #expect(response.response?.modelID == "gpt-4o-mini-2024-07-18")
    }

    @Test("Decodes a tool call")
    func decodesToolCall() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-tool-call.json")))
        let model = provider(transport).languageModel("gpt-4o-mini")

        let response = try await model.generate(options())
        let call = try #require(response.content.toolCalls.first)

        #expect(call.toolCallID == "call_9xTqPl2")
        #expect(call.toolName == "weather")
        #expect(call.input == ["city": "Malmö"])
        #expect(response.finishReason == .toolCalls)
    }

    @Test("Puts reasoning before the answer")
    func decodesReasoning() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-reasoning.json")))
        let model = provider(transport).languageModel("deepseek-reasoner")

        let response = try await model.generate(options())

        // The wire carries both as sibling fields; reasoning precedes the answer chronologically.
        #expect(response.content.first?.kindName == "reasoning")
        #expect(response.content.reasoningText == "Rayleigh scattering favours shorter wavelengths.")
        #expect(response.content.text == "Because blue light scatters more.")
        #expect(response.usage.reasoningTokens == 21)
    }

    @Test("Turns an error response into a readable failure")
    func mapsErrorResponse() async throws {
        let transport = MockHTTPTransport(
            exchange: .failure(statusCode: 401, body: try fixture("error-401.json"))
        )
        let model = provider(transport).languageModel("gpt-4o-mini")

        var caught: APICallError?
        do {
            _ = try await model.generate(options())
        } catch let error as APICallError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.statusCode == 401)
        #expect(error.message.contains("Incorrect API key"))
        #expect(error.isRetryable == false)
        #expect(error.data?["error"]?["code"]?.stringValue == "invalid_api_key")
    }

    // MARK: - Request construction

    @Test("Sends credentials and the model identifier")
    func sendsCredentialsAndModel() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        _ = try await provider(transport).languageModel("gpt-4o-mini").generate(options())

        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["Authorization"] == "Bearer test-key")
        #expect(request.url.absoluteString == "https://api.example.com/v1/chat/completions")
        #expect(try transport.recordedRequestBody()["model"]?.stringValue == "gpt-4o-mini")
    }

    @Test("Sends plain text content as a string")
    func sendsPlainTextAsString() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        _ = try await provider(transport).languageModel("m").generate(options())

        // Several compatible servers only accept the string form.
        let body = try transport.recordedRequestBody()
        #expect(body["messages"]?[0]?["content"]?.stringValue == "Why is the sky blue?")
    }

    @Test("Sends mixed content as an array of parts")
    func sendsMixedContentAsArray() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let prompt: [ModelMessage] = [
            .user([.text("What is this?"), .file(.data(Data([0x89, 0x50]), mediaType: "image/png"))])
        ]
        _ = try await provider(transport).languageModel("m").generate(options(prompt))

        let parts = try #require(try transport.recordedRequestBody()["messages"]?[0]?["content"]?.arrayValue)
        #expect(parts[0]["type"]?.stringValue == "text")
        #expect(parts[1]["type"]?.stringValue == "image_url")
        #expect(parts[1]["image_url"]?["url"]?.stringValue?.hasPrefix("data:image/png;base64,") == true)
    }

    @Test("Sends a system message under the configured role")
    func sendsSystemRole() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        var quirks = OpenAICompatibleQuirks.default
        quirks.systemRole = "developer"

        _ = try await provider(transport, quirks: quirks).languageModel("m")
            .generate(options([.system("Be terse."), .user("Hi")]))

        #expect(try transport.recordedRequestBody()["messages"]?[0]?["role"]?.stringValue == "developer")
    }

    @Test("Sends one tool message per result")
    func sendsOneMessagePerToolResult() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "c1", toolName: "weather", input: ["city": "Malmö"]))]),
            .tool([
                ToolResultPart(toolCallID: "c1", toolName: "weather", output: .text("Rain.")),
                ToolResultPart(toolCallID: "c2", toolName: "weather", output: .json(["temp": 14])),
            ]),
        ]
        _ = try await provider(transport).languageModel("m").generate(options(prompt))

        let messages = try #require(try transport.recordedRequestBody()["messages"]?.arrayValue)
        #expect(messages.count == 4)
        #expect(messages[2]["role"]?.stringValue == "tool")
        #expect(messages[2]["tool_call_id"]?.stringValue == "c1")
        #expect(messages[3]["content"]?.stringValue == #"{"temp":14}"#)
    }

    @Test("An assistant turn with only a tool call sends null content")
    func assistantToolCallSendsNullContent() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "c1", toolName: "weather", input: [:]))]),
        ]
        _ = try await provider(transport).languageModel("m").generate(options(prompt))

        // An empty string here is rejected by several servers; null is the correct spelling.
        #expect(try transport.recordedRequestBody()["messages"]?[1]?["content"]?.isNull == true)
    }

    @Test("Chooses the token limit field the server expects", arguments: [true, false])
    func choosesTokenLimitField(usesMaxCompletionTokens: Bool) async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        var quirks = OpenAICompatibleQuirks.default
        quirks.usesMaxCompletionTokens = usesMaxCompletionTokens

        _ = try await provider(transport, quirks: quirks).languageModel("m")
            .generate(options(maxOutputTokens: 500))

        let body = try transport.recordedRequestBody()
        let expected = usesMaxCompletionTokens ? "max_completion_tokens" : "max_tokens"
        let unexpected = usesMaxCompletionTokens ? "max_tokens" : "max_completion_tokens"
        #expect(body[expected]?.intValue == 500)
        #expect(body[unexpected] == nil)
    }

    @Test("Warns about a setting the API cannot express")
    func warnsAboutUnsupportedSetting() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let response = try await provider(transport).languageModel("m").generate(options(topK: 40))

        // The call still succeeds; the caller is told what was dropped.
        #expect(response.warnings.count == 1)
        #expect(response.warnings.first?.description.contains("topK") == true)
    }

    @Test("Sends tools with strict schemas when the server supports them")
    func sendsStrictSchemas() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-tool-call.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(
                name: "weather",
                description: "Look up the weather.",
                inputSchema: .object(
                    properties: ["city": .string(), "days": .integer()],
                    required: ["city"]
                )
            )
        )

        _ = try await provider(transport, quirks: .openAI).languageModel("m")
            .generate(options(tools: [tool]))

        let function = try #require(try transport.recordedRequestBody()["tools"]?[0]?["function"])
        #expect(function["strict"]?.boolValue == true)
        // Strict mode requires every property listed as required; optional ones become nullable.
        #expect(function["parameters"]?["required"] == .array(["city", "days"]))
        #expect(function["parameters"]?["properties"]?["days"]?["type"] == .array(["integer", "null"]))
    }

    @Test("Maps every tool choice", arguments: [
        (ToolChoice.auto, JSONValue.string("auto")),
        (.never, .string("none")),
        (.required, .string("required")),
    ])
    func mapsToolChoice(choice: ToolChoice, expected: JSONValue) async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(name: "t", description: "d", inputSchema: .object(properties: [:], required: []))
        )

        _ = try await provider(transport).languageModel("m")
            .generate(options(tools: [tool], toolChoice: choice))
        #expect(try transport.recordedRequestBody()["tool_choice"] == expected)
    }

    @Test("Names a specific tool when required")
    func mapsSpecificToolChoice() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(name: "weather", description: "d", inputSchema: .object(properties: [:], required: []))
        )

        _ = try await provider(transport).languageModel("m")
            .generate(options(tools: [tool], toolChoice: .tool(named: "weather")))

        let choice = try #require(try transport.recordedRequestBody()["tool_choice"])
        #expect(choice["function"]?["name"]?.stringValue == "weather")
    }

    @Test("Sends a JSON schema response format")
    func sendsSchemaResponseFormat() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let schema = JSONSchema.object(properties: ["name": .string()], required: ["name"])

        _ = try await provider(transport, quirks: .openAI).languageModel("m")
            .generate(options(responseFormat: .json(schema: schema, name: "person", description: "A person.")))

        let format = try #require(try transport.recordedRequestBody()["response_format"])
        #expect(format["type"]?.stringValue == "json_schema")
        #expect(format["json_schema"]?["name"]?.stringValue == "person")
        #expect(format["json_schema"]?["strict"]?.boolValue == true)
    }

    @Test("Falls back to an instruction when schemas cannot be enforced")
    func fallsBackToSchemaInstruction() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        var quirks = OpenAICompatibleQuirks.default
        quirks.supportsStructuredOutputs = false
        let schema = JSONSchema.object(properties: ["name": .string()], required: ["name"])

        let response = try await provider(transport, quirks: quirks).languageModel("m")
            .generate(options(responseFormat: .json(schema: schema, name: nil, description: nil)))

        let body = try transport.recordedRequestBody()
        #expect(body["response_format"]?["type"]?.stringValue == "json_object")
        // The schema still reaches the model, as an instruction rather than a constraint.
        let instruction = try #require(body["messages"]?[0]?["content"]?.stringValue)
        #expect(instruction.contains("JSON Schema"))
        #expect(instruction.contains("\"name\""))
        // And the caller is warned that conformance is no longer guaranteed.
        #expect(response.warnings.contains { $0.description.contains("not guaranteed") })
    }

    @Test("Merges provider options into the request body")
    func mergesProviderOptions() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("chat-completion.json")))
        let options = options(providerOptions: ["openaiCompatible": ["logit_bias": ["50256": -100]]])

        _ = try await provider(transport).languageModel("m").generate(options)
        #expect(try transport.recordedRequestBody()["logit_bias"]?["50256"]?.intValue == -100)
    }

    // MARK: - Streaming

    @Test("Streams text as a well-formed block", arguments: [1, 7, 64, 4096])
    func streamsText(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("chat-stream.sse"), chunkSize: chunkSize)
        )
        let model = provider(transport).languageModel("gpt-4o-mini")

        let response = try await model.stream(options())
        let parts = try await response.stream.collect()

        #expect(parts.first?.kind == .streamStart)
        #expect(parts.compactMap(\.textDelta).joined() == "Blue light scatters.")
        // Deltas are bracketed by a start and an end, whatever the wire format did.
        #expect(parts.contains { $0.kind == .textStart })
        #expect(parts.contains { $0.kind == .textEnd })

        guard case .finish(let reason, let usage, _)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .stop)
        #expect(usage.totalTokens == 14)
    }

    @Test("Reassembles a tool call from argument fragments", arguments: [1, 5, 64])
    func streamsToolCall(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("chat-stream-tools.sse"), chunkSize: chunkSize)
        )
        let model = provider(transport).languageModel("gpt-4o-mini")

        let parts = try await model.stream(options()).stream.collect()

        // Fragments are published as they arrive, then the parsed call once it is complete.
        let fragments = parts.compactMap { part -> String? in
            guard case .toolInputDelta(_, let delta) = part else { return nil }
            return delta
        }.joined()
        #expect(try JSONValue.parse(fragments) == ["city": "Malmö"])

        guard let call = parts.compactMap({ part -> ToolCallPart? in
            guard case .toolCall(let call) = part else { return nil }
            return call
        }).first else {
            Issue.record("Expected a parsed tool call.")
            return
        }
        #expect(call.toolCallID == "call_abc123")
        #expect(call.toolName == "weather")
        #expect(call.input == ["city": "Malmö"])
    }

    @Test("Closes the reasoning block when the answer begins")
    func streamsReasoningThenText() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("chat-stream-reasoning.sse"), chunkSize: 11)
        )
        let model = provider(transport).languageModel("deepseek-reasoner")

        let parts = try await model.stream(options()).stream.collect()
        let kinds = parts.map(\.kind)

        #expect(parts.compactMap(\.reasoningDelta).joined() == "Shorter wavelengths scatter.")
        #expect(parts.compactMap(\.textDelta).joined() == "Because of scattering.")

        // Reasoning must close before text opens, or a consumer cannot tell them apart.
        let reasoningEnd = try #require(kinds.firstIndex(of: .reasoningEnd))
        let textStart = try #require(kinds.firstIndex(of: .textStart))
        #expect(reasoningEnd < textStart)
    }

    @Test("Requests usage on streamed responses when supported")
    func requestsStreamUsage() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("chat-stream.sse"), chunkSize: 64)
        )
        _ = try await provider(transport).languageModel("m").stream(options())

        let body = try transport.recordedRequestBody()
        #expect(body["stream"]?.boolValue == true)
        // Without this, OpenAI reports no usage at all for streamed responses.
        #expect(body["stream_options"]?["include_usage"]?.boolValue == true)
    }

    @Test("Omits stream options for servers that reject them")
    func omitsStreamOptionsWhenUnsupported() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("chat-stream.sse"), chunkSize: 64)
        )
        _ = try await provider(transport, quirks: .ollama).languageModel("m").stream(options())
        #expect(try transport.recordedRequestBody()["stream_options"] == nil)
    }

    @Test("Reads the error body before failing a stream")
    func streamingErrorReadsBody() async throws {
        let transport = MockHTTPTransport(
            exchange: .failure(statusCode: 401, body: try fixture("error-401.json"))
        )

        var caught: APICallError?
        do {
            _ = try await provider(transport).languageModel("m").stream(options())
        } catch let error as APICallError {
            caught = error
        }
        #expect(try #require(caught).message.contains("Incorrect API key"))
    }

    // MARK: - Embeddings

    @Test("Sorts embeddings back into request order")
    func sortsEmbeddingsByIndex() async throws {
        // The fixture deliberately returns index 1 before index 0.
        let transport = MockHTTPTransport(exchange: .json(try fixture("embeddings.json")))
        let model = provider(transport).embeddingModel("text-embedding-3-small")

        let response = try await model.embed(EmbeddingModelCallOptions(values: ["first", "second"]))
        #expect(response.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        #expect(response.usage.inputTokens == 8)
    }

    @Test("Sends the values as an input array")
    func sendsEmbeddingInput() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("embeddings.json")))
        _ = try await provider(transport).embeddingModel("m")
            .embed(EmbeddingModelCallOptions(values: ["a", "b"]))

        let body = try transport.recordedRequestBody()
        #expect(body["input"] == .array(["a", "b"]))
        #expect(transport.recordedRequests.first?.url.absoluteString.hasSuffix("/embeddings") == true)
    }
}

/// A short label for a content part, for assertions about ordering.
extension ModelContent {
    var kindName: String {
        switch self {
        case .text: return "text"
        case .reasoning: return "reasoning"
        case .file: return "file"
        case .source: return "source"
        case .toolCall: return "toolCall"
        case .toolResult: return "toolResult"
        }
    }
}

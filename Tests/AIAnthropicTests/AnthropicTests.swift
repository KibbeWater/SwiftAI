import AIProviderSpec
import AIProviderUtils
import AITestSupport
import Foundation
import Testing

@testable import AIAnthropic

@Suite("Anthropic Messages")
struct AnthropicTests {
    private func fixture(_ name: String) throws -> String {
        try Fixtures.text(name, bundle: Bundle.module)
    }

    private func model(_ transport: MockHTTPTransport, defaultMaxTokens: Int = 8192) -> any LanguageModel {
        AnthropicProvider(
            apiKey: "test-key",
            defaultMaxTokens: defaultMaxTokens,
            transport: transport
        )
        .languageModel("claude-sonnet-4-5")
    }

    private func options(
        _ prompt: [ModelMessage] = [.user("Why is the sky blue?")],
        tools: [LanguageModelTool] = [],
        toolChoice: ToolChoice? = nil,
        responseFormat: ResponseFormat? = nil,
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        seed: Int? = nil,
        providerOptions: ProviderOptions? = nil
    ) -> LanguageModelCallOptions {
        LanguageModelCallOptions(
            prompt: prompt,
            maxOutputTokens: maxOutputTokens,
            temperature: temperature,
            seed: seed,
            responseFormat: responseFormat,
            tools: tools,
            toolChoice: toolChoice,
            providerOptions: providerOptions
        )
    }

    // MARK: - Buffered responses

    @Test("Decodes a message")
    func decodesMessage() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let response = try await model(transport).generate(options())

        #expect(response.content.text == "Blue light scatters more in the atmosphere.")
        #expect(response.finishReason == .stop)
        #expect(response.response?.id == "msg_01XyZaBcDeFgHiJkLmNoPq")
    }

    @Test("Counts cached tokens toward the input total")
    func countsCachedTokens() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let response = try await model(transport).generate(options())

        // The wire reports 18 fresh input tokens plus 120 read from cache, and counts neither
        // together; a caller reading `inputTokens` should see everything that was billed.
        #expect(response.usage.inputTokens == 138)
        #expect(response.usage.cachedInputTokens == 120)
        #expect(response.usage.totalTokens == 147)
        #expect(response.providerMetadata?["anthropic"]?["cacheReadInputTokens"]?.intValue == 120)
    }

    @Test("Decodes a tool call alongside text")
    func decodesToolUse() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message-tool-use.json")))
        let response = try await model(transport).generate(options())

        #expect(response.content.text == "Let me check that.")
        let call = try #require(response.content.toolCalls.first)
        #expect(call.toolCallID == "toolu_01A1B2C3")
        #expect(call.input == ["city": "Malmö"])
        #expect(response.finishReason == .toolCalls)
    }

    @Test("Preserves the signature on a thinking block")
    func preservesThinkingSignature() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message-thinking.json")))
        let response = try await model(transport).generate(options())

        guard case .reasoning(let reasoning)? = response.content.first else {
            Issue.record("Expected a reasoning part first.")
            return
        }
        // Without the signature, replaying this turn is rejected by the API.
        #expect(reasoning.signature == "ErUBCkYIBRgCIkD3xK2sig==")
        #expect(response.content.text == "Because of Rayleigh scattering.")
    }

    @Test("Maps an error response")
    func mapsErrorResponse() async throws {
        let transport = MockHTTPTransport(
            exchange: .failure(statusCode: 429, body: try fixture("error-429.json"))
        )

        var caught: APICallError?
        do {
            _ = try await model(transport).generate(options())
        } catch let error as APICallError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.message.contains("rate limit"))
        #expect(error.isRetryable)
    }

    @Test("Maps stop reasons", arguments: [
        ("end_turn", FinishReason.stop),
        ("stop_sequence", .stop),
        ("max_tokens", .length),
        ("tool_use", .toolCalls),
        ("refusal", .contentFilter),
        ("pause_turn", .other),
    ])
    func mapsStopReasons(raw: String, expected: FinishReason) {
        #expect(FinishReason.fromAnthropic(raw) == expected)
    }

    // MARK: - Request construction

    @Test("Sends the required headers")
    func sendsHeaders() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        _ = try await model(transport).generate(options())

        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["x-api-key"] == "test-key")
        #expect(request.headers["anthropic-version"] == "2023-06-01")
        #expect(request.url.absoluteString == "https://api.anthropic.com/v1/messages")
    }

    @Test("Always sends a token limit, since the API requires one")
    func alwaysSendsMaxTokens() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        _ = try await model(transport, defaultMaxTokens: 4096).generate(options())
        #expect(try transport.recordedRequestBody()["max_tokens"]?.intValue == 4096)
    }

    @Test("A call's token limit overrides the default")
    func callOverridesDefaultMaxTokens() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        _ = try await model(transport).generate(options(maxOutputTokens: 100))
        #expect(try transport.recordedRequestBody()["max_tokens"]?.intValue == 100)
    }

    @Test("Sends system instructions as a top-level field")
    func sendsSystemAtTopLevel() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        _ = try await model(transport).generate(options([.system("Be terse."), .user("Hi")]))

        let body = try transport.recordedRequestBody()
        #expect(body["system"]?[0]?["text"]?.stringValue == "Be terse.")
        // The system message must not also appear in the message list.
        #expect(body["messages"]?.arrayValue?.count == 1)
        #expect(body["messages"]?[0]?["role"]?.stringValue == "user")
    }

    @Test("Sends tool results as user content, not a tool role")
    func sendsToolResultsAsUserContent() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "t1", toolName: "weather", input: ["city": "Malmö"]))]),
            .tool([ToolResultPart(toolCallID: "t1", toolName: "weather", output: .text("Rain."))]),
        ]
        _ = try await model(transport).generate(options(prompt))

        let messages = try #require(try transport.recordedRequestBody()["messages"]?.arrayValue)
        #expect(messages.map { $0["role"]?.stringValue } == ["user", "assistant", "user"])
        #expect(messages[2]["content"]?[0]?["type"]?.stringValue == "tool_result")
        #expect(messages[2]["content"]?[0]?["tool_use_id"]?.stringValue == "t1")
    }

    @Test("Marks a failed tool result as an error")
    func marksToolErrors() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "t1", toolName: "weather", input: [:]))]),
            .tool([ToolResultPart(toolCallID: "t1", toolName: "weather", output: .errorText("No such city."))]),
        ]
        _ = try await model(transport).generate(options(prompt))

        let block = try #require(try transport.recordedRequestBody()["messages"]?[2]?["content"]?[0])
        #expect(block["is_error"]?.boolValue == true)
    }

    @Test("Merges consecutive same-role messages")
    func mergesAdjacentRoles() async throws {
        // Tool results become user messages, so a tool round trip would otherwise produce two
        // user turns in a row — which the API rejects.
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let prompt: [ModelMessage] = [
            .tool([ToolResultPart(toolCallID: "t1", toolName: "weather", output: .text("Rain."))]),
            .user("And tomorrow?"),
        ]
        _ = try await model(transport).generate(options(prompt))

        let messages = try #require(try transport.recordedRequestBody()["messages"]?.arrayValue)
        #expect(messages.count == 1)
        #expect(messages[0]["content"]?.arrayValue?.count == 2)
    }

    @Test("Replays a thinking block only when it is signed")
    func replaysSignedThinkingOnly() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let prompt: [ModelMessage] = [
            .user("Why?"),
            .assistant([
                .reasoning(ReasoningPart("Signed reasoning.", signature: "sig-abc")),
                .reasoning(ReasoningPart("Unsigned reasoning.")),
                .text("Answer."),
            ]),
            .user("And?"),
        ]
        _ = try await model(transport).generate(options(prompt))

        let blocks = try #require(try transport.recordedRequestBody()["messages"]?[1]?["content"]?.arrayValue)
        // The unsigned block is dropped: sending it back would be rejected.
        #expect(blocks.count == 2)
        #expect(blocks[0]["type"]?.stringValue == "thinking")
        #expect(blocks[0]["signature"]?.stringValue == "sig-abc")
    }

    @Test("Sends images as a base64 source")
    func sendsImages() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let prompt: [ModelMessage] = [
            .user([.text("What is this?"), .file(.data(Data([0x89, 0x50]), mediaType: "image/png"))])
        ]
        _ = try await model(transport).generate(options(prompt))

        let block = try #require(try transport.recordedRequestBody()["messages"]?[0]?["content"]?[1])
        #expect(block["type"]?.stringValue == "image")
        #expect(block["source"]?["type"]?.stringValue == "base64")
        #expect(block["source"]?["media_type"]?.stringValue == "image/png")
    }

    @Test("Applies a cache breakpoint to the last block of a message")
    func appliesCacheControl() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let prompt: [ModelMessage] = [
            .system("Long instructions.", providerOptions: ["anthropic": ["cacheControl": ["type": "ephemeral"]]]),
            .user("Hi"),
        ]
        _ = try await model(transport).generate(options(prompt))

        let system = try #require(try transport.recordedRequestBody()["system"]?[0])
        #expect(system["cache_control"]?["type"]?.stringValue == "ephemeral")
    }

    @Test("Warns about settings the API cannot express")
    func warnsAboutUnsupportedSettings() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let response = try await model(transport).generate(options(seed: 42))

        #expect(response.warnings.contains { $0.description.contains("seed") })
    }

    @Test("Maps tool choices", arguments: [
        (ToolChoice.auto, "auto"),
        (.never, "none"),
        (.required, "any"),
    ])
    func mapsToolChoice(choice: ToolChoice, expected: String) async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let tool = LanguageModelTool.function(
            FunctionTool(name: "t", description: "d", inputSchema: .object(properties: [:], required: []))
        )
        _ = try await model(transport).generate(options(tools: [tool], toolChoice: choice))

        #expect(try transport.recordedRequestBody()["tool_choice"]?["type"]?.stringValue == expected)
    }

    @Test("Passes provider options through to the body")
    func passesProviderOptions() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message-thinking.json")))
        let options = options(
            providerOptions: ["anthropic": ["thinking": ["type": "enabled", "budget_tokens": 4096]]]
        )
        _ = try await model(transport).generate(options)

        let thinking = try #require(try transport.recordedRequestBody()["thinking"])
        #expect(thinking["type"]?.stringValue == "enabled")
        #expect(thinking["budget_tokens"]?.intValue == 4096)
    }

    // MARK: - Structured output

    @Test("Constrains output by forcing a tool call")
    func structuredOutputForcesTool() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("message.json")))
        let schema = JSONSchema.object(properties: ["name": .string()], required: ["name"])

        _ = try await model(transport).generate(
            options(responseFormat: .json(schema: schema, name: "person", description: nil))
        )

        // The API has no response format, so a forced tool call provides the same guarantee.
        let body = try transport.recordedRequestBody()
        #expect(body["tool_choice"]?["type"]?.stringValue == "tool")
        #expect(body["tools"]?[0]?["input_schema"]?["properties"]?["name"] != nil)
    }

    @Test("Republishes the forced tool call as text")
    func structuredOutputSurfacesAsText() async throws {
        let payload = """
            {"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5",
             "content":[{"type":"tool_use","id":"t1","name":"respond_with_structured_output",
                         "input":{"name":"Ada"}}],
             "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}
            """
        let transport = MockHTTPTransport(exchange: .json(payload))
        let schema = JSONSchema.object(properties: ["name": .string()], required: ["name"])

        let response = try await model(transport).generate(
            options(responseFormat: .json(schema: schema, name: nil, description: nil))
        )

        // A caller who asked for JSON should receive JSON, not a tool call they never registered.
        #expect(response.content.text == #"{"name":"Ada"}"#)
        #expect(response.content.toolCalls.isEmpty)
        #expect(response.finishReason == .stop)
    }

    // MARK: - Streaming

    @Test("Streams text", arguments: [1, 9, 128, 8192])
    func streamsText(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("message-stream.sse"), chunkSize: chunkSize)
        )
        let parts = try await model(transport).stream(options()).stream.collect()

        #expect(parts.compactMap(\.textDelta).joined() == "Blue light scatters.")
        #expect(parts.first?.kind == .streamStart)

        guard case .finish(let reason, let usage, _)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .stop)
        #expect(usage.inputTokens == 138)
        #expect(usage.outputTokens == 9)
    }

    @Test("Ignores keep-alive pings")
    func ignoresPings() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("message-stream.sse"), chunkSize: 64)
        )
        let parts = try await model(transport).stream(options()).stream.collect()

        // A `ping` event carries nothing and must not appear as a content part.
        #expect(parts.filter { $0.kind == .textStart }.count == 1)
    }

    @Test("Streams a tool call after text", arguments: [1, 11, 4096])
    func streamsToolCall(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("message-stream-tools.sse"), chunkSize: chunkSize)
        )
        let parts = try await model(transport).stream(options()).stream.collect()

        #expect(parts.compactMap(\.textDelta).joined() == "Checking.")

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
        #expect(call.toolCallID == "toolu_01A1B2C3")
        #expect(call.toolName == "weather")
    }

    @Test("Attaches a streamed signature to the reasoning block")
    func streamsThinkingWithSignature() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("message-stream-thinking.sse"), chunkSize: 23)
        )
        let parts = try await model(transport).stream(options()).stream.collect()

        #expect(parts.compactMap(\.reasoningDelta).joined() == "Shorter wavelengths scatter more.")
        #expect(parts.compactMap(\.textDelta).joined() == "Rayleigh scattering.")

        // The signature arrives as its own delta type and belongs on the block's end part.
        guard let signature = parts.compactMap({ part -> String? in
            guard case .reasoningEnd(_, let metadata) = part else { return nil }
            return metadata?["anthropic"]?["signature"]?.stringValue
        }).first else {
            Issue.record("Expected a signature on the reasoning end part.")
            return
        }
        #expect(signature == "ErUBCkYIBRgCIkD3xK2sig==")
    }

    @Test("Does not stop the stream at a done sentinel")
    func doesNotStopAtDoneSentinel() async throws {
        // Anthropic ends with `message_stop`; a payload of `[DONE]` would be ordinary content.
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("message-stream.sse"), chunkSize: 64)
        )
        let parts = try await model(transport).stream(options()).stream.collect()
        #expect(parts.last?.kind == .finish)
    }
}

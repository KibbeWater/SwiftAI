import AIProviderSpec
import AIProviderUtils
import AITestSupport
import Foundation
import SwiftAI
import Testing

@testable import AIOpenRouter

/// Tests driven by payloads recorded from OpenRouter's API.
///
/// The fixtures are verbatim, including the whitespace OpenRouter pads buffered responses with
/// and the `: OPENROUTER PROCESSING` comments it sends while an upstream warms up. The one
/// exception is image data, replaced with a 1×1 PNG to keep the repository small.
@Suite("OpenRouter chat")
struct OpenRouterChatTests {
    // MARK: - Helpers

    static func fixture(_ name: String) throws -> String {
        try Fixtures.text(name, bundle: Bundle.module)
    }

    static func provider(
        _ transport: MockHTTPTransport,
        appName: String? = nil,
        appURL: URL? = nil,
        providerAPIKeys: [String: String] = [:],
        extraBody: [String: JSONValue] = [:]
    ) -> OpenRouterProvider {
        OpenRouterProvider(
            apiKey: "test-key",
            appName: appName,
            appURL: appURL,
            providerAPIKeys: providerAPIKeys,
            extraBody: extraBody,
            transport: transport
        )
    }

    private func generate(
        _ fixtureName: String,
        prompt: [ModelMessage] = [.user("Hi")],
        tools: [LanguageModelTool] = [],
        providerOptions: ProviderOptions? = nil
    ) async throws -> (LanguageModelResponse, MockHTTPTransport) {
        let transport = MockHTTPTransport(exchange: .json(try Self.fixture(fixtureName)))
        let response = try await Self.provider(transport).languageModel("m").generate(
            LanguageModelCallOptions(prompt: prompt, tools: tools, providerOptions: providerOptions)
        )
        return (response, transport)
    }

    private func stream(_ fixtureName: String, chunkSize: Int = 64) async throws -> [LanguageModelStreamPart] {
        let transport = MockHTTPTransport(exchange: .serverSentEvents(try Self.fixture(fixtureName), chunkSize: chunkSize))
        return try await Self.provider(transport).languageModel("m")
            .stream(LanguageModelCallOptions(prompt: [.user("Hi")]))
            .stream.collect()
    }

    /// The request body a prompt produces, without a real response.
    private func requestBody(
        _ prompt: [ModelMessage],
        tools: [LanguageModelTool] = [],
        responseFormat: ResponseFormat? = nil,
        providerOptions: ProviderOptions? = nil,
        extraBody: [String: JSONValue] = [:]
    ) async throws -> JSONValue {
        let transport = MockHTTPTransport(exchange: .json(try Self.fixture("chat-completion.json")))
        _ = try await Self.provider(transport, extraBody: extraBody).languageModel("m").generate(
            LanguageModelCallOptions(prompt: prompt, responseFormat: responseFormat, tools: tools, providerOptions: providerOptions)
        )
        return try transport.recordedRequestBody()
    }

    // MARK: - Buffered responses

    @Test("Decodes a completion, including OpenRouter's cost and routing")
    func decodesCompletion() async throws {
        let (response, _) = try await generate("chat-completion.json")

        #expect(response.content.text.hasPrefix("The sky appears blue"))
        #expect(response.finishReason == .stop)
        #expect(response.usage.inputTokens == 18)
        #expect(response.usage.outputTokens == 27)
        #expect(response.response?.id == "gen-1790248151-IK4JO74cMbx8faZrZL23")
        #expect(response.providerMetadata?["openrouter"]?["cost"]?.numberValue == 0.0000126)
        #expect(response.providerMetadata?["openrouter"]?["upstreamInferenceCost"]?.numberValue == 0.0000126)
        #expect(response.providerMetadata?["openrouter"]?["provider"] == "OpenAI")
    }

    @Test("Decodes a tool call")
    func decodesToolCall() async throws {
        let (response, _) = try await generate("chat-tool-calls.json")
        let call = try #require(response.content.toolCalls.first)

        #expect(call.toolCallID == "call_lnfITuFsQiAW7zsYkgjgiiHd")
        #expect(call.input == ["city": "Malmö"])
        #expect(response.finishReason == .toolCalls)
    }

    @Test("Keeps Anthropic's signed reasoning details on the reasoning part")
    func decodesSignedReasoning() async throws {
        let (response, _) = try await generate("chat-reasoning-anthropic.json")
        let reasoning = try #require(response.content.first.flatMap { if case .reasoning(let part) = $0 { part } else { nil } })
        let details = try #require(reasoning.providerOptions?.value("reasoning_details", for: "openrouter")?.arrayValue)

        #expect(reasoning.text.hasSuffix("= 391"))
        #expect(details.count == 1)
        #expect(details[0]["signature"]?.stringValue?.isEmpty == false)
        #expect(response.content.text == "391")
        #expect(response.usage.reasoningTokens == 67)
    }

    @Test("Shows an OpenAI reasoning summary while keeping its encrypted blob")
    func decodesSummaryAndEncrypted() async throws {
        let (response, _) = try await generate("chat-reasoning-openai.json")
        let reasoning = try #require(response.content.first.flatMap { if case .reasoning(let part) = $0 { part } else { nil } })
        let types = reasoning.providerOptions?.value("reasoning_details", for: "openrouter")?.arrayValue?.compactMap { $0["type"]?.stringValue }

        #expect(reasoning.text.hasPrefix("**Calculating a simple multiplication**"))
        #expect(types == ["reasoning.summary", "reasoning.encrypted"])
    }

    @Test("Turns web search citations into sources")
    func decodesCitations() async throws {
        let (response, _) = try await generate("chat-web-search.json")
        let sources = response.content.sources

        #expect(sources.map(\.id) == ["https://github.com/swiftlang/swift/blob/main/CHANGELOG.md", "https://swiftversion.net/"])
        #expect(sources.first?.title == "CHANGELOG.md")
    }

    @Test("Turns generated images into file parts")
    func decodesImages() async throws {
        let (response, _) = try await generate("chat-image.json")
        let image = try #require(response.content.files.first)

        #expect(image.mediaType == "image/png")
        #expect(image.data?.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
        #expect(response.content.text == "Here you go: ")
    }

    @Test("Treats an error body sent with 200 OK as a failure")
    func throwsOnErrorWithSuccessStatus() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"error":{"message":"Upstream timed out","code":502}}"#))
        do {
            _ = try await Self.provider(transport).languageModel("m").generate(LanguageModelCallOptions(prompt: [.user("Hi")]))
            Issue.record("Expected an APICallError")
        } catch let error as APICallError {
            #expect(error.statusCode == 200)
            #expect(error.message.contains("Upstream timed out"))
            // The embedded code, not the 200, says whether a retry could help.
            #expect(error.isRetryable)
        }
    }

    @Test("Reports the upstream provider's own error message")
    func mapsUpstreamError() async throws {
        let transport = MockHTTPTransport(exchange: .failure(statusCode: 400, body: try Self.fixture("error-400-upstream.json")))
        do {
            _ = try await Self.provider(transport).languageModel("m").generate(LanguageModelCallOptions(prompt: [.user("Hi")]))
            Issue.record("Expected an APICallError")
        } catch let error as APICallError {
            // The top-level message is only "Provider returned error"; the cause is in `raw`.
            #expect(error.message.contains("[OpenAI] Invalid 'tools[0].name'"))
            #expect(error.isRetryable == false)
        }
    }

    @Test("Reports an authentication failure")
    func mapsAuthenticationError() async throws {
        let transport = MockHTTPTransport(exchange: .failure(statusCode: 401, body: try Self.fixture("error-401.json")))
        await #expect(throws: APICallError.self) {
            _ = try await Self.provider(transport).languageModel("m").generate(LanguageModelCallOptions(prompt: [.user("Hi")]))
        }
    }

    // MARK: - Streaming

    @Test("Streams text as a well-formed block", arguments: [1, 7, 64, 4096])
    func streamsText(chunkSize: Int) async throws {
        let parts = try await stream("chat-stream.sse", chunkSize: chunkSize)

        #expect(parts.first?.kind == .streamStart)
        #expect(parts.compactMap(\.textDelta).joined().hasSuffix("more effectively than other colors."))
        #expect(parts.filter { $0.kind == .textStart }.count == 1)
        guard case .finish(let reason, let usage, let metadata)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .stop)
        #expect(usage.totalTokens == 45)
        #expect(metadata?["openrouter"]?["cost"]?.numberValue == 0.0000126)
    }

    @Test("Streams a tool call", arguments: [1, 7, 4096])
    func streamsToolCall(chunkSize: Int) async throws {
        let parts = try await stream("chat-stream-tool-calls.sse", chunkSize: chunkSize)
        let call = try #require(parts.compactMap { if case .toolCall(let call) = $0 { call } else { nil } }.first)

        #expect(call.input == ["city": "Malmö"])
        let blockKinds: [LanguageModelStreamPart.Kind] = parts.map(\.kind).filter { $0 == .toolInputStart || $0 == .toolInputEnd }
        #expect(blockKinds == [.toolInputStart, .toolInputEnd])
        guard case .finish(let reason, _, _)? = parts.last else { return }
        #expect(reason == .toolCalls)
    }

    @Test("Assembles streamed reasoning fragments into one signed detail", arguments: [1, 64, 4096])
    func assemblesSignedReasoning(chunkSize: Int) async throws {
        let parts = try await stream("chat-stream-reasoning-anthropic.sse", chunkSize: chunkSize)
        guard case .reasoningEnd(_, let metadata)? = parts.first(where: { $0.kind == .reasoningEnd }) else {
            Issue.record("Expected a reasoning block")
            return
        }
        let details = try #require(metadata?["openrouter"]?["reasoning_details"]?.arrayValue)

        // Twelve wire fragments, one of them only a signature, make one detail.
        #expect(details.count == 1)
        #expect(details[0]["text"]?.stringValue == parts.compactMap(\.reasoningDelta).joined())
        #expect(details[0]["signature"]?.stringValue?.isEmpty == false)
        // Reasoning closes before the answer opens.
        let kinds = parts.map(\.kind)
        #expect(kinds.firstIndex(of: .reasoningEnd)! < kinds.firstIndex(of: .textStart)!)
    }

    @Test("Carries the signed reasoning on the first tool call", arguments: [1, 4096])
    func reasoningRidesOnToolCall(chunkSize: Int) async throws {
        let parts = try await stream("chat-stream-reasoning-tools-anthropic.sse", chunkSize: chunkSize)
        let call = try #require(parts.compactMap { if case .toolCall(let call) = $0 { call } else { nil } }.first)
        let details = try #require(call.providerOptions?.value("reasoning_details", for: "openrouter")?.arrayValue)

        #expect(call.input == ["city": "Malmö"])
        #expect(details.count == 1)
        #expect(details[0]["signature"]?.stringValue?.isEmpty == false)
    }

    @Test("Keeps encrypted-only reasoning, which has no visible text")
    func keepsEncryptedReasoning() async throws {
        let parts = try await stream("chat-stream-reasoning-openai.sse")
        guard case .reasoningEnd(_, let metadata)? = parts.first(where: { $0.kind == .reasoningEnd }) else {
            Issue.record("Expected a reasoning block even without visible text")
            return
        }
        #expect(parts.compactMap(\.reasoningDelta).isEmpty)
        #expect(metadata?["openrouter"]?["reasoning_details"]?[0]?["type"] == "reasoning.encrypted")
    }

    @Test("Fails the stream on an error chunk")
    func failsOnErrorChunk() async throws {
        let body = """
            data: {"id":"gen-1","choices":[{"index":0,"delta":{"content":"Hel"}}]}

            data: {"error":{"message":"Upstream disconnected","code":502}}


            """
        let transport = MockHTTPTransport(exchange: .serverSentEvents(body))
        let response = try await Self.provider(transport).languageModel("m").stream(LanguageModelCallOptions(prompt: [.user("Hi")]))
        await #expect(throws: APICallError.self) { _ = try await response.stream.collect() }
    }

    // MARK: - Request construction

    @Test("Sends credentials, attribution, and bring-your-own-key headers")
    func sendsHeaders() async throws {
        let transport = MockHTTPTransport(exchange: .json(try Self.fixture("chat-completion.json")))
        let provider = Self.provider(
            transport,
            appName: "SwiftAI Tests",
            appURL: URL(string: "https://example.com")!,
            providerAPIKeys: ["anthropic": "sk-ant"]
        )
        _ = try await provider.languageModel("m").generate(LanguageModelCallOptions(prompt: [.user("Hi")]))
        let request = try #require(transport.recordedRequests.first)

        #expect(request.url.absoluteString == "https://openrouter.ai/api/v1/chat/completions")
        #expect(request.headers["Authorization"] == "Bearer test-key")
        #expect(request.headers["X-OpenRouter-Title"] == "SwiftAI Tests")
        #expect(request.headers["HTTP-Referer"] == "https://example.com")
        #expect(request.headers["X-Provider-API-Keys"] == #"{"anthropic":"sk-ant"}"#)
    }

    @Test("Sends system messages as content arrays and plain user text as a string")
    func sendsMessageShapes() async throws {
        let body = try await requestBody([.system("Be brief."), .user("Hi")])

        #expect(body["messages"]?[0] == ["role": "system", "content": [["type": "text", "text": "Be brief."]]])
        #expect(body["messages"]?[1]?["content"] == "Hi")
    }

    @Test("Places a message's cache breakpoint on its last text part")
    func placesCacheControl() async throws {
        let cache: ProviderOptions = ["anthropic": ["cacheControl": ["type": "ephemeral"]]]
        let body = try await requestBody([
            .user([.text("Long document…"), .text("Question?")], providerOptions: cache),
        ])
        let parts = try #require(body["messages"]?[0]?["content"]?.arrayValue)

        // Anthropic's spelling is honored, since prompts are often written for Anthropic first.
        #expect(parts[0]["cache_control"] == nil)
        #expect(parts[1]["cache_control"] == ["type": "ephemeral"])
    }

    @Test("Merges provider options verbatim, over the provider's extra body")
    func mergesProviderOptions() async throws {
        let body = try await requestBody(
            [.user("Hi")],
            providerOptions: ["openrouter": [
                "models": ["a/b", "c/d"],
                "provider": ["sort": "price"],
                "cacheControl": ["type": "ephemeral"],
            ]],
            extraBody: ["provider": ["sort": "latency"], "user": "u-1"]
        )

        #expect(body["models"] == ["a/b", "c/d"])
        #expect(body["provider"] == ["sort": "price"])
        #expect(body["user"] == "u-1")
        #expect(body["cache_control"] == ["type": "ephemeral"])
        #expect(body["cacheControl"] == nil)
    }

    @Test("Asks for usage on streamed responses")
    func requestsStreamUsage() async throws {
        let transport = MockHTTPTransport(exchange: .serverSentEvents(try Self.fixture("chat-stream.sse")))
        _ = try await Self.provider(transport).languageModel("m").stream(LanguageModelCallOptions(prompt: [.user("Hi")])).stream.collect()

        #expect(try transport.recordedRequestBody()["stream_options"] == ["include_usage": true])
    }

    @Test("Requests a strict JSON schema")
    func requestsJSONSchema() async throws {
        let body = try await requestBody(
            [.user("Hi")],
            responseFormat: .json(schema: .object(JSONSchema.ObjectConstraints(properties: ["a": .string()], required: ["a"])), name: "thing")
        )
        #expect(body["response_format"]?["type"] == "json_schema")
        #expect(body["response_format"]?["json_schema"]?["name"] == "thing")
        #expect(body["response_format"]?["json_schema"]?["strict"] == true)
    }

    @Test("Maps the web search tool and skips another provider's tools")
    func mapsServerTools() async throws {
        let transport = MockHTTPTransport(exchange: .json(try Self.fixture("chat-completion.json")))
        let response = try await Self.provider(transport).languageModel("m").generate(LanguageModelCallOptions(
            prompt: [.user("Hi")],
            tools: [
                OpenRouterTools.webSearch(maxResults: 3, engine: "exa"),
                .providerDefined(ProviderDefinedTool(id: "anthropic.web_search_20250305", name: "web_search")),
            ]
        ))
        let tools = try #require(try transport.recordedRequestBody()["tools"]?.arrayValue)

        #expect(tools == [["type": "openrouter:web_search", "max_results": 3, "engine": "exa"]])
        #expect(response.warnings.count == 1)
    }

    @Test("Encodes media by type", arguments: [
        ("audio/mp4", "input_audio"),
        ("video/mp4", "video_url"),
        ("image/png", "image_url"),
        ("application/pdf", "file"),
    ])
    func encodesMedia(mediaType: String, partType: String) async throws {
        let body = try await requestBody([.user([.file(.data(Data([1, 2, 3]), mediaType: mediaType, filename: "f"))])])
        let part = body["messages"]?[0]?["content"]?[0]

        #expect(part?["type"]?.stringValue == partType)
        if partType == "input_audio" {
            // `audio/mp4` is an M4A container, and OpenRouter names the format that way.
            #expect(part?["input_audio"]?["format"] == "m4a")
        }
    }

    @Test("Rejects audio by URL, which OpenRouter cannot fetch")
    func rejectsAudioURL() async throws {
        await #expect(throws: UnsupportedFunctionalityError.self) {
            try await requestBody([.user([.file(.url(URL(string: "https://example.com/a.mp3")!, mediaType: "audio/mpeg"))])])
        }
    }

    @Test("Reports which URLs OpenRouter fetches itself")
    func supportsNativeURLs() {
        let model = OpenRouterProvider(apiKey: "k").languageModel("m")
        #expect(model.supportsNativeURL(URL(string: "https://example.com/a.PNG?x=1")!, mediaType: "image/png"))
        #expect(!model.supportsNativeURL(URL(string: "https://example.com/image")!, mediaType: "image/png"))
        #expect(model.supportsNativeURL(URL(string: "https://example.com/doc")!, mediaType: "application/pdf"))
        #expect(!model.supportsNativeURL(URL(string: "https://example.com/a.mp3")!, mediaType: "audio/mpeg"))
    }

    @Test("Sends tool results with the tool's name and multimodal content")
    func sendsToolResults() async throws {
        let body = try await requestBody([
            .user("Screenshot the page"),
            .assistant([.toolCall(ToolCallPart(toolCallID: "c1", toolName: "screenshot", input: ["b": 2, "a": 1]))]),
            .tool([ToolResultPart(toolCallID: "c1", toolName: "screenshot", output: .content([
                .text("Here it is"),
                .file(.data(Data([1]), mediaType: "image/png")),
            ]))]),
        ])
        let assistant = body["messages"]?[1]
        let tool = body["messages"]?[2]

        #expect(assistant?["content"] == .null)
        // Sorted keys keep the prompt byte-identical across turns, so prompt caching holds.
        #expect(assistant?["tool_calls"]?[0]?["function"]?["arguments"] == #"{"a":1,"b":2}"#)
        #expect(tool?["name"] == "screenshot")
        #expect(tool?["content"]?[1]?["type"] == "image_url")
    }

    // MARK: - Reasoning replay

    @Test("Replays streamed reasoning on the next turn, signature intact")
    func replaysStreamedReasoning() async throws {
        // Turn one: stream a reasoning + tool call response through the core, as an app would.
        let first = MockHTTPTransport(exchange: .serverSentEvents(try Self.fixture("chat-stream-reasoning-tools-anthropic.sse")))
        let result = streamText(model: Self.provider(first).languageModel("anthropic/claude-haiku-4.5"), prompt: "Weather in Malmö?")
        let history: [ModelMessage] = [.user("Weather in Malmö?")] + (try await result.responseMessages)

        // Turn two: the assistant message must carry the signed details back exactly once.
        let body = try await requestBody(history + [.tool([ToolResultPart(toolCallID: "x", toolName: "weather", output: .text("Rain"))])])
        let assistant = try #require(body["messages"]?.arrayValue?.first { $0["role"] == "assistant" })
        let details = try #require(assistant["reasoning_details"]?.arrayValue)

        #expect(details.count == 1)
        #expect(details[0]["signature"]?.stringValue?.isEmpty == false)
        #expect(assistant["reasoning"]?.stringValue?.isEmpty == false)
    }

    @Test("Sends each reasoning detail only once across a prompt")
    func deduplicatesDetails() async throws {
        let detail: JSONValue = ["type": "reasoning.encrypted", "id": "rs_1", "data": "blob"]
        let reasoning = ModelContent.reasoning(ReasoningPart("", providerOptions: ["openrouter": ["reasoning_details": [detail]]]))
        // The same assistant turn twice, as happens when history is rebuilt from a store that
        // already contained it.
        let body = try await requestBody([
            .user("Hi"), .assistant([reasoning, .text("A")]),
            .user("Again"), .assistant([reasoning, .text("B")]),
            .user("Once more"),
        ])
        let assistants = body["messages"]?.arrayValue?.filter { $0["role"] == "assistant" } ?? []

        #expect(assistants[0]["reasoning_details"] == [detail])
        #expect(assistants[1]["reasoning_details"] == nil)
    }

    @Test("Drops unsigned Anthropic reasoning an upstream would reject")
    func stripsUnsignedReasoning() async throws {
        let unsigned: JSONValue = ["type": "reasoning.text", "text": "Hmm", "format": "anthropic-claude-v1"]
        let openAI: JSONValue = ["type": "reasoning.text", "text": "Hmm", "format": "openai-responses-v1"]
        let body = try await requestBody([
            .user("Hi"),
            .assistant([.reasoning(ReasoningPart("Hmm", providerOptions: ["openrouter": ["reasoning_details": [unsigned, openAI]]])), .text("A")]),
            .user("Next"),
        ])

        #expect(body["messages"]?[1]?["reasoning_details"] == [openAI])
    }

    @Test("Echoes an explicit empty reasoning_details, which DeepSeek requires")
    func echoesEmptyDetails() async throws {
        let body = try await requestBody([
            .user("Hi"),
            .assistant([.text("A")], providerOptions: ["openrouter": ["reasoning_details": []]]),
            .user("Next"),
        ])
        #expect(body["messages"]?[1]?["reasoning_details"] == [])
    }
}

// MARK: - Other model kinds

@Suite("OpenRouter models")
struct OpenRouterModelTests {
    private func fixture(_ name: String) throws -> String { try OpenRouterChatTests.fixture(name) }

    @Test("Decodes embeddings in index order")
    func decodesEmbeddings() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("embeddings.json")))
        let response = try await OpenRouterChatTests.provider(transport).embeddingModel("thenlper/gte-base")
            .embed(EmbeddingModelCallOptions(values: ["The sky is blue.", "Grass is green."]))

        #expect(response.embeddings.count == 2)
        #expect(response.embeddings[0].count == 768)
        #expect(try transport.recordedRequestBody()["input"] == ["The sky is blue.", "Grass is green."])
    }

    @Test("Decodes generated images")
    func decodesImages() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("images.json")))
        let response = try await OpenRouterChatTests.provider(transport).imageModel("google/gemini-2.5-flash-image")
            .generate(ImageModelCallOptions(prompt: "A red circle", aspectRatio: AspectRatio(width: 16, height: 9)))

        #expect(response.images.first?.mediaType == "image/png")
        #expect(response.providerMetadata?["openrouter"]?["cost"]?.numberValue == 0.038703)
        #expect(transport.recordedRequests.first?.url.path == "/api/v1/images")
        #expect(try transport.recordedRequestBody()["aspect_ratio"] == "16:9")
    }
}

@Suite("OpenRouter decisions")
struct OpenRouterDecisionTests {
    private let questions: EvaluationQuestions = [
        "department": .choice(
            "Which team should handle this?",
            options: ["billing": "Payments, charges and refunds", "technical": "Bugs and outages", "other": nil]
        ),
        "severity": .score("How severe is the issue?", levels: ["Cosmetic", "Workaround exists", "Blocking; no workaround"]),
        "refund": .boolean("Is the customer asking for money back?"),
    ]

    private func evaluateFixture() async throws -> (EvaluationResult, MockHTTPTransport) {
        let transport = MockHTTPTransport(exchange: .json(try OpenRouterChatTests.fixture("decisions.json")))
        let result = try await evaluate(
            model: OpenRouterChatTests.provider(transport).decisionModel("typesafe/jev-1.13"),
            state: ["message": "I was charged twice for my subscription this month. Please refund the extra charge."],
            questions: questions
        )
        return (result, transport)
    }

    @Test("Decodes native answers and passes the core's validation")
    func decodesAnswers() async throws {
        // Run through `evaluate` so the recorded answers are held to the full validation rules:
        // the score of 1.08 only agrees with its rounded distribution within the declared rounding.
        let (result, _) = try await evaluateFixture()

        #expect(result["department"] == .choice("billing", probabilities: ["billing": 1, "technical": 0, "other": 0]))
        #expect(result["severity"] == .score(1.08, probabilities: [0: 0.14, 1: 0.64, 2: 0.22]))
        #expect(result["refund"] == .boolean(probability: 0.99))
        #expect(result.rounding == EvaluationRounding(probabilityDecimals: 2, scoreDecimals: 2))
        #expect(result.usage.totalTokens == 471)
        #expect(result.providerMetadata?["openrouter"]?["confidence"]?["severity"]?.numberValue == 0.46)
        #expect(result.response.modelID == "typesafe/jev-1.13-20260917")
    }

    @Test("Sends booleans as noul to the alpha Decisions endpoint")
    func sendsWireQuestions() async throws {
        let (_, transport) = try await evaluateFixture()
        let body = try transport.recordedRequestBody()

        #expect(transport.recordedRequests.first?.url.absoluteString == "https://openrouter.ai/api/alpha/decisions")
        #expect(body["questions"]?["refund"] == ["type": "noul", "instructions": "Is the customer asking for money back?"])
        #expect(body["questions"]?["department"]?["criteria"]?["other"] == .null)
        #expect(body["model"] == "typesafe/jev-1.13")
    }

    @Test("Rejects question shapes the Decisions API does not accept, before any request")
    func rejectsUnsupportedShapes() async throws {
        let transport = MockHTTPTransport(exchange: .json(try OpenRouterChatTests.fixture("decisions.json")))
        let model = OpenRouterChatTests.provider(transport).decisionModel("typesafe/jev-1.13")

        await #expect(throws: InvalidArgumentError.self) {
            try await evaluate(model: model, state: "x", questions: ["s": .score("?", levels: ["Low", nil])])
        }
        await #expect(throws: InvalidArgumentError.self) {
            try await evaluate(model: model, state: "x", questions: ["b": .boolean("?", whenTrue: "Yes")])
        }
        #expect(transport.requestCount == 0)
    }

    @Test("Derives the Decisions URL from a custom base URL")
    func derivesDecisionsURL() {
        #expect(OpenRouterProvider.deriveDecisionsBaseURL(from: URL(string: "https://proxy.example.com/api/v1/")!)?.absoluteString
            == "https://proxy.example.com/api/alpha")
        #expect(OpenRouterProvider.deriveDecisionsBaseURL(from: URL(string: "https://proxy.example.com/openrouter")!) == nil)
    }

    @Test("Explains how to fix an underivable Decisions URL")
    func explainsMissingDecisionsURL() async throws {
        let provider = OpenRouterProvider(apiKey: "k", baseURL: URL(string: "https://proxy.example.com/openrouter")!)
        do {
            _ = try await evaluate(model: provider.decisionModel("m"), state: "x", questions: ["b": .boolean("?")])
            Issue.record("Expected an InvalidArgumentError")
        } catch let error as InvalidArgumentError {
            #expect(error.message.contains("decisionsBaseURL"))
        }
    }

    @Test("Resolves decision models as evaluation models")
    func resolvesAsEvaluationModel() throws {
        let registry = ProviderRegistry(["openrouter": OpenRouterProvider(apiKey: "k")])
        #expect(try registry.evaluationModel("openrouter:typesafe/jev-1.13").modelID == "typesafe/jev-1.13")
    }
}

import AIProviderSpec
import AIProviderUtils
import AITestSupport
import Foundation
import Testing

@testable import AIOpenAI

@Suite("OpenAI Responses")
struct OpenAIResponsesTests {
    private func fixture(_ name: String) throws -> String {
        try Fixtures.text(name, bundle: Bundle.module)
    }

    private func provider(_ transport: MockHTTPTransport) -> OpenAIProvider {
        OpenAIProvider(apiKey: "test-key", transport: transport)
    }

    private func options(
        _ prompt: [ModelMessage] = [.user("Why is the sky blue?")],
        tools: [LanguageModelTool] = [],
        toolChoice: ToolChoice? = nil,
        responseFormat: ResponseFormat? = nil,
        maxOutputTokens: Int? = nil,
        seed: Int? = nil,
        providerOptions: ProviderOptions? = nil
    ) -> LanguageModelCallOptions {
        LanguageModelCallOptions(
            prompt: prompt,
            maxOutputTokens: maxOutputTokens,
            seed: seed,
            responseFormat: responseFormat,
            tools: tools,
            toolChoice: toolChoice,
            providerOptions: providerOptions
        )
    }

    // MARK: - Buffered responses

    @Test("Decodes reasoning and text output items")
    func decodesOutputItems() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let response = try await provider(transport).languageModel("gpt-5").generate(options())

        #expect(response.content.reasoningText == "Shorter wavelengths scatter more.")
        #expect(response.content.text == "Blue light scatters more in the atmosphere.")
        #expect(response.finishReason == .stop)
        #expect(response.usage.reasoningTokens == 32)
        #expect(response.usage.cachedInputTokens == 6)
        #expect(response.response?.id == "resp_68b1c2d3e4f5")
    }

    @Test("Keeps the reasoning item identifier for replay")
    func keepsReasoningItemIdentifier() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let response = try await provider(transport).languageModel("gpt-5").generate(options())

        guard case .reasoning(let reasoning)? = response.content.first else {
            Issue.record("Expected a reasoning part.")
            return
        }
        // Without these, the model restarts its reasoning on every turn.
        #expect(reasoning.providerOptions?.value("itemId", for: "openai")?.stringValue == "rs_68b1c2d3aaa")
        #expect(
            reasoning.providerOptions?.value("encryptedContent", for: "openai")?.stringValue
                == "gAAAAABo_encrypted"
        )
    }

    @Test("Replays reasoning items on the next turn")
    func replaysReasoningItems() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let prompt: [ModelMessage] = [
            .user("Why?"),
            .assistant([
                .reasoning(
                    ReasoningPart(
                        "Earlier reasoning.",
                        providerOptions: [
                            "openai": ["itemId": "rs_prev", "encryptedContent": "gAAAprev"]
                        ]
                    )
                ),
                .text("Earlier answer."),
            ]),
            .user("And?"),
        ]
        _ = try await provider(transport).languageModel("gpt-5").generate(options(prompt))

        let input = try #require(try transport.recordedRequestBody()["input"]?.arrayValue)
        let reasoningItem = try #require(input.first { $0["type"]?.stringValue == "reasoning" })
        #expect(reasoningItem["id"]?.stringValue == "rs_prev")
        #expect(reasoningItem["encrypted_content"]?.stringValue == "gAAAprev")
    }

    @Test("Decodes a function call")
    func decodesFunctionCall() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response-tool-call.json")))
        let response = try await provider(transport).languageModel("gpt-5").generate(options())

        let call = try #require(response.content.toolCalls.first)
        // The call identifier, not the item identifier, is what correlates the result.
        #expect(call.toolCallID == "call_KpQ2mR7x")
        #expect(call.toolName == "weather")
        #expect(call.input == ["city": "Malmö"])
        #expect(response.finishReason == .toolCalls)
    }

    @Test("Reports a truncated response as a token limit")
    func reportsTruncationAsLength() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response-incomplete.json")))
        let response = try await provider(transport).languageModel("gpt-5").generate(options())

        #expect(response.finishReason == .length)
        #expect(response.content.text == "Blue light scat")
    }

    // MARK: - Request construction

    @Test("Sends credentials and targets the responses endpoint")
    func sendsCredentials() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        _ = try await provider(transport).languageModel("gpt-5").generate(options())

        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["Authorization"] == "Bearer test-key")
        #expect(request.url.absoluteString == "https://api.openai.com/v1/responses")
    }

    @Test("Sends organization and project headers when configured")
    func sendsOrganizationHeaders() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let provider = OpenAIProvider(
            apiKey: "k",
            organization: "org-123",
            project: "proj-456",
            transport: transport
        )
        _ = try await provider.languageModel("gpt-5").generate(options())

        let headers = try #require(transport.recordedRequests.first?.headers)
        #expect(headers["OpenAI-Organization"] == "org-123")
        #expect(headers["OpenAI-Project"] == "proj-456")
    }

    @Test("Sends the system message as instructions")
    func sendsInstructions() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        _ = try await provider(transport).languageModel("gpt-5")
            .generate(options([.system("Be terse."), .user("Hi")]))

        let body = try transport.recordedRequestBody()
        #expect(body["instructions"]?.stringValue == "Be terse.")
        #expect(body["input"]?.arrayValue?.count == 1)
    }

    @Test("Flattens tool calls and outputs into separate input items")
    func flattensToolItems() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let prompt: [ModelMessage] = [
            .user("Weather?"),
            .assistant([
                .text("Checking."),
                .toolCall(ToolCallPart(toolCallID: "call_1", toolName: "weather", input: ["city": "Malmö"])),
            ]),
            .tool([ToolResultPart(toolCallID: "call_1", toolName: "weather", output: .text("Rain."))]),
        ]
        _ = try await provider(transport).languageModel("gpt-5").generate(options(prompt))

        let input = try #require(try transport.recordedRequestBody()["input"]?.arrayValue)
        // A user turn, a function call item, an assistant message, then the output item.
        #expect(input.contains { $0["type"]?.stringValue == "function_call" })
        let output = try #require(input.first { $0["type"]?.stringValue == "function_call_output" })
        #expect(output["call_id"]?.stringValue == "call_1")
        #expect(output["output"]?.stringValue == "Rain.")
    }

    @Test("Sends the token limit under its Responses name")
    func sendsMaxOutputTokens() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        _ = try await provider(transport).languageModel("gpt-5").generate(options(maxOutputTokens: 300))
        #expect(try transport.recordedRequestBody()["max_output_tokens"]?.intValue == 300)
    }

    @Test("Warns about settings only Chat Completions supports")
    func warnsAboutChatOnlySettings() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let response = try await provider(transport).languageModel("gpt-5").generate(options(seed: 7))

        #expect(response.warnings.contains { $0.description.contains("seed") })
        // The advice names the alternative rather than just refusing.
        #expect(response.warnings.contains { $0.description.contains("chat model") })
    }

    @Test("Sends a strict schema in the text format")
    func sendsStrictSchema() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let schema = JSONSchema.object(properties: ["name": .string()], required: ["name"])

        _ = try await provider(transport).languageModel("gpt-5")
            .generate(options(responseFormat: .json(schema: schema, name: "person", description: nil)))

        let format = try #require(try transport.recordedRequestBody()["text"]?["format"])
        #expect(format["type"]?.stringValue == "json_schema")
        #expect(format["name"]?.stringValue == "person")
        #expect(format["strict"]?.boolValue == true)
    }

    @Test("Declares provider-executed tools by type")
    func declaresProviderTools() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let tool = LanguageModelTool.providerDefined(
            OpenAIProvider.Tools.webSearch(searchContextSize: "high")
        )
        _ = try await provider(transport).languageModel("gpt-5").generate(options(tools: [tool]))

        let declared = try #require(try transport.recordedRequestBody()["tools"]?[0])
        #expect(declared["type"]?.stringValue == "web_search")
        #expect(declared["search_context_size"]?.stringValue == "high")
    }

    @Test("Ignores another provider's tools with a warning")
    func ignoresForeignProviderTools() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let tool = LanguageModelTool.providerDefined(
            ProviderDefinedTool(id: "anthropic.computer", name: "computer")
        )
        let response = try await provider(transport).languageModel("gpt-5").generate(options(tools: [tool]))

        #expect(response.warnings.contains { $0.description.contains("another provider") })
        #expect(try transport.recordedRequestBody()["tools"] == nil)
    }

    @Test("Passes provider options through")
    func passesProviderOptions() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("response.json")))
        let options = options(
            providerOptions: ["openai": ["reasoning": ["effort": "high", "summary": "auto"]]]
        )
        _ = try await provider(transport).languageModel("gpt-5").generate(options)

        let reasoning = try #require(try transport.recordedRequestBody()["reasoning"])
        #expect(reasoning["effort"]?.stringValue == "high")
    }

    // MARK: - Streaming

    @Test("Streams text", arguments: [1, 17, 8192])
    func streamsText(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("response-stream.sse"), chunkSize: chunkSize)
        )
        let parts = try await provider(transport).languageModel("gpt-5").stream(options()).stream.collect()

        #expect(parts.compactMap(\.textDelta).joined() == "Blue light scatters.")
        #expect(parts.contains { $0.kind == .textStart })
        #expect(parts.contains { $0.kind == .textEnd })

        guard case .finish(let reason, let usage, _)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .stop)
        #expect(usage.totalTokens == 14)
    }

    @Test("Streams function call arguments", arguments: [1, 23, 4096])
    func streamsFunctionCallArguments(chunkSize: Int) async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("response-stream-tools.sse"), chunkSize: chunkSize)
        )
        let parts = try await provider(transport).languageModel("gpt-5").stream(options()).stream.collect()

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
        // Deltas are keyed by item identifier on the wire but must surface under the call
        // identifier, which is what the result has to reference.
        #expect(call.toolCallID == "call_KpQ2mR7x")
        #expect(call.toolName == "weather")

        guard case .finish(let reason, _, _)? = parts.last else {
            Issue.record("Expected a finish part.")
            return
        }
        #expect(reason == .toolCalls)
    }

    @Test("Streams reasoning summaries before the answer")
    func streamsReasoningSummaries() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents(try fixture("response-stream-reasoning.sse"), chunkSize: 31)
        )
        let parts = try await provider(transport).languageModel("gpt-5").stream(options()).stream.collect()
        let kinds = parts.map(\.kind)

        #expect(parts.compactMap(\.reasoningDelta).joined() == "Shorter wavelengths scatter more.")
        #expect(parts.compactMap(\.textDelta).joined() == "Rayleigh scattering.")

        let reasoningEnd = try #require(kinds.firstIndex(of: .reasoningEnd))
        let textStart = try #require(kinds.firstIndex(of: .textStart))
        #expect(reasoningEnd < textStart)
    }

    // MARK: - Chat model

    @Test("The chat model targets Chat Completions")
    func chatModelUsesChatCompletions() async throws {
        let payload = """
            {"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o",
             "choices":[{"index":0,"message":{"role":"assistant","content":"Hi."},"finish_reason":"stop"}],
             "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
            """
        let transport = MockHTTPTransport(exchange: .json(payload))
        let response = try await provider(transport).chatModel("gpt-4o").generate(options())

        #expect(response.content.text == "Hi.")
        #expect(transport.recordedRequests.first?.url.absoluteString.hasSuffix("/chat/completions") == true)
        // The chat endpoint takes settings the Responses API rejects.
        #expect(try transport.recordedRequestBody()["max_completion_tokens"] == nil)
    }
}

@Suite("OpenAI media")
struct OpenAIMediaTests {
    private func fixture(_ name: String) throws -> String {
        try Fixtures.text(name, bundle: Bundle.module)
    }

    private func provider(_ transport: MockHTTPTransport) -> OpenAIProvider {
        OpenAIProvider(apiKey: "test-key", transport: transport)
    }

    @Test("Generates images and decodes the base64 payload")
    func generatesImages() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("image.json")))
        let response = try await provider(transport).imageModel("gpt-image-1")
            .generate(ImageModelCallOptions(prompt: "A cat.", count: 1, size: .square1024))

        #expect(response.images.count == 1)
        #expect(String(decoding: response.images[0].data, as: UTF8.self) == "hello")

        let body = try transport.recordedRequestBody()
        #expect(body["size"]?.stringValue == "1024x1024")
        // `gpt-image-1` always returns base64, so the field is not sent.
        #expect(body["response_format"] == nil)
    }

    @Test("Requests base64 explicitly from the DALL·E models")
    func requestsBase64FromDallE() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("image.json")))
        _ = try await provider(transport).imageModel("dall-e-3")
            .generate(ImageModelCallOptions(prompt: "A cat."))

        // Otherwise the API returns a URL that expires within the hour.
        #expect(try transport.recordedRequestBody()["response_format"]?.stringValue == "b64_json")
    }

    @Test("Warns about image settings OpenAI cannot honor")
    func warnsAboutImageSettings() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("image.json")))
        let response = try await provider(transport).imageModel("gpt-image-1")
            .generate(ImageModelCallOptions(prompt: "A cat.", aspectRatio: .landscape, seed: 7))

        #expect(response.warnings.contains { $0.description.contains("aspectRatio") })
        #expect(response.warnings.contains { $0.description.contains("seed") })
    }

    @Test("Synthesizes speech and reports the media type")
    func synthesizesSpeech() async throws {
        let transport = MockHTTPTransport(
            exchange: MockHTTPTransport.Exchange(chunks: [Data("audio-bytes".utf8)])
        )
        let response = try await provider(transport).speechModel("gpt-4o-mini-tts")
            .generate(SpeechModelCallOptions(text: "Hello.", voice: "alloy"))

        #expect(String(decoding: response.audio.data, as: UTF8.self) == "audio-bytes")
        #expect(response.audio.mediaType == "audio/mpeg")

        let body = try transport.recordedRequestBody()
        #expect(body["voice"]?.stringValue == "alloy")
        #expect(body["response_format"]?.stringValue == "mp3")
    }

    @Test("Maps output formats to media types", arguments: [
        ("mp3", "audio/mpeg"), ("wav", "audio/wav"), ("opus", "audio/opus"), ("flac", "audio/flac"),
    ])
    func mapsSpeechFormats(format: String, mediaType: String) async throws {
        let transport = MockHTTPTransport(
            exchange: MockHTTPTransport.Exchange(chunks: [Data("x".utf8)])
        )
        let response = try await provider(transport).speechModel("tts-1")
            .generate(SpeechModelCallOptions(text: "Hi.", outputFormat: format))
        #expect(response.audio.mediaType == mediaType)
    }

    @Test("Transcribes audio with timed segments")
    func transcribesWithSegments() async throws {
        let transport = MockHTTPTransport(exchange: .json(try fixture("transcription.json")))
        let response = try await provider(transport).transcriptionModel("whisper-1")
            .transcribe(
                TranscriptionModelCallOptions(
                    audio: Data("audio".utf8),
                    mediaType: "audio/mpeg",
                    filename: "clip.mp3"
                )
            )

        #expect(response.text == "Hello from Malmö.")
        #expect(response.segments.count == 1)
        #expect(response.segments[0].endSecond == 2.48)
        #expect(response.durationInSeconds == 2.48)

        // The upload is multipart, and the filename matters: several models infer the container
        // format from its extension.
        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["Content-Type"]?.hasPrefix("multipart/form-data") == true)
        let body = String(decoding: request.body ?? Data(), as: UTF8.self)
        #expect(body.contains("filename=\"clip.mp3\""))
        #expect(body.contains("verbose_json"))
    }

    @Test("Warns that newer transcription models return no segments")
    func warnsAboutMissingSegments() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"text":"Hello."}"#))
        let response = try await provider(transport).transcriptionModel("gpt-4o-transcribe")
            .transcribe(TranscriptionModelCallOptions(audio: Data("a".utf8), mediaType: "audio/mpeg"))

        #expect(response.text == "Hello.")
        #expect(response.segments.isEmpty)
        #expect(response.warnings.contains { $0.description.contains("timed segments") })

        // Those models reject `verbose_json` outright.
        let body = String(decoding: transport.recordedRequests.first?.body ?? Data(), as: UTF8.self)
        #expect(!body.contains("verbose_json"))
    }
}

@Suite("OpenAI evaluation")
struct OpenAIEvaluationTests {
    @Test("Turns reasoning down as far as each model family allows", arguments: [
        ("gpt-5.1", "none"),
        ("gpt-5.6-luna", "none"),
        ("gpt-6-luna", "none"),
        ("gpt-5", "minimal"),
        ("gpt-5-mini", "minimal"),
        ("o4-mini", "low"),
    ])
    func minimalEffort(modelID: String, effort: String) {
        #expect(OpenAIProvider.minimalReasoningEffort(for: modelID) == effort)
    }

    @Test("Sends no reasoning setting to models that reject one", arguments: ["gpt-4.1", "gpt-4o-mini", "gpt-5-chat-latest"])
    func noEffortForNonReasoningModels(modelID: String) {
        #expect(OpenAIProvider.minimalReasoningEffort(for: modelID) == nil)
    }
}

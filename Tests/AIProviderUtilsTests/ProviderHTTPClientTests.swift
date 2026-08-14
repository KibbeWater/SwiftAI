import AIProviderSpec
import AITestSupport
import Foundation
import Testing

@testable import AIProviderUtils

@Suite("ProviderHTTPClient")
struct ProviderHTTPClientTests {
    private struct EchoRequest: Encodable {
        var model: String
        var stream: Bool
    }

    private struct EchoResponse: Decodable, Equatable {
        var id: String
        var value: Int
    }

    private func makeClient(
        transport: any HTTPTransport,
        baseURL: String = "https://api.example.com/v1"
    ) -> ProviderHTTPClient {
        ProviderHTTPClient(
            baseURL: URL(string: baseURL)!,
            provider: "example",
            headers: { ["Authorization": "Bearer secret"] },
            transport: transport
        )
    }

    // MARK: - URL construction

    @Test(
        "Joins base URLs and paths without duplicating or dropping separators",
        arguments: [
            ("https://api.example.com/v1", "chat/completions", "https://api.example.com/v1/chat/completions"),
            ("https://api.example.com/v1/", "chat/completions", "https://api.example.com/v1/chat/completions"),
            ("https://api.example.com/v1", "/chat/completions", "https://api.example.com/v1/chat/completions"),
            ("https://api.example.com/v1/", "/chat/completions", "https://api.example.com/v1/chat/completions"),
            ("https://api.example.com", "models", "https://api.example.com/models"),
        ]
    )
    func buildsURLs(baseURL: String, path: String, expected: String) throws {
        let client = makeClient(transport: MockHTTPTransport(exchanges: []), baseURL: baseURL)
        #expect(try client.url(path: path).absoluteString == expected)
    }

    @Test("Appends query items")
    func appendsQueryItems() throws {
        let client = makeClient(transport: MockHTTPTransport(exchanges: []))
        let url = try client.url(path: "models/gemini:generateContent", queryItems: [
            URLQueryItem(name: "alt", value: "sse")
        ])
        #expect(url.absoluteString.hasSuffix("?alt=sse"))
    }

    // MARK: - Requests

    @Test("Sends credentials and content type, and decodes the response")
    func sendsAndDecodes() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"id":"abc","value":7}"#))
        let client = makeClient(transport: transport)

        let (value, head, requestBody) = try await client.postJSON(
            path: "chat/completions",
            body: EchoRequest(model: "m", stream: false),
            as: EchoResponse.self
        )

        #expect(value == EchoResponse(id: "abc", value: 7))
        #expect(head.statusCode == 200)
        #expect(requestBody == #"{"model":"m","stream":false}"#)

        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["Authorization"] == "Bearer secret")
        #expect(request.headers["Content-Type"] == "application/json")
        #expect(request.url.absoluteString == "https://api.example.com/v1/chat/completions")
    }

    @Test("Additional headers override the client defaults")
    func additionalHeadersOverrideDefaults() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"id":"a","value":1}"#))
        let client = makeClient(transport: transport)

        _ = try await client.postJSON(
            path: "x",
            body: EchoRequest(model: "m", stream: false),
            additionalHeaders: ["Authorization": "Bearer override", "X-Extra": "1"],
            as: EchoResponse.self
        )

        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["Authorization"] == "Bearer override")
        #expect(request.headers["X-Extra"] == "1")
    }

    @Test("Request bodies encode with sorted keys so they are reproducible")
    func requestBodiesAreDeterministic() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"id":"a","value":1}"#))
        let client = makeClient(transport: transport)

        let (_, _, body) = try await client.postJSON(
            path: "x",
            body: EchoRequest(model: "zeta", stream: true),
            as: EchoResponse.self
        )
        #expect(body == #"{"model":"zeta","stream":true}"#)
    }

    // MARK: - Error mapping

    @Test("Turns a provider error body into a readable APICallError")
    func mapsProviderErrorBody() async throws {
        let transport = MockHTTPTransport(
            exchange: .failure(
                statusCode: 429,
                body: #"{"error":{"message":"Rate limit reached","type":"rate_limit_error"}}"#,
                headers: ["retry-after": "3"]
            )
        )
        let client = makeClient(transport: transport)

        var caught: APICallError?
        do {
            _ = try await client.postJSON(
                path: "x",
                body: EchoRequest(model: "m", stream: false),
                as: EchoResponse.self
            )
        } catch let error as APICallError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.statusCode == 429)
        #expect(error.message.contains("Rate limit reached"))
        #expect(error.message.contains("example"))
        #expect(error.isRetryable)
        #expect(error.retryAfterDelay == .seconds(3))
        // The parsed body is preserved for callers who need provider-specific fields.
        #expect(error.data?["error"]?["type"]?.stringValue == "rate_limit_error")
    }

    @Test("Never records the request body on an error, since it may contain user data")
    func errorsOmitRequestBody() async throws {
        let transport = MockHTTPTransport(exchange: .failure(statusCode: 400, body: #"{"error":"bad"}"#))
        let client = makeClient(transport: transport)

        var caught: APICallError?
        do {
            _ = try await client.postJSON(path: "x", body: EchoRequest(model: "m", stream: false), as: EchoResponse.self)
        } catch let error as APICallError {
            caught = error
        }
        #expect(try #require(caught).requestBody == nil)
    }

    @Test(
        "Recognizes the error body shapes providers actually use",
        arguments: [
            (#"{"error":{"message":"nested"}}"#, "nested"),
            (#"{"error":"flat"}"#, "flat"),
            (#"{"message":"top level"}"#, "top level"),
            (#"{"detail":"fastapi"}"#, "fastapi"),
            (#"{"detail":[{"msg":"validation"}]}"#, "validation"),
        ]
    )
    func recognizesErrorShapes(body: String, expectedMessage: String) throws {
        let parsed = try JSONValue.parse(body)
        #expect(ProviderErrorMapper.standard.extractMessage(parsed) == expectedMessage)
    }

    @Test("Falls back to the raw body when the error shape is unrecognized")
    func fallsBackToRawBody() async throws {
        let transport = MockHTTPTransport(exchange: .failure(statusCode: 500, body: "Internal Server Error"))
        let client = makeClient(transport: transport)

        var caught: APICallError?
        do {
            _ = try await client.postJSON(path: "x", body: EchoRequest(model: "m", stream: false), as: EchoResponse.self)
        } catch let error as APICallError {
            caught = error
        }
        #expect(try #require(caught).message.contains("Internal Server Error"))
    }

    @Test("Reports a decoding failure with the offending key path")
    func reportsDecodingFailures() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"id":"abc"}"#))
        let client = makeClient(transport: transport)

        var caught: TypeValidationError?
        do {
            _ = try await client.postJSON(path: "x", body: EchoRequest(model: "m", stream: false), as: EchoResponse.self)
        } catch let error as TypeValidationError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.path == "value")
        #expect(error.message.contains("value"))
    }

    // MARK: - Streaming

    @Test("Streams server-sent events")
    func streamsEvents() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents("data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n", chunkSize: 3)
        )
        let client = makeClient(transport: transport)

        let (events, head, _) = try await client.postJSONForServerSentEvents(
            path: "chat/completions",
            body: EchoRequest(model: "m", stream: true)
        )

        #expect(head.statusCode == 200)
        let collected = try await events.collect()
        #expect(collected.map(\.data) == [#"{"a":1}"#, #"{"a":2}"#])

        let request = try #require(transport.recordedRequests.first)
        #expect(request.headers["Accept"] == "text/event-stream")
    }

    @Test("Reads the error body before failing a streaming request")
    func streamingFailureReadsErrorBody() async throws {
        let transport = MockHTTPTransport(
            exchange: .failure(statusCode: 401, body: #"{"error":{"message":"Invalid API key"}}"#)
        )
        let client = makeClient(transport: transport)

        var caught: APICallError?
        do {
            _ = try await client.postJSONForServerSentEvents(
                path: "x",
                body: EchoRequest(model: "m", stream: true)
            )
        } catch let error as APICallError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.statusCode == 401)
        // A status code alone would not tell the caller their key is wrong.
        #expect(error.message.contains("Invalid API key"))
        #expect(error.isRetryable == false)
    }

    // MARK: - Credentials

    @Test("Prefers an explicitly supplied API key")
    func prefersExplicitAPIKey() throws {
        let key = try ProviderHTTPClient.resolveAPIKey(
            explicit: "explicit",
            environmentVariable: "SWIFTAI_TEST_KEY_UNSET",
            provider: "example"
        )
        #expect(key == "explicit")
    }

    @Test("Names the environment variable when no key is available")
    func missingKeyErrorIsActionable() throws {
        var caught: MissingAPIKeyError?
        do {
            _ = try ProviderHTTPClient.resolveAPIKey(
                explicit: nil,
                environmentVariable: "SWIFTAI_TEST_KEY_UNSET",
                provider: "example"
            )
        } catch let error as MissingAPIKeyError {
            caught = error
        }
        #expect(try #require(caught).message.contains("SWIFTAI_TEST_KEY_UNSET"))
    }

    @Test("Treats an empty key as absent")
    func emptyKeyIsTreatedAsAbsent() {
        #expect(throws: MissingAPIKeyError.self) {
            _ = try ProviderHTTPClient.resolveAPIKey(
                explicit: "",
                environmentVariable: "SWIFTAI_TEST_KEY_UNSET",
                provider: "example"
            )
        }
    }
}

@Suite("MockHTTPTransport")
struct MockHTTPTransportTests {
    @Test("Fails loudly when a test makes more requests than it scripted")
    func failsWhenExhausted() async throws {
        let transport = MockHTTPTransport(exchanges: [.json("{}")])
        _ = try await transport.send(HTTPRequest(url: URL(string: "https://example.com")!))

        await #expect(throws: MockTransportError.self) {
            _ = try await transport.send(HTTPRequest(url: URL(string: "https://example.com")!))
        }
    }

    @Test("Repeats the final exchange when configured to")
    func repeatsLastExchange() async throws {
        let transport = MockHTTPTransport(exchange: .json(#"{"ok":true}"#))
        for _ in 0..<3 {
            _ = try await transport.send(HTTPRequest(url: URL(string: "https://example.com")!))
        }
        #expect(transport.requestCount == 3)
    }

    @Test("Splits a body into chunks of the requested size")
    func splitsBodyIntoChunks() async throws {
        let transport = MockHTTPTransport(exchange: .serverSentEvents("data: hello\n\n", chunkSize: 4))
        let response = try await transport.stream(HTTPRequest(url: URL(string: "https://example.com")!))

        var chunkCount = 0
        var body = Data()
        for try await chunk in response.body {
            chunkCount += 1
            body.append(chunk)
        }
        #expect(chunkCount == 4)
        #expect(String(decoding: body, as: UTF8.self) == "data: hello\n\n")
    }
}

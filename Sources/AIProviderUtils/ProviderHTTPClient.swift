import AIProviderSpec
import Foundation

/// Turns a provider's error response body into a human-readable message.
///
/// Providers wrap their errors differently — OpenAI uses `{"error": {"message": …}}`, Google uses
/// `{"error": {"message": …, "status": …}}`, Anthropic uses `{"error": {"message": …}}`, and
/// self-hosted OpenAI-compatible servers use whatever they like. The default handles all of the
/// common shapes, and a provider with an unusual one supplies its own.
public struct ProviderErrorMapper: Sendable {
    /// Extracts a message from a parsed error body, or returns `nil` to fall back to the raw body.
    public var extractMessage: @Sendable (JSONValue) -> String?

    public init(extractMessage: @escaping @Sendable (JSONValue) -> String?) {
        self.extractMessage = extractMessage
    }

    /// Recognizes the shapes used by every provider this package ships.
    public static let standard = ProviderErrorMapper { body in
        // `{"error": {"message": "…"}}` — OpenAI, Anthropic, Google, most compatible servers.
        if let message = body["error"]?["message"]?.stringValue { return message }
        // `{"error": "…"}` — several self-hosted servers.
        if let message = body["error"]?.stringValue { return message }
        // `{"message": "…"}` — gateways and proxies.
        if let message = body["message"]?.stringValue { return message }
        // `{"detail": "…"}` — FastAPI-based servers, which vLLM and others use.
        if let message = body["detail"]?.stringValue { return message }
        if let message = body["detail"]?[0]?["msg"]?.stringValue { return message }
        return nil
    }
}

/// The HTTP plumbing shared by every provider.
///
/// A provider composes one of these with its base URL and credentials, then expresses each
/// endpoint as a single call. Everything cross-cutting — building URLs, merging headers, parsing
/// errors into ``APICallError``, decoding server-sent events — happens here once.
///
/// ```swift
/// let (response, head) = try await client.postJSON(
///     path: "chat/completions",
///     body: requestBody,
///     as: ChatCompletionResponse.self
/// )
/// ```
public struct ProviderHTTPClient: Sendable {
    /// The API root. Paths are appended to this, so it should end with a trailing slash
    /// semantically even though this type normalizes either form.
    public var baseURL: URL

    /// Produces the headers sent with every request.
    ///
    /// A closure rather than a stored dictionary so that credentials can be read lazily — from an
    /// environment variable, a keychain, or a token that refreshes.
    public var headers: @Sendable () async throws -> [String: String]

    /// The transport that performs requests. Substitute this in tests.
    public var transport: any HTTPTransport

    /// The provider's short name, used in error messages.
    public var provider: String

    /// How to read the provider's error responses.
    public var errorMapper: ProviderErrorMapper

    /// The decoder used for response bodies.
    ///
    /// Providers whose APIs use snake case pass ``ProviderJSON/snakeCaseDecoder`` so their wire
    /// types can be written with ordinary Swift property names.
    public var decoder: JSONDecoder

    /// Whether to retain response bodies on ``ResponseInfo``.
    ///
    /// Off by default: bodies can be large, and holding them for the lifetime of a result is
    /// rarely what a caller wants. Turn it on when debugging.
    public var retainsResponseBodies: Bool

    public init(
        baseURL: URL,
        provider: String,
        headers: @escaping @Sendable () async throws -> [String: String],
        transport: any HTTPTransport = URLSessionTransport.shared,
        errorMapper: ProviderErrorMapper = .standard,
        decoder: JSONDecoder = ProviderJSON.decoder,
        retainsResponseBodies: Bool = false
    ) {
        self.baseURL = baseURL
        self.provider = provider
        self.headers = headers
        self.transport = transport
        self.errorMapper = errorMapper
        self.decoder = decoder
        self.retainsResponseBodies = retainsResponseBodies
    }

    // MARK: - URL construction

    /// Builds a request URL by appending a path and optional query items to the base URL.
    public func url(path: String, queryItems: [URLQueryItem] = []) throws -> URL {
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        let base = baseURL.absoluteString.hasSuffix("/") ? baseURL : baseURL.appendingPathComponent("")
        guard var components = URLComponents(
            url: URL(string: trimmed, relativeTo: base) ?? base,
            resolvingAgainstBaseURL: true
        ) else {
            throw InvalidArgumentError(
                argument: "baseURL",
                message: "'\(baseURL.absoluteString)' and path '\(path)' do not form a valid URL."
            )
        }
        if !queryItems.isEmpty {
            components.queryItems = (components.queryItems ?? []) + queryItems
        }
        guard let url = components.url else {
            throw InvalidArgumentError(
                argument: "baseURL",
                message: "'\(baseURL.absoluteString)' and path '\(path)' do not form a valid URL."
            )
        }
        return url
    }

    // MARK: - Buffered requests

    /// Sends a JSON request and decodes a JSON response.
    ///
    /// - Parameters:
    ///   - path: The endpoint path, relative to ``baseURL``.
    ///   - body: The request payload.
    ///   - queryItems: Query parameters to append.
    ///   - additionalHeaders: Headers merged over the client's defaults.
    ///   - responseType: The shape to decode.
    /// - Returns: The decoded response, the response head, and the serialized request body for
    ///   inclusion in ``RequestInfo``.
    /// - Throws: ``APICallError`` for non-2xx responses, ``TypeValidationError`` when the payload
    ///   does not match `responseType`.
    public func postJSON<Response: Decodable>(
        path: String,
        body: some Encodable,
        queryItems: [URLQueryItem] = [],
        additionalHeaders: [String: String] = [:],
        as responseType: Response.Type
    ) async throws -> (value: Response, head: HTTPResponseHead, requestBody: String) {
        let bodyData = try ProviderJSON.encode(body)
        let request = try await makeRequest(
            path: path,
            queryItems: queryItems,
            bodyData: bodyData,
            contentType: "application/json",
            additionalHeaders: additionalHeaders
        )

        let response = try await performBuffered(request)
        let value = try ProviderJSON.decode(
            responseType,
            from: response.body,
            context: "the response from \(provider)",
            decoder: decoder
        )
        return (value, response.head, String(decoding: bodyData, as: UTF8.self))
    }

    /// Sends a request with a pre-built body — a multipart upload, for instance — and decodes a
    /// JSON response.
    public func post<Response: Decodable>(
        path: String,
        bodyData: Data,
        contentType: String,
        queryItems: [URLQueryItem] = [],
        additionalHeaders: [String: String] = [:],
        as responseType: Response.Type
    ) async throws -> (value: Response, head: HTTPResponseHead) {
        let request = try await makeRequest(
            path: path,
            queryItems: queryItems,
            bodyData: bodyData,
            contentType: contentType,
            additionalHeaders: additionalHeaders
        )
        let response = try await performBuffered(request)
        let value = try ProviderJSON.decode(
            responseType,
            from: response.body,
            context: "the response from \(provider)",
            decoder: decoder
        )
        return (value, response.head)
    }

    /// Sends a request and returns the raw response body, for endpoints that return binary data
    /// such as synthesized speech.
    public func postForData(
        path: String,
        body: some Encodable,
        queryItems: [URLQueryItem] = [],
        additionalHeaders: [String: String] = [:]
    ) async throws -> (data: Data, head: HTTPResponseHead, requestBody: String) {
        let bodyData = try ProviderJSON.encode(body)
        let request = try await makeRequest(
            path: path,
            queryItems: queryItems,
            bodyData: bodyData,
            contentType: "application/json",
            additionalHeaders: additionalHeaders
        )
        let response = try await performBuffered(request)
        return (response.body, response.head, String(decoding: bodyData, as: UTF8.self))
    }

    // MARK: - Streaming requests

    /// Sends a JSON request and returns the response as a stream of server-sent events.
    ///
    /// - Parameters:
    ///   - path: The endpoint path, relative to ``baseURL``.
    ///   - body: The request payload.
    ///   - queryItems: Query parameters to append.
    ///   - additionalHeaders: Headers merged over the client's defaults.
    ///   - stopAtDoneSentinel: Whether a `[DONE]` payload ends the stream. OpenAI-style APIs send
    ///     one; Anthropic and Google do not.
    /// - Returns: The event stream, the response head, and the serialized request body.
    /// - Throws: ``APICallError`` if the response status is not 2xx. The body is drained first so
    ///   the error carries the provider's explanation.
    public func postJSONForServerSentEvents(
        path: String,
        body: some Encodable,
        queryItems: [URLQueryItem] = [],
        additionalHeaders: [String: String] = [:],
        stopAtDoneSentinel: Bool = true
    ) async throws -> (
        events: AsyncThrowingStream<ServerSentEvent, any Error>,
        head: HTTPResponseHead,
        requestBody: String
    ) {
        let bodyData = try ProviderJSON.encode(body)
        var request = try await makeRequest(
            path: path,
            queryItems: queryItems,
            bodyData: bodyData,
            contentType: "application/json",
            additionalHeaders: additionalHeaders
        )
        request.headers["Accept"] = "text/event-stream"

        let response: HTTPStreamResponse
        do {
            response = try await transport.stream(request)
        } catch {
            throw URLSessionTransport.mapTransportError(error, request: request)
        }

        guard response.head.isSuccess else {
            // Read the error body before failing, so the caller learns *why* rather than just
            // seeing a status code.
            var body = Data()
            for try await chunk in response.body { body.append(chunk) }
            throw makeAPICallError(request: request, head: response.head, body: body)
        }

        return (
            response.body.serverSentEvents(stopAtDoneSentinel: stopAtDoneSentinel),
            response.head,
            String(decoding: bodyData, as: UTF8.self)
        )
    }

    // MARK: - Plumbing

    private func makeRequest(
        path: String,
        queryItems: [URLQueryItem],
        bodyData: Data?,
        contentType: String?,
        additionalHeaders: [String: String]
    ) async throws -> HTTPRequest {
        var merged = try await headers()
        for (name, value) in additionalHeaders { merged[name] = value }
        if let contentType { merged["Content-Type"] = contentType }

        return HTTPRequest(
            url: try url(path: path, queryItems: queryItems),
            method: .post,
            headers: merged,
            body: bodyData
        )
    }

    private func performBuffered(_ request: HTTPRequest) async throws -> HTTPResponse {
        let response: HTTPResponse
        do {
            response = try await transport.send(request)
        } catch {
            throw URLSessionTransport.mapTransportError(error, request: request)
        }
        guard response.head.isSuccess else {
            throw makeAPICallError(request: request, head: response.head, body: response.body)
        }
        return response
    }

    /// Builds a rich error from a failed response.
    func makeAPICallError(request: HTTPRequest, head: HTTPResponseHead, body: Data) -> APICallError {
        let bodyText = String(data: body, encoding: .utf8)
        let parsed = try? JSONValue.parse(body)
        let message = parsed.flatMap(errorMapper.extractMessage)
            ?? bodyText.map { $0.isEmpty ? "The response body was empty." : String($0.prefix(1024)) }
            ?? "The response body was not valid UTF-8."

        return APICallError(
            message: "\(provider) returned \(head.statusCode): \(message)",
            url: request.url,
            method: request.method.rawValue,
            statusCode: head.statusCode,
            responseHeaders: head.headers,
            responseBody: bodyText,
            // The request body is deliberately omitted: it can contain user data, and credentials
            // live in headers which are never captured.
            requestBody: nil,
            data: parsed
        )
    }

    /// Builds the ``ResponseInfo`` attached to results.
    public func responseInfo(
        head: HTTPResponseHead,
        id: String? = nil,
        modelID: String? = nil,
        timestamp: Date? = nil,
        body: Data? = nil
    ) -> ResponseInfo {
        ResponseInfo(
            id: id,
            modelID: modelID,
            timestamp: timestamp,
            headers: head.headers,
            body: retainsResponseBodies ? body.flatMap { String(data: $0, encoding: .utf8) } : nil
        )
    }
}

// MARK: - Credential resolution

extension ProviderHTTPClient {
    /// Resolves an API key from an explicit value or an environment variable.
    ///
    /// - Parameters:
    ///   - explicit: The key passed to the provider's initializer, if any.
    ///   - environmentVariable: The variable to consult when no key was passed.
    ///   - provider: The provider name, used in the error message.
    /// - Returns: The resolved key.
    /// - Throws: ``MissingAPIKeyError`` naming the variable to set.
    public static func resolveAPIKey(
        explicit: String?,
        environmentVariable: String,
        provider: String
    ) throws -> String {
        if let explicit, !explicit.isEmpty { return explicit }
        if let value = ProcessInfo.processInfo.environment[environmentVariable], !value.isEmpty {
            return value
        }
        throw MissingAPIKeyError(provider: provider, environmentVariable: environmentVariable)
    }
}

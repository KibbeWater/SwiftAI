import Foundation

/// An HTTP method.
///
/// Modeled as a wrapper around a string rather than a closed enum so that providers needing an
/// uncommon verb are not blocked by this type.
public struct HTTPMethod: Sendable, Hashable, RawRepresentable, ExpressibleByStringLiteral {
    public var rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public static let get = HTTPMethod(rawValue: "GET")
    public static let post = HTTPMethod(rawValue: "POST")
    public static let put = HTTPMethod(rawValue: "PUT")
    public static let patch = HTTPMethod(rawValue: "PATCH")
    public static let delete = HTTPMethod(rawValue: "DELETE")
}

/// A request to send to a provider's API.
public struct HTTPRequest: Sendable {
    public var url: URL
    public var method: HTTPMethod
    public var headers: [String: String]
    public var body: Data?

    /// How long to wait for the response to begin, in seconds.
    ///
    /// Applies to establishing the response, not to the total time a stream stays open — a long
    /// generation must not be cut short by a request timeout.
    public var timeout: TimeInterval?

    public init(
        url: URL,
        method: HTTPMethod = .post,
        headers: [String: String] = [:],
        body: Data? = nil,
        timeout: TimeInterval? = nil
    ) {
        self.url = url
        self.method = method
        self.headers = headers
        self.body = body
        self.timeout = timeout
    }

    /// Returns a copy with `additionalHeaders` merged in. Existing values win, so a caller's
    /// explicit header is never silently replaced by a provider default.
    public func addingHeaders(_ additionalHeaders: [String: String]) -> HTTPRequest {
        var copy = self
        copy.headers = additionalHeaders.merging(headers) { _, existing in existing }
        return copy
    }
}

/// The status line and headers of a response, available before the body has been read.
public struct HTTPResponseHead: Sendable, Hashable {
    public var statusCode: Int

    /// Response headers, with lowercased names.
    ///
    /// HTTP header names are case-insensitive but the casing servers use varies, so they are
    /// normalized here and providers can look them up without guessing.
    public var headers: [String: String]

    public init(statusCode: Int, headers: [String: String] = [:]) {
        self.statusCode = statusCode
        self.headers = headers.reduce(into: [:]) { $0[$1.key.lowercased()] = $1.value }
    }

    /// Whether the status code is in the 2xx range.
    public var isSuccess: Bool { (200..<300).contains(statusCode) }

    /// Looks up a header, ignoring case.
    public func header(_ name: String) -> String? { headers[name.lowercased()] }
}

/// A complete response, with the body fully buffered.
public struct HTTPResponse: Sendable {
    public var head: HTTPResponseHead
    public var body: Data

    public init(head: HTTPResponseHead, body: Data) {
        self.head = head
        self.body = body
    }

    public var statusCode: Int { head.statusCode }
    public var headers: [String: String] { head.headers }

    /// The body decoded as UTF-8, for error reporting and debugging.
    public var bodyText: String? { String(data: body, encoding: .utf8) }
}

/// A response whose body is delivered incrementally.
public struct HTTPStreamResponse: Sendable {
    public var head: HTTPResponseHead

    /// The body, in the chunks the network delivered them.
    ///
    /// Chunk boundaries are arbitrary and carry no meaning: a single SSE event may straddle
    /// several chunks, and one chunk may contain many events. Consumers must buffer across
    /// boundaries, which ``EventSourceParser`` does.
    public var body: AsyncThrowingStream<Data, any Error>

    public init(head: HTTPResponseHead, body: AsyncThrowingStream<Data, any Error>) {
        self.head = head
        self.body = body
    }
}

/// Performs HTTP requests on behalf of providers.
///
/// Abstracting the network behind a protocol serves three purposes: it lets the SDK use a
/// delegate-driven `URLSession` on Linux where `URLSession.bytes` is unavailable, it lets
/// server-side deployments substitute a client with better connection pooling, and — most
/// usefully day to day — it lets the entire SDK be tested against recorded fixtures with no
/// network at all.
///
/// Implementations must propagate Swift's cooperative task cancellation to the underlying
/// request, so that cancelling a generation actually stops the upstream call.
public protocol HTTPTransport: Sendable {
    /// Performs a request and buffers the whole response.
    ///
    /// - Parameter request: The request to send.
    /// - Returns: The response, including non-2xx responses. Implementations report transport
    ///   failures by throwing, and leave status-code interpretation to the caller.
    /// - Throws: ``APICallError`` when the request cannot be completed.
    func send(_ request: HTTPRequest) async throws -> HTTPResponse

    /// Performs a request and delivers the response body incrementally.
    ///
    /// - Parameter request: The request to send.
    /// - Returns: The response head, available as soon as it arrives, and a stream of body chunks.
    /// - Throws: ``APICallError`` when the response head cannot be obtained. Failures after that
    ///   point terminate the body stream instead.
    func stream(_ request: HTTPRequest) async throws -> HTTPStreamResponse
}

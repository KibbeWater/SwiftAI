import AIProviderSpec
import AIProviderUtils
import Foundation

/// A scripted ``HTTPTransport`` for tests.
///
/// Responses are queued in advance and consumed in order, and every request is recorded for
/// inspection. Because the whole SDK reaches the network through `HTTPTransport`, substituting
/// this makes every layer above it — providers, the tool loop, streaming — testable end to end
/// with no network and no flakiness.
///
/// ```swift
/// let transport = MockHTTPTransport(exchanges: [
///     .serverSentEvents(fixture, chunkSize: 7)
/// ])
/// let model = OpenAIProvider(apiKey: "test", transport: transport).languageModel("gpt-5")
/// ```
public final class MockHTTPTransport: HTTPTransport, @unchecked Sendable {
    /// One scripted request/response round trip.
    public struct Exchange: Sendable {
        /// The status and headers to return.
        public var head: HTTPResponseHead

        /// The body, split into the chunks the transport will deliver.
        ///
        /// Splitting a fixture at deliberately awkward boundaries is the point: it is how a
        /// stream parser's buffering gets exercised.
        public var chunks: [Data]

        /// An error to raise instead of completing the body.
        public var error: (any Error)?

        /// Whether the error occurs before the response head, simulating a connection failure
        /// rather than a mid-stream failure.
        public var failsBeforeHead: Bool

        public init(
            head: HTTPResponseHead = HTTPResponseHead(statusCode: 200),
            chunks: [Data] = [],
            error: (any Error)? = nil,
            failsBeforeHead: Bool = false
        ) {
            self.head = head
            self.chunks = chunks
            self.error = error
            self.failsBeforeHead = failsBeforeHead
        }

        /// A successful JSON response.
        public static func json(
            _ text: String,
            statusCode: Int = 200,
            headers: [String: String] = [:]
        ) -> Exchange {
            Exchange(
                head: HTTPResponseHead(
                    statusCode: statusCode,
                    headers: headers.merging(["content-type": "application/json"]) { current, _ in current }
                ),
                chunks: [Data(text.utf8)]
            )
        }

        /// A successful JSON response loaded from `Data`.
        public static func json(
            _ data: Data,
            statusCode: Int = 200,
            headers: [String: String] = [:]
        ) -> Exchange {
            Exchange(
                head: HTTPResponseHead(
                    statusCode: statusCode,
                    headers: headers.merging(["content-type": "application/json"]) { current, _ in current }
                ),
                chunks: [data]
            )
        }

        /// A `text/event-stream` response, split into fixed-size chunks.
        ///
        /// - Parameters:
        ///   - text: The complete SSE body.
        ///   - chunkSize: How many bytes to deliver at a time. A small, non-round value splits
        ///     events across chunk boundaries, which is exactly the condition that breaks naive
        ///     parsers. Pass `nil` to deliver the body in one piece.
        ///   - headers: Extra response headers.
        public static func serverSentEvents(
            _ text: String,
            chunkSize: Int? = 13,
            statusCode: Int = 200,
            headers: [String: String] = [:]
        ) -> Exchange {
            let data = Data(text.utf8)
            let chunks: [Data]
            if let chunkSize, chunkSize > 0 {
                chunks = stride(from: 0, to: data.count, by: chunkSize).map { start in
                    data[start..<min(start + chunkSize, data.count)]
                }
            } else {
                chunks = [data]
            }
            return Exchange(
                head: HTTPResponseHead(
                    statusCode: statusCode,
                    headers: headers.merging(["content-type": "text/event-stream"]) { current, _ in current }
                ),
                chunks: chunks
            )
        }

        /// A failing response carrying a provider error body.
        public static func failure(
            statusCode: Int,
            body: String,
            headers: [String: String] = [:]
        ) -> Exchange {
            Exchange(
                head: HTTPResponseHead(statusCode: statusCode, headers: headers),
                chunks: [Data(body.utf8)]
            )
        }

        /// A transport-level failure, as though the connection never succeeded.
        public static func transportFailure(_ error: any Error) -> Exchange {
            Exchange(error: error, failsBeforeHead: true)
        }
    }

    private let lock = NSLock()
    private var queue: [Exchange]
    private var recorded: [HTTPRequest] = []

    /// Whether the final exchange repeats once the queue is exhausted.
    ///
    /// Off by default, so an unexpected extra request fails loudly instead of silently
    /// succeeding — a test that makes two calls when it meant to make one should not pass.
    public let repeatsLastExchange: Bool

    public init(exchanges: [Exchange], repeatsLastExchange: Bool = false) {
        self.queue = exchanges
        self.repeatsLastExchange = repeatsLastExchange
    }

    /// Creates a transport that answers every request with the same exchange.
    public convenience init(exchange: Exchange) {
        self.init(exchanges: [exchange], repeatsLastExchange: true)
    }

    /// Every request the transport has been asked to perform, in order.
    public var recordedRequests: [HTTPRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// The body of a recorded request, parsed as JSON.
    ///
    /// - Parameter index: Which request to inspect. Defaults to the first.
    public func recordedRequestBody(at index: Int = 0) throws -> JSONValue {
        let requests = recordedRequests
        guard requests.indices.contains(index), let body = requests[index].body else {
            throw MockTransportError.noRecordedRequest(index: index)
        }
        return try JSONValue.parse(body)
    }

    /// How many requests have been performed.
    public var requestCount: Int { recordedRequests.count }

    private func nextExchange(for request: HTTPRequest) throws -> Exchange {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(request)

        if queue.count > 1 || !repeatsLastExchange {
            guard !queue.isEmpty else {
                throw MockTransportError.exchangesExhausted(
                    requestCount: recorded.count,
                    url: request.url.absoluteString
                )
            }
            return queue.removeFirst()
        }
        guard let last = queue.first else {
            throw MockTransportError.exchangesExhausted(
                requestCount: recorded.count,
                url: request.url.absoluteString
            )
        }
        return last
    }

    public func send(_ request: HTTPRequest) async throws -> HTTPResponse {
        let exchange = try nextExchange(for: request)
        if let error = exchange.error { throw error }
        var body = Data()
        for chunk in exchange.chunks { body.append(chunk) }
        return HTTPResponse(head: exchange.head, body: body)
    }

    public func stream(_ request: HTTPRequest) async throws -> HTTPStreamResponse {
        let exchange = try nextExchange(for: request)
        if let error = exchange.error, exchange.failsBeforeHead { throw error }

        let (stream, continuation) = AsyncThrowingStream<Data, any Error>.makeStream()
        for chunk in exchange.chunks {
            continuation.yield(chunk)
        }
        if let error = exchange.error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
        return HTTPStreamResponse(head: exchange.head, body: stream)
    }
}

/// Failures raised by ``MockHTTPTransport`` itself, as opposed to scripted provider failures.
public enum MockTransportError: Error, CustomStringConvertible {
    /// More requests were made than the test scripted responses for.
    case exchangesExhausted(requestCount: Int, url: String)
    /// A request was inspected that was never made.
    case noRecordedRequest(index: Int)

    public var description: String {
        switch self {
        case .exchangesExhausted(let requestCount, let url):
            return """
                MockHTTPTransport ran out of scripted exchanges on request \(requestCount) to \(url). \
                Queue another exchange, or pass repeatsLastExchange: true.
                """
        case .noRecordedRequest(let index):
            return "MockHTTPTransport recorded no request at index \(index)."
        }
    }
}

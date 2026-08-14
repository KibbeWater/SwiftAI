import AIProviderSpec
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// The default ``HTTPTransport``, built on `URLSession`.
///
/// Streaming is driven by `URLSessionDataDelegate` callbacks rather than `URLSession.bytes`.
/// That is a deliberate choice: `URLSession.bytes` and `AsyncBytes` do not exist in
/// swift-corelibs-foundation, so a `bytes`-based implementation would need a second, separately
/// tested code path for Linux. The delegate API is available identically on every platform, so
/// one implementation serves all of them and server-side Swift is a first-class target rather
/// than an afterthought.
///
/// ## Buffering
///
/// Body chunks are buffered without bound between the network and the consumer. For language
/// model responses this is safe — the ceiling is one response — but this transport is not
/// suitable for downloading arbitrarily large payloads.
public final class URLSessionTransport: HTTPTransport, @unchecked Sendable {
    /// A transport backed by a default-configured session, suitable for most applications.
    public static let shared = URLSessionTransport()

    private let session: URLSession
    private let sessionDelegate: StreamingSessionDelegate

    /// Creates a transport with its own `URLSession`.
    ///
    /// A private session is required because the streaming implementation installs a delegate,
    /// which `URLSession.shared` does not allow.
    ///
    /// - Parameter configuration: The session configuration. The default disables the URL cache,
    ///   since model responses are not usefully cacheable and caching them wastes memory.
    public init(configuration: URLSessionConfiguration = URLSessionTransport.defaultConfiguration()) {
        let delegate = StreamingSessionDelegate()
        self.sessionDelegate = delegate
        self.session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
    }

    deinit {
        // Lets in-flight requests finish, then releases the session's reference to its delegate.
        session.finishTasksAndInvalidate()
    }

    /// The configuration used when none is supplied.
    public static func defaultConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.default
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        // Time to first byte. A generation that streams for minutes must not be cut off, so only
        // the idle timeout below governs an established stream.
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 3600
        return configuration
    }

    public func send(_ request: HTTPRequest) async throws -> HTTPResponse {
        let response = try await stream(request)
        var body = Data()
        for try await chunk in response.body {
            body.append(chunk)
        }
        return HTTPResponse(head: response.head, body: body)
    }

    public func stream(_ request: HTTPRequest) async throws -> HTTPStreamResponse {
        let task = session.dataTask(with: request.makeURLRequest())
        let identifier = task.taskIdentifier

        let (body, bodyContinuation) = AsyncThrowingStream<Data, any Error>.makeStream()
        // Cancelling the upstream request when the consumer stops reading is what makes
        // abandoning a stream actually stop the billing meter.
        bodyContinuation.onTermination = { _ in task.cancel() }
        sessionDelegate.register(identifier: identifier, bodyContinuation: bodyContinuation)

        do {
            let head = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    // Registering before `resume()` guarantees no callback can arrive first.
                    sessionDelegate.setHeadContinuation(continuation, for: identifier)
                    task.resume()
                }
            } onCancel: {
                task.cancel()
            }
            return HTTPStreamResponse(head: head, body: body)
        } catch {
            sessionDelegate.finish(identifier: identifier, error: error)
            throw URLSessionTransport.mapTransportError(error, request: request)
        }
    }

    /// Translates a `URLSession` failure into the SDK's error vocabulary.
    ///
    /// Cancellation is surfaced as `CancellationError` rather than an API failure, so callers can
    /// distinguish "the user navigated away" from "the provider is down".
    static func mapTransportError(_ error: any Error, request: HTTPRequest) -> any Error {
        if error is CancellationError { return error }

        guard let urlError = error as? URLError else {
            if let apiError = error as? APICallError { return apiError }
            return APICallError(
                message: "The request to \(request.url.absoluteString) failed: \(error)",
                url: request.url,
                method: request.method.rawValue,
                isRetryable: false,
                underlyingError: error
            )
        }

        if urlError.code == .cancelled { return CancellationError() }

        // Transient network conditions are worth another attempt; a bad URL or a failed TLS
        // handshake will fail identically no matter how many times it is retried.
        let retryableCodes: Set<URLError.Code> = [
            .timedOut, .cannotConnectToHost, .networkConnectionLost, .notConnectedToInternet,
            .dnsLookupFailed, .cannotFindHost, .resourceUnavailable, .badServerResponse,
        ]
        return APICallError(
            message: "The request to \(request.url.absoluteString) failed: \(urlError.localizedDescription)",
            url: request.url,
            method: request.method.rawValue,
            isRetryable: retryableCodes.contains(urlError.code),
            underlyingError: urlError
        )
    }
}

// MARK: - Request conversion

extension HTTPRequest {
    /// Converts to the `URLRequest` the session expects.
    func makeURLRequest() -> URLRequest {
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = method.rawValue
        urlRequest.httpBody = body
        for (name, value) in headers {
            urlRequest.setValue(value, forHTTPHeaderField: name)
        }
        if let timeout {
            urlRequest.timeoutInterval = timeout
        }
        return urlRequest
    }
}

// MARK: - Delegate

/// Bridges `URLSession`'s callback-based streaming onto `AsyncThrowingStream`.
///
/// One delegate instance serves every task on its session, keyed by task identifier. Access is
/// guarded by a lock rather than an actor because the delegate methods are synchronous and
/// non-isolated; hopping to an actor from each callback would reorder body chunks.
private final class StreamingSessionDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    /// Everything tracked for one in-flight request.
    private final class TaskState {
        var headContinuation: CheckedContinuation<HTTPResponseHead, any Error>?
        var bodyContinuation: AsyncThrowingStream<Data, any Error>.Continuation?
        var hasDeliveredHead = false
    }

    private let lock = NSLock()
    private var states: [Int: TaskState] = [:]

    /// Runs `body` with exclusive access to the state for a task, creating it if needed.
    private func withState<T>(_ identifier: Int, _ body: (TaskState) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        let state = states[identifier] ?? {
            let state = TaskState()
            states[identifier] = state
            return state
        }()
        return body(state)
    }

    func register(identifier: Int, bodyContinuation: AsyncThrowingStream<Data, any Error>.Continuation) {
        withState(identifier) { $0.bodyContinuation = bodyContinuation }
    }

    func setHeadContinuation(
        _ continuation: CheckedContinuation<HTTPResponseHead, any Error>,
        for identifier: Int
    ) {
        withState(identifier) { $0.headContinuation = continuation }
    }

    /// Terminates a request's streams, resuming the head continuation if it is still waiting.
    func finish(identifier: Int, error: (any Error)?) {
        lock.lock()
        let state = states.removeValue(forKey: identifier)
        lock.unlock()

        guard let state else { return }
        if let headContinuation = state.headContinuation {
            state.headContinuation = nil
            headContinuation.resume(throwing: error ?? CancellationError())
        }
        if let error {
            state.bodyContinuation?.finish(throwing: error)
        } else {
            state.bodyContinuation?.finish()
        }
    }

    // MARK: URLSessionDataDelegate

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let identifier = dataTask.taskIdentifier
        let head: HTTPResponseHead
        if let httpResponse = response as? HTTPURLResponse {
            var headers: [String: String] = [:]
            for (name, value) in httpResponse.allHeaderFields {
                guard let name = name as? String else { continue }
                headers[name] = String(describing: value)
            }
            head = HTTPResponseHead(statusCode: httpResponse.statusCode, headers: headers)
        } else {
            // Non-HTTP responses have no status; treat them as success and let the body speak.
            head = HTTPResponseHead(statusCode: 200)
        }

        let continuation: CheckedContinuation<HTTPResponseHead, any Error>? = withState(identifier) { state in
            defer { state.headContinuation = nil }
            state.hasDeliveredHead = true
            return state.headContinuation
        }
        continuation?.resume(returning: head)
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let continuation = withState(dataTask.taskIdentifier) { $0.bodyContinuation }
        continuation?.yield(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let identifier = task.taskIdentifier
        lock.lock()
        let state = states.removeValue(forKey: identifier)
        lock.unlock()
        guard let state else { return }

        // A failure before the head arrived must surface from `stream(_:)` itself rather than as
        // an empty stream that fails later, so the caller can retry it as a request failure.
        if !state.hasDeliveredHead, let headContinuation = state.headContinuation {
            state.headContinuation = nil
            headContinuation.resume(
                throwing: error ?? APICallError(
                    message: "The connection closed before a response was received.",
                    url: task.originalRequest?.url ?? URL(string: "about:blank")!,
                    isRetryable: true
                )
            )
            state.bodyContinuation?.finish()
            return
        }

        if let error {
            state.bodyContinuation?.finish(throwing: (error as? URLError)?.code == .cancelled
                ? CancellationError()
                : error)
        } else {
            state.bodyContinuation?.finish()
        }
    }
}

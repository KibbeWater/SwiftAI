import Foundation

/// Fans one sequence of elements out to any number of consumers.
///
/// `AsyncThrowingStream` supports a single iterator, but a streaming result needs several views of
/// the same data at once — the text, the full part stream, and the promised final values are all
/// derived from one provider response. A broadcaster sits between them.
///
/// Every element is retained and replayed to late subscribers, so a stream can be consumed after
/// it has already finished, and two consumers that start at different times still see identical
/// sequences. The memory cost is bounded by a single response, which is the right trade for the
/// alternative: a consumer that attaches a moment too late and silently misses the beginning.
///
/// Elements are delivered while the lock is held. Yielding to an `AsyncThrowingStream`
/// continuation only appends to its buffer and, at most, schedules a suspended consumer to
/// resume — it never runs consumer code inline — so this cannot deadlock, and it is what
/// guarantees a subscriber's replay is never interleaved with concurrently sent elements.
final class StreamBroadcaster<Element: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: [Element] = []
    private var subscribers: [UUID: AsyncThrowingStream<Element, any Error>.Continuation] = [:]
    private var terminal: Result<Void, any Error>?

    /// Whether the sequence has ended.
    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return terminal != nil
    }

    /// Every element sent so far.
    var elements: [Element] {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }

    /// Delivers an element to every current and future subscriber.
    ///
    /// Ignored once the sequence has finished, so a late-arriving part from a provider cannot
    /// reopen a completed stream.
    func send(_ element: Element) {
        lock.lock()
        defer { lock.unlock() }
        guard terminal == nil else { return }
        buffer.append(element)
        for continuation in subscribers.values {
            continuation.yield(element)
        }
    }

    /// Ends the sequence, successfully or with an error.
    ///
    /// Only the first call takes effect.
    ///
    /// Subscribers are finished *after* the lock is released. Finishing a continuation invokes
    /// its termination handler synchronously, and that handler takes this same lock to
    /// deregister — doing it while still holding the lock would deadlock.
    func finish(throwing error: (any Error)? = nil) {
        lock.lock()
        guard terminal == nil else {
            lock.unlock()
            return
        }
        terminal = error.map(Result.failure) ?? .success(())
        let pending = subscribers
        subscribers.removeAll()
        lock.unlock()

        // Safe to release first: `terminal` is already set, so `send` is now a no-op and no
        // element can slip in behind these.
        for continuation in pending.values {
            continuation.finish(throwing: error)
        }
    }

    /// Returns a new stream over every element, past and future.
    func subscribe() -> AsyncThrowingStream<Element, any Error> {
        AsyncThrowingStream<Element, any Error> { continuation in
            let id = UUID()
            lock.lock()
            // Replaying inside the lock is what keeps a subscriber's view ordered: a concurrent
            // `send` cannot slip an element in between the replay and going live.
            for element in buffer {
                continuation.yield(element)
            }
            switch terminal {
            case .success:
                continuation.finish()
            case .failure(let error):
                continuation.finish(throwing: error)
            case nil:
                subscribers[id] = continuation
                continuation.onTermination = { [weak self] _ in
                    self?.removeSubscriber(id)
                }
            }
            lock.unlock()
        }
    }

    private func removeSubscriber(_ id: UUID) {
        lock.lock()
        subscribers.removeValue(forKey: id)
        lock.unlock()
    }
}

/// A value that becomes available once, later.
///
/// Backs the promised accessors on streaming results — `result.usage` has to wait for the stream
/// to finish, but should be awaitable any number of times, from anywhere, before or after it
/// resolves.
final class AsyncPromise<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Value, any Error>?
    private var waiters: [UUID: CheckedContinuation<Value, any Error>] = [:]

    /// Resolves the promise. Only the first call takes effect.
    func fulfill(_ value: Value) {
        complete(.success(value))
    }

    /// Fails the promise. Only the first call takes effect.
    func fail(_ error: any Error) {
        complete(.failure(error))
    }

    private func complete(_ outcome: Result<Value, any Error>) {
        lock.lock()
        guard result == nil else {
            lock.unlock()
            return
        }
        result = outcome
        let pending = waiters
        waiters.removeAll()
        lock.unlock()

        for continuation in pending.values {
            continuation.resume(with: outcome)
        }
    }

    /// The value, waiting for it if necessary.
    ///
    /// The lock is only ever taken from the synchronous helpers below, because `NSLock` may not
    /// be held across a suspension point.
    var value: Value {
        get async throws {
            if let settled = currentResult() {
                return try settled.get()
            }

            let id = UUID()
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    // The promise may have resolved between the check above and this point.
                    if let settled = register(continuation, as: id) {
                        continuation.resume(with: settled)
                    }
                }
            } onCancel: {
                removeWaiter(id)?.resume(throwing: CancellationError())
            }
        }
    }

    private func currentResult() -> Result<Value, any Error>? {
        lock.withLock { result }
    }

    /// Registers a waiter, or returns the already-settled result if there is one.
    private func register(
        _ continuation: CheckedContinuation<Value, any Error>,
        as id: UUID
    ) -> Result<Value, any Error>? {
        lock.withLock {
            if let result { return result }
            waiters[id] = continuation
            return nil
        }
    }

    private func removeWaiter(_ id: UUID) -> CheckedContinuation<Value, any Error>? {
        lock.withLock { waiters.removeValue(forKey: id) }
    }
}

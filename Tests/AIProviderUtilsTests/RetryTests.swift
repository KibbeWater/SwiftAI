import AIProviderSpec
import Foundation
import Testing

@testable import AIProviderUtils

@Suite("Retry")
struct RetryTests {
    /// Records the delays a policy asks for, without spending real time.
    private final class SleepRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [Duration] = []

        var delays: [Duration] {
            lock.lock()
            defer { lock.unlock() }
            return recorded
        }

        func sleep(_ duration: Duration) {
            lock.lock()
            recorded.append(duration)
            lock.unlock()
        }
    }

    private func retryableError(statusCode: Int = 503, headers: [String: String] = [:]) -> APICallError {
        APICallError(
            message: "Service unavailable.",
            url: URL(string: "https://example.com/v1/chat")!,
            statusCode: statusCode,
            responseHeaders: headers
        )
    }

    @Test("Returns the first successful result without sleeping")
    func succeedsWithoutRetrying() async throws {
        let recorder = SleepRecorder()
        let value = try await withRetries(sleep: { recorder.sleep($0) }) { _ in "ok" }

        #expect(value == "ok")
        #expect(recorder.delays.isEmpty)
    }

    @Test("Retries a transient failure and then succeeds")
    func retriesThenSucceeds() async throws {
        let recorder = SleepRecorder()
        let attempts = Counter()

        let value = try await withRetries(sleep: { recorder.sleep($0) }) { attempt in
            attempts.increment()
            if attempt < 2 { throw self.retryableError() }
            return attempt
        }

        #expect(value == 2)
        #expect(attempts.value == 3)
        // Exponential backoff: two seconds, then four.
        #expect(recorder.delays == [.seconds(2), .seconds(4)])
    }

    @Test("Throws RetryError once the budget is exhausted")
    func throwsWhenBudgetExhausted() async {
        let recorder = SleepRecorder()

        await #expect(throws: RetryError.self) {
            try await withRetries(
                policy: RetryPolicy(maximumRetries: 2),
                sleep: { recorder.sleep($0) }
            ) { _ in
                throw self.retryableError()
            }
        }
        // Three attempts in total means two waits.
        #expect(recorder.delays.count == 2)
    }

    @Test("RetryError carries every attempt's failure")
    func retryErrorCarriesHistory() async throws {
        let recorder = SleepRecorder()
        var caught: RetryError?

        do {
            try await withRetries(
                policy: RetryPolicy(maximumRetries: 2),
                sleep: { recorder.sleep($0) }
            ) { _ in
                throw self.retryableError()
            }
        } catch let error as RetryError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.attempts == 3)
        #expect(error.errors.count == 3)
        #expect((error.lastError as? APICallError)?.statusCode == 503)
    }

    @Test("Propagates a non-retryable failure unchanged")
    func doesNotRetryNonRetryableFailures() async {
        let recorder = SleepRecorder()
        let attempts = Counter()

        // The caller should see the 401 itself, not a RetryError wrapping it.
        await #expect(throws: APICallError.self) {
            try await withRetries(sleep: { recorder.sleep($0) }) { _ in
                attempts.increment()
                throw self.retryableError(statusCode: 401)
            }
        }
        #expect(attempts.value == 1)
        #expect(recorder.delays.isEmpty)
    }

    @Test("Does not retry errors that are not API failures")
    func doesNotRetryArbitraryErrors() async {
        struct Boom: Error {}
        let attempts = Counter()

        await #expect(throws: Boom.self) {
            try await withRetries(sleep: { _ in }) { _ in
                attempts.increment()
                throw Boom()
            }
        }
        #expect(attempts.value == 1)
    }

    @Test("A policy with no retries makes exactly one attempt")
    func policyWithoutRetries() async {
        let attempts = Counter()

        await #expect(throws: RetryError.self) {
            try await withRetries(policy: .none, sleep: { _ in }) { _ in
                attempts.increment()
                throw self.retryableError()
            }
        }
        #expect(attempts.value == 1)
    }

    @Test("Caps the delay at the policy maximum")
    func capsDelayAtMaximum() async {
        let recorder = SleepRecorder()

        await #expect(throws: RetryError.self) {
            try await withRetries(
                policy: RetryPolicy(
                    maximumRetries: 4,
                    initialDelay: .seconds(10),
                    multiplier: 10,
                    maximumDelay: .seconds(30)
                ),
                sleep: { recorder.sleep($0) }
            ) { _ in
                throw self.retryableError()
            }
        }
        #expect(recorder.delays == [.seconds(10), .seconds(30), .seconds(30), .seconds(30)])
    }

    @Test("Honors a Retry-After header expressed in seconds")
    func honorsRetryAfterSeconds() async {
        let recorder = SleepRecorder()

        await #expect(throws: RetryError.self) {
            try await withRetries(
                policy: RetryPolicy(maximumRetries: 1),
                sleep: { recorder.sleep($0) }
            ) { _ in
                throw self.retryableError(statusCode: 429, headers: ["retry-after": "7"])
            }
        }
        #expect(recorder.delays == [.seconds(7)])
    }

    @Test("Ignores Retry-After when the policy disables it")
    func ignoresRetryAfterWhenDisabled() async {
        let recorder = SleepRecorder()

        await #expect(throws: RetryError.self) {
            try await withRetries(
                policy: RetryPolicy(maximumRetries: 1, honorsRetryAfterHeader: false),
                sleep: { recorder.sleep($0) }
            ) { _ in
                throw self.retryableError(statusCode: 429, headers: ["retry-after": "7"])
            }
        }
        #expect(recorder.delays == [.seconds(2)])
    }

    @Test("Reads Retry-After from a header name in any casing")
    func retryAfterIsCaseInsensitive() {
        let error = retryableError(statusCode: 429, headers: ["Retry-After": "12"])
        #expect(error.retryAfterDelay == .seconds(12))
    }

    @Test("Reads Retry-After expressed as an HTTP date")
    func retryAfterAcceptsHTTPDate() {
        let future = Date().addingTimeInterval(30)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"

        let error = retryableError(statusCode: 429, headers: ["retry-after": formatter.string(from: future)])
        let delay = try? #require(error.retryAfterDelay)
        // Allow a wide margin: the formatter truncates to whole seconds.
        #expect((delay ?? .zero) > .seconds(25))
        #expect((delay ?? .zero) <= .seconds(31))
    }

    @Test("A Retry-After date in the past yields no delay override")
    func retryAfterInThePastIsIgnored() {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
        let past = formatter.string(from: Date().addingTimeInterval(-3600))

        #expect(retryableError(headers: ["retry-after": past]).retryAfterDelay == nil)
    }

    @Test("Stops retrying when the surrounding task is cancelled")
    func stopsOnCancellation() async throws {
        let attempts = Counter()

        let task = Task {
            try await withRetries(
                policy: RetryPolicy(maximumRetries: 10),
                sleep: { _ in try await Task.sleep(for: .milliseconds(50)) }
            ) { _ in
                attempts.increment()
                throw self.retryableError()
            }
        }

        // Let the first attempt fail and the retry wait begin, then cancel.
        try await Task.sleep(for: .milliseconds(20))
        task.cancel()

        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(attempts.value < 10)
    }

    @Test("Duration scaling saturates rather than overflowing")
    func durationScalingSaturates() {
        #expect(Duration.seconds(2).multiplied(by: 2) == .seconds(4))
        #expect(Duration.seconds(1).multiplied(by: 0.5) == .milliseconds(500))
        // A factor large enough to overflow clamps instead of trapping.
        #expect(Duration.seconds(Int64.max / 2).multiplied(by: 1e9) == .seconds(Int64.max))
    }
}

/// A thread-safe counter for observing how many times a closure ran.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }
}

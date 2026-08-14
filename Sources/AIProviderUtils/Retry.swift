import AIProviderSpec
import Foundation

/// How failed requests are retried.
///
/// Retrying lives here, above the providers, so that every provider inherits identical behaviour
/// and none of them has to implement backoff. Providers' only responsibility is to classify their
/// failures accurately with ``APICallError/isRetryable``.
public struct RetryPolicy: Sendable, Hashable {
    /// How many additional attempts to make after the first one fails.
    ///
    /// A value of `2` means up to three attempts in total.
    public var maximumRetries: Int

    /// How long to wait before the first retry.
    public var initialDelay: Duration

    /// The factor by which the delay grows after each attempt.
    public var multiplier: Double

    /// A ceiling on the delay, so exponential growth cannot produce an absurd wait.
    public var maximumDelay: Duration

    /// Whether a `Retry-After` response header overrides the computed delay.
    ///
    /// Providers send this when rate limiting, and honoring it recovers faster than guessing
    /// while also being the polite thing to do.
    public var honorsRetryAfterHeader: Bool

    public init(
        maximumRetries: Int = 2,
        initialDelay: Duration = .seconds(2),
        multiplier: Double = 2,
        maximumDelay: Duration = .seconds(60),
        honorsRetryAfterHeader: Bool = true
    ) {
        self.maximumRetries = max(0, maximumRetries)
        self.initialDelay = initialDelay
        self.multiplier = multiplier
        self.maximumDelay = maximumDelay
        self.honorsRetryAfterHeader = honorsRetryAfterHeader
    }

    /// Two retries with a two-second initial delay, doubling each time.
    public static let `default` = RetryPolicy()

    /// No retries at all. Useful in tests and for callers doing their own scheduling.
    public static let none = RetryPolicy(maximumRetries: 0)
}

/// Runs an operation, retrying transient failures according to a policy.
///
/// Only ``APICallError``s that report themselves retryable are retried. Everything else — a
/// malformed request, an authentication failure, a bug in a tool body — propagates immediately
/// and unchanged, because repeating it would only waste time and quota.
///
/// Cancellation is honored between attempts and while waiting, so cancelling a generation does
/// not leave a retry timer running.
///
/// - Parameters:
///   - policy: How many times to retry, and how long to wait.
///   - sleep: How to wait between attempts. Injectable so tests do not spend real time; defaults
///     to `Task.sleep(for:)`.
///   - operation: The work to attempt. Receives the zero-based attempt number.
/// - Returns: The first successful result.
/// - Throws: The original error when it is not retryable, ``RetryError`` when the budget is
///   exhausted, or `CancellationError` when the surrounding task is cancelled.
public func withRetries<T>(
    policy: RetryPolicy = .default,
    sleep: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
    operation: (_ attempt: Int) async throws -> T
) async throws -> T {
    var errors: [any Error] = []
    var delay = policy.initialDelay

    for attempt in 0...policy.maximumRetries {
        try Task.checkCancellation()
        do {
            return try await operation(attempt)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // A cancelled URL request surfaces as a `URLError`, not a `CancellationError`.
            if Task.isCancelled { throw CancellationError() }

            guard let apiError = error as? APICallError, apiError.isRetryable else {
                throw error
            }
            errors.append(error)
            guard attempt < policy.maximumRetries else {
                throw RetryError(errors: errors, lastError: error)
            }

            let waitFor = policy.honorsRetryAfterHeader
                ? (apiError.retryAfterDelay ?? delay)
                : delay
            try await sleep(min(waitFor, policy.maximumDelay))
            delay = min(delay.multiplied(by: policy.multiplier), policy.maximumDelay)
        }
    }

    // Unreachable: the loop either returns or throws. Kept for exhaustiveness.
    throw RetryError(
        errors: errors,
        lastError: errors.last ?? InvalidArgumentError(argument: "policy", message: "No attempts were made.")
    )
}

extension APICallError {
    /// The delay requested by a `Retry-After` header, if the provider sent one.
    ///
    /// Both forms defined by RFC 9110 are recognized: a count of seconds, and an HTTP date.
    public var retryAfterDelay: Duration? {
        guard let value = responseHeaders.first(where: { $0.key.lowercased() == "retry-after" })?.value else {
            return nil
        }
        if let seconds = Double(value.trimmingCharacters(in: .whitespaces)), seconds >= 0 {
            return .seconds(seconds)
        }
        guard let date = APICallError.httpDateFormatter.date(from: value) else { return nil }
        let interval = date.timeIntervalSinceNow
        return interval > 0 ? .seconds(interval) : nil
    }

    /// RFC 9110 preferred date format, in the fixed locale the specification requires.
    private static let httpDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
        return formatter
    }()
}

extension Duration {
    /// Scales a duration by a factor, saturating rather than overflowing.
    func multiplied(by factor: Double) -> Duration {
        let (seconds, attoseconds) = components
        let total = Double(seconds) + Double(attoseconds) / 1e18
        let scaled = total * factor
        guard scaled.isFinite, scaled < Double(Int64.max) else { return .seconds(Int64.max) }
        return .seconds(scaled)
    }
}

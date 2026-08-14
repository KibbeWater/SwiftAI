import AIProviderSpec
import AIProviderUtils
import Foundation

/// A generation in progress.
///
/// The result is returned immediately, before the model has produced anything. Read it as a
/// stream of deltas, as a stream of typed events, or — once it finishes — as a set of final
/// values. Each view is independent and may be consumed at any time, including after the
/// generation completes.
///
/// ```swift
/// let stream = streamText(model: model, prompt: "Write a haiku about Malmö.")
/// for try await delta in stream.textStream {
///     print(delta, terminator: "")
/// }
/// print("\nUsed \(try await stream.totalUsage)")
/// ```
///
/// ## Failures
///
/// Nothing is thrown when the generation is started, because at that point nothing has been
/// attempted. A failure — a bad key, a rate limit, a dropped connection — surfaces when you
/// iterate, so wrap the loop rather than the call.
///
/// ## Cancellation
///
/// Cancelling the task that is iterating, or calling ``cancel()``, aborts the upstream request
/// and terminates every view with `CancellationError`. A result that is discarded without being
/// consumed also cancels, so an abandoned generation does not keep billing.
public final class StreamTextResult: Sendable {
    private let broadcaster = StreamBroadcaster<TextStreamPart>()
    private let stepsPromise = AsyncPromise<[StepResult]>()
    private let pumpTask: LockedBox<Task<Void, Never>?> = LockedBox(nil)

    init() {}

    // MARK: - Streams

    /// Every event, in order.
    ///
    /// Use this to drive an interface that shows more than the answer: reasoning as it is
    /// produced, tools as they run, progress through a multi-step generation.
    public var fullStream: AsyncThrowingStream<TextStreamPart, any Error> {
        broadcaster.subscribe()
    }

    /// The visible text, delta by delta.
    ///
    /// Reasoning, tool activity, and step boundaries are filtered out — this is what you would
    /// print to a terminal or append to a chat bubble.
    public var textStream: AsyncThrowingStream<String, any Error> {
        let upstream = broadcaster.subscribe()
        return AsyncThrowingStream<String, any Error> { continuation in
            let task = Task {
                do {
                    for try await part in upstream {
                        if let delta = part.textDelta {
                            continuation.yield(delta)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The reasoning the model exposed, delta by delta.
    public var reasoningStream: AsyncThrowingStream<String, any Error> {
        let upstream = broadcaster.subscribe()
        return AsyncThrowingStream<String, any Error> { continuation in
            let task = Task {
                do {
                    for try await part in upstream {
                        if let delta = part.reasoningDelta {
                            continuation.yield(delta)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Final values

    /// Every step that ran, once the generation finishes.
    public var steps: [StepResult] {
        get async throws { try await stepsPromise.value }
    }

    /// The complete text of the final step.
    public var text: String {
        get async throws { try await finalStep.text }
    }

    /// The reasoning of the final step, if the model exposed any.
    public var reasoningText: String? {
        get async throws { try await finalStep.reasoningText }
    }

    /// Everything the final step produced.
    public var content: [ModelContent] {
        get async throws { try await finalStep.content }
    }

    /// The tools the model asked for in the final step.
    public var toolCalls: [ToolCallPart] {
        get async throws { try await finalStep.toolCalls }
    }

    /// The results of tools that ran in the final step.
    public var toolResults: [ToolResultPart] {
        get async throws { try await finalStep.toolResults }
    }

    /// Why the generation stopped.
    public var finishReason: FinishReason {
        get async throws { try await finalStep.finishReason }
    }

    /// Token usage for the final step.
    public var usage: Usage {
        get async throws { try await finalStep.usage }
    }

    /// Token usage across every step. This is what the generation cost.
    public var totalUsage: Usage {
        get async throws {
            try await steps.reduce(Usage.none) { $0.adding($1.usage) }
        }
    }

    /// Settings no provider could honor.
    public var warnings: [CallWarning] {
        get async throws { try await steps.flatMap(\.warnings) }
    }

    /// Metadata about the final response.
    public var response: ResponseInfo? {
        get async throws { try await finalStep.response }
    }

    /// Every message the generation added to the conversation.
    ///
    /// Append these to your history to continue where the generation left off.
    public var responseMessages: [ModelMessage] {
        get async throws { try await steps.flatMap(\.messages) }
    }

    /// The complete result, in the same shape a buffered generation returns.
    ///
    /// Useful when the same code has to handle both, and when a caller streams for responsiveness
    /// but still wants the whole result afterwards.
    public var collected: GenerateTextResult {
        get async throws { GenerateTextResult(steps: try await steps) }
    }

    private var finalStep: StepResult {
        get async throws {
            let steps = try await steps
            guard let last = steps.last else {
                throw InvalidArgumentError(argument: "steps", message: "The generation produced no steps.")
            }
            return last
        }
    }

    // MARK: - Control

    /// Drains the stream without inspecting it.
    ///
    /// Use this when the parts are not needed but the generation should still run to completion —
    /// on a server that logs usage after a client has disconnected, for instance. Without it, a
    /// result nobody reads is cancelled.
    public func consume() async throws {
        for try await _ in fullStream {}
    }

    /// Aborts the generation.
    ///
    /// The upstream request is cancelled and every view terminates with `CancellationError`.
    public func cancel() {
        pumpTask.value?.cancel()
    }

    deinit {
        // A result that is discarded without being consumed should not keep generating.
        pumpTask.value?.cancel()
    }

    // MARK: - Internal plumbing

    func start(_ body: @escaping @Sendable (StreamTextResult) async -> Void) {
        let task = Task { [weak self] in
            guard let self else { return }
            await body(self)
        }
        pumpTask.value = task
    }

    func emit(_ part: TextStreamPart) {
        broadcaster.send(part)
    }

    /// Resolves the promised values without closing the stream.
    ///
    /// Split from ``closeStream()`` so that `onFinish` can run — and read the final values —
    /// before consumers observe the stream ending.
    func resolve(steps: [StepResult]) {
        stepsPromise.fulfill(steps)
    }

    func closeStream() {
        broadcaster.finish()
    }

    func fail(_ error: any Error) {
        stepsPromise.fail(error)
        broadcaster.finish(throwing: error)
    }
}

/// A minimal lock-protected box, for the small amount of mutable state a streaming result holds.
final class LockedBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Value

    init(_ value: Value) {
        self.storage = value
    }

    var value: Value {
        get {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
        set {
            lock.lock()
            storage = newValue
            lock.unlock()
        }
    }

    /// Mutates the value under the lock and returns whatever the closure produces.
    @discardableResult
    func withValue<T>(_ body: (inout Value) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(&storage)
    }
}

// MARK: - streamText

/// Streams generated text, optionally calling tools along the way.
///
/// The streaming counterpart to
/// ``generateText(model:system:prompt:tools:toolChoice:settings:stopWhen:prepareStep:onStepFinish:)``,
/// with identical semantics: the same tool loop, the same stop conditions, the same results. The
/// difference is that output is published as it arrives instead of at the end.
///
/// ```swift
/// let stream = streamText(
///     model: model,
///     system: "You are a helpful assistant.",
///     prompt: "Summarize the news.",
///     tools: [searchTool],
///     stopWhen: [.stepCount(5)]
/// )
///
/// for try await delta in stream.textStream {
///     await MainActor.run { message.text += delta }
/// }
/// ```
///
/// This function does not throw and is not `async`: it returns immediately, and failures surface
/// when the returned result is iterated.
///
/// - Parameters:
///   - model: The model to call.
///   - system: Instructions framing the conversation.
///   - prompt: The question, or the conversation so far.
///   - tools: Tools the model may call.
///   - toolChoice: How freely the model may call them.
///   - settings: Temperature, token limits, retries, and provider-specific options.
///   - stopWhen: When to stop looping. Defaults to a single step.
///   - prepareStep: Adjusts the model, tools, or conversation before each step.
///   - onStepFinish: Called after each step completes.
///   - onFinish: Called once with the complete result, after the last step.
///   - onError: Called if the generation fails. The error is also thrown to anyone iterating.
/// - Returns: The in-progress generation.
public func streamText(
    model: any LanguageModel,
    system: String? = nil,
    prompt: Prompt,
    tools: [any Tool] = [],
    toolChoice: ToolChoice = .auto,
    settings: GenerationSettings = GenerationSettings(),
    stopWhen: [StopCondition] = [.stepCount(1)],
    prepareStep: (@Sendable (PrepareStepContext) async throws -> PrepareStepAdjustments?)? = nil,
    onStepFinish: (@Sendable (StepResult) async -> Void)? = nil,
    onFinish: (@Sendable (GenerateTextResult) async -> Void)? = nil,
    onError: (@Sendable (any Error) async -> Void)? = nil
) -> StreamTextResult {
    let result = StreamTextResult()

    result.start { result in
        let observer = StreamingObserver(result: result, onStepFinish: onStepFinish)
        let loop = ToolLoop(
            model: model,
            tools: tools,
            toolChoice: toolChoice,
            settings: settings,
            stopWhen: stopWhen,
            responseFormat: nil,
            activeTools: nil,
            prepareStep: prepareStep,
            observer: observer
        )

        do {
            let messages = try prompt.resolved(system: system)
            result.emit(.start)

            let steps = try await loop.run(messages: messages) { model, options, retryPolicy, _ in
                try await performStreamingStep(
                    model: model,
                    options: options,
                    retryPolicy: retryPolicy,
                    result: result
                )
            }

            let totalUsage = steps.reduce(Usage.none) { $0.adding($1.usage) }
            result.emit(
                .finish(finishReason: steps.last?.finishReason ?? .unknown, totalUsage: totalUsage)
            )
            // Final values resolve first, then the callback runs, and only then does the stream
            // close. A caller who sees the stream end can therefore rely on `onFinish` having
            // already completed — which is what makes it usable for logging and persistence.
            result.resolve(steps: steps)
            await onFinish?(GenerateTextResult(steps: steps))
            result.closeStream()
        } catch {
            await onError?(error)
            result.fail(error)
        }
    }

    return result
}

/// Runs one streaming model call, republishing each part as it arrives.
///
/// Retrying covers establishing the stream only. Once a part has reached the consumer, replaying
/// the call would duplicate output, so a failure from that point on propagates.
private func performStreamingStep(
    model: any LanguageModel,
    options: LanguageModelCallOptions,
    retryPolicy: RetryPolicy,
    result: StreamTextResult
) async throws -> RawStep {
    let response = try await withRetries(policy: retryPolicy) { _ in
        try await model.stream(options)
    }

    var accumulator = StepAccumulator()
    for try await part in response.stream {
        try Task.checkCancellation()
        if let published = accumulator.consume(part) {
            result.emit(published)
        }
    }
    return accumulator.finalize(request: response.request, response: response.response)
}

/// Republishes the loop's own events onto the public stream.
private struct StreamingObserver: ToolLoopObserver {
    let result: StreamTextResult
    let onStepFinish: (@Sendable (StepResult) async -> Void)?

    func stepWillBegin(stepNumber: Int) async {
        result.emit(.stepStart(stepNumber: stepNumber))
    }

    func toolWillRun(call: ToolCallPart) async {
        result.emit(.toolWillRun(call))
    }

    func toolDidRun(result toolResult: ToolResultPart) async {
        result.emit(.toolResult(toolResult))
    }

    func stepDidFinish(_ step: StepResult) async {
        result.emit(.stepFinish(step))
        await onStepFinish?(step)
    }
}

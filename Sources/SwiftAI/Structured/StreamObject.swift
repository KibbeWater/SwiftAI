import AIProviderSpec
import AIProviderUtils
import Foundation

/// The machinery shared by the streaming object result types.
///
/// Holds the two broadcasters a structured stream produces — raw JSON text and decoded snapshots
/// — plus the promise that resolves when the value is complete.
final class ObjectStreamCore: Sendable {
    let text = StreamBroadcaster<String>()
    let snapshots = StreamBroadcaster<JSONValue>()
    let outcome = AsyncPromise<StructuredOutcome>()
    private let pumpTask = LockedBox<Task<Void, Never>?>(nil)

    func start(_ task: Task<Void, Never>) {
        pumpTask.value = task
    }

    func cancel() {
        pumpTask.value?.cancel()
    }

    func fail(_ error: any Error) {
        outcome.fail(error)
        text.finish(throwing: error)
        snapshots.finish(throwing: error)
    }

    func succeed(_ result: StructuredOutcome) {
        outcome.fulfill(result)
        text.finish()
        snapshots.finish()
    }

    deinit {
        pumpTask.value?.cancel()
    }
}

/// A structured generation in progress.
///
/// The value is published as it arrives, as a sequence of snapshots. Every snapshot is a complete,
/// valid ``StructuredValue/Partial`` in which properties that have not been produced yet are
/// `nil` — never a half-parsed string or an invalid intermediate. That makes the stream directly
/// bindable: a view can render each snapshot without checking whether the JSON happens to be
/// parseable at that instant.
///
/// ```swift
/// let stream = streamObject(model: model, of: Recipe.self, prompt: "A quick pasta dish.")
/// for try await recipe in stream.partialStream {
///     await MainActor.run {
///         title = recipe.name ?? "…"
///         ingredients = recipe.ingredients ?? []
///     }
/// }
/// let complete = try await stream.object
/// ```
///
/// Snapshots only ever gain information, so a field that has appeared never reverts to `nil`.
public final class StreamObjectResult<Value: StructuredOutput>: Sendable {
    private let core: ObjectStreamCore

    init(core: ObjectStreamCore) {
        self.core = core
    }

    /// The value as it is built up, one snapshot per meaningful change.
    ///
    /// Identical consecutive states are collapsed, so a snapshot arrives only when something
    /// actually changed rather than on every token.
    public var partialStream: AsyncThrowingStream<Value.Partial, any Error> {
        core.snapshots.mapped { Value.partial(from: $0) }
    }

    /// The raw JSON text, delta by delta.
    ///
    /// Rarely what you want — ``partialStream`` is the useful view — but available for logging
    /// and for forwarding a response verbatim.
    public var textStream: AsyncThrowingStream<String, any Error> {
        core.text.subscribe()
    }

    /// The finished, validated value.
    ///
    /// - Throws: ``NoObjectGeneratedError`` if the model's output does not parse or does not match
    ///   the schema.
    public var object: Value {
        get async throws {
            let outcome = try await core.outcome.value
            do {
                return try Value(structuredJSON: outcome.json)
            } catch {
                throw NoObjectGeneratedError(
                    text: outcome.rawText,
                    finishReason: outcome.step.finishReason,
                    usage: outcome.step.usage,
                    cause: error,
                    response: outcome.step.response
                )
            }
        }
    }

    /// Why generation stopped.
    public var finishReason: FinishReason {
        get async throws { try await core.outcome.value.step.finishReason }
    }

    /// Token usage for the call.
    public var usage: Usage {
        get async throws { try await core.outcome.value.step.usage }
    }

    /// Settings the provider could not honor.
    public var warnings: [CallWarning] {
        get async throws { try await core.outcome.value.step.warnings }
    }

    /// Metadata about the response.
    public var response: ResponseInfo? {
        get async throws { try await core.outcome.value.step.response }
    }

    /// The complete raw JSON the model produced.
    public var rawText: String {
        get async throws { try await core.outcome.value.rawText }
    }

    /// Aborts the generation.
    public func cancel() {
        core.cancel()
    }
}

/// A streaming generation of a list.
///
/// Adds ``elementStream`` to what ``StreamObjectResult`` offers: elements are published as soon as
/// each one is complete, which is what you want for a list that renders row by row.
public final class StreamArrayResult<Element: StructuredOutput>: Sendable {
    private let core: ObjectStreamCore

    init(core: ObjectStreamCore) {
        self.core = core
    }

    /// The list as it is built up, as snapshots of partially generated elements.
    public var partialStream: AsyncThrowingStream<[Element.Partial], any Error> {
        core.snapshots.mapped { [Element].partialValue(from: $0) ?? [] }
    }

    /// Complete elements, published as each one finishes.
    ///
    /// An element is treated as complete once a later element has started, since the model writes
    /// a list in order. The final element is published when the response ends.
    ///
    /// ```swift
    /// for try await recipe in stream.elementStream {
    ///     await MainActor.run { recipes.append(recipe) }
    /// }
    /// ```
    public var elementStream: AsyncThrowingStream<Element, any Error> {
        let snapshots = core.snapshots.subscribe()
        let outcome = core.outcome
        return AsyncThrowingStream<Element, any Error> { continuation in
            let task = Task {
                var emitted = 0
                do {
                    for try await snapshot in snapshots {
                        guard let elements = snapshot.arrayValue else { continue }
                        // Everything before the last entry has been fully written.
                        let complete = max(0, elements.count - 1)
                        while emitted < complete {
                            guard let element = try? Element(structuredJSON: elements[emitted]) else {
                                // Not decodable yet; wait for a later snapshot rather than
                                // emitting something half-formed.
                                break
                            }
                            continuation.yield(element)
                            emitted += 1
                        }
                    }
                    // The response is finished, so the last element is complete too.
                    let final = try await outcome.value
                    let elements = final.json.arrayValue ?? []
                    while emitted < elements.count {
                        continuation.yield(try Element(structuredJSON: elements[emitted]))
                        emitted += 1
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The raw JSON text, delta by delta.
    public var textStream: AsyncThrowingStream<String, any Error> {
        core.text.subscribe()
    }

    /// The finished, validated list.
    public var object: [Element] {
        get async throws {
            let outcome = try await core.outcome.value
            do {
                return try [Element](structuredJSON: outcome.json)
            } catch {
                throw NoObjectGeneratedError(
                    text: outcome.rawText,
                    finishReason: outcome.step.finishReason,
                    usage: outcome.step.usage,
                    cause: error,
                    response: outcome.step.response
                )
            }
        }
    }

    /// Why generation stopped.
    public var finishReason: FinishReason {
        get async throws { try await core.outcome.value.step.finishReason }
    }

    /// Token usage for the call.
    public var usage: Usage {
        get async throws { try await core.outcome.value.step.usage }
    }

    /// Settings the provider could not honor.
    public var warnings: [CallWarning] {
        get async throws { try await core.outcome.value.step.warnings }
    }

    /// Aborts the generation.
    public func cancel() {
        core.cancel()
    }
}

// MARK: - Public API

/// Streams a value of a given type, publishing snapshots as it is built up.
///
/// The streaming counterpart to ``generateObject(model:of:system:prompt:settings:)``. Use it when
/// a structured result takes long enough that showing progress matters — which, for anything
/// larger than a couple of fields, it usually does.
///
/// - Parameters:
///   - model: The model to call.
///   - type: The type to generate.
///   - system: Instructions framing the request.
///   - prompt: What to generate.
///   - settings: Temperature, token limits, retries, and provider-specific options.
/// - Returns: The in-progress generation. Failures surface when it is read, not here.
public func streamObject<Value: StructuredOutput>(
    model: any LanguageModel,
    of type: Value.Type = Value.self,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) -> StreamObjectResult<Value> {
    let core = ObjectStreamCore()
    startObjectStream(
        core: core,
        model: model,
        mode: .object(Value.jsonSchema),
        system: system,
        prompt: prompt,
        settings: settings
    )
    return StreamObjectResult(core: core)
}

/// Streams a list, publishing each element as it is completed.
///
/// - Parameters:
///   - model: The model to call.
///   - type: The element type.
///   - system: Instructions framing the request.
///   - prompt: What to generate.
///   - settings: Temperature, token limits, retries, and provider-specific options.
/// - Returns: The in-progress generation.
public func streamObject<Element: StructuredOutput>(
    model: any LanguageModel,
    arrayOf type: Element.Type,
    system: String? = nil,
    prompt: Prompt,
    settings: GenerationSettings = GenerationSettings()
) -> StreamArrayResult<Element> {
    let core = ObjectStreamCore()
    startObjectStream(
        core: core,
        model: model,
        mode: .array(element: Element.jsonSchema),
        system: system,
        prompt: prompt,
        settings: settings
    )
    return StreamArrayResult(core: core)
}

// MARK: - Implementation

/// The text accumulated so far, and the last snapshot published.
///
/// Held in a box because the step performer is a `@Sendable` closure and cannot capture mutable
/// local state.
private struct ObjectAccumulation {
    var text = ""
    var lastSnapshot: JSONValue?
}

/// Starts the background work behind a streaming structured generation.
private func startObjectStream(
    core: ObjectStreamCore,
    model: any LanguageModel,
    mode: ObjectOutputMode,
    schemaName: String? = nil,
    schemaDescription: String? = nil,
    system: String?,
    prompt: Prompt,
    settings: GenerationSettings
) {
    let task = Task {
        let accumulation = LockedBox(ObjectAccumulation())

        do {
            let messages = try prompt.resolved(system: system)
            let loop = ToolLoop(
                model: model,
                tools: [],
                toolChoice: .auto,
                settings: settings,
                stopWhen: [.stepCount(1)],
                responseFormat: .json(
                    schema: mode.requestSchema,
                    name: schemaName ?? mode.schemaName,
                    description: schemaDescription
                ),
                activeTools: nil,
                prepareStep: nil,
                observer: nil
            )

            let steps = try await loop.run(messages: messages) { model, options, retryPolicy, _ in
                let response = try await withRetries(policy: retryPolicy) { _ in
                    try await model.stream(options)
                }
                var accumulator = StepAccumulator()

                for try await part in response.stream {
                    try Task.checkCancellation()
                    _ = accumulator.consume(part)
                    guard case .textDelta(_, let delta) = part else { continue }

                    core.text.send(delta)

                    // Every prefix of the response is re-parsed leniently, so a snapshot is
                    // available at every point rather than only at token boundaries that happen
                    // to leave valid JSON.
                    let snapshot: JSONValue? = accumulation.withValue { state in
                        state.text += delta
                        guard let parsed = try? JSONValue.parse(state.text, mode: .partial),
                              let unwrapped = mode.unwrap(parsed),
                              unwrapped != state.lastSnapshot
                        else { return nil }
                        state.lastSnapshot = unwrapped
                        return unwrapped
                    }
                    if let snapshot {
                        core.snapshots.send(snapshot)
                    }
                }
                return accumulator.finalize(request: response.request, response: response.response)
            }

            guard let step = steps.last else {
                throw NoObjectGeneratedError(message: "The generation produced no steps.")
            }
            core.succeed(try finalize(step: step, mode: mode))
        } catch {
            core.fail(error)
        }
    }
    core.start(task)
}

/// Parses and unwraps the completed response.
private func finalize(step: StepResult, mode: ObjectOutputMode) throws -> StructuredOutcome {
    let rawText = step.text
    guard !rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw NoObjectGeneratedError(
            message: "The model returned no text to parse.",
            text: rawText,
            finishReason: step.finishReason,
            usage: step.usage,
            response: step.response
        )
    }

    let parsed: JSONValue
    do {
        parsed = try JSONValue.parse(rawText)
    } catch {
        throw NoObjectGeneratedError(
            text: rawText,
            finishReason: step.finishReason,
            usage: step.usage,
            cause: error,
            response: step.response
        )
    }

    guard let unwrapped = mode.unwrap(parsed) else {
        throw NoObjectGeneratedError(
            message: "The response was missing the expected wrapper property.",
            text: rawText,
            finishReason: step.finishReason,
            usage: step.usage,
            response: step.response
        )
    }
    return StructuredOutcome(json: unwrapped, rawText: rawText, step: step)
}

// MARK: - Broadcaster mapping

extension StreamBroadcaster {
    /// Returns a stream of transformed elements.
    func mapped<Transformed: Sendable>(
        _ transform: @escaping @Sendable (Element) -> Transformed
    ) -> AsyncThrowingStream<Transformed, any Error> {
        let upstream = subscribe()
        return AsyncThrowingStream<Transformed, any Error> { continuation in
            let task = Task {
                do {
                    for try await element in upstream {
                        continuation.yield(transform(element))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

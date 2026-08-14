import AIProviderSpec

/// Everything a text generation produced.
///
/// The properties describing "what the model said" — ``text``, ``toolCalls``, ``usage`` — refer to
/// the **final** step. For a single-step call that is the whole story. For a multi-step call, use
/// ``steps`` for the full history and ``totalUsage`` for what it all cost.
public struct GenerateTextResult: Sendable {
    /// Every step that ran, in order. Never empty.
    public var steps: [StepResult]

    public init(steps: [StepResult]) {
        precondition(!steps.isEmpty, "A generation always produces at least one step.")
        self.steps = steps
    }

    /// The last step, which holds the model's final answer.
    public var finalStep: StepResult { steps[steps.count - 1] }

    /// The text of the final step.
    ///
    /// This is the answer, with reasoning and tool activity excluded.
    public var text: String { finalStep.text }

    /// The reasoning the model exposed in the final step, if any.
    public var reasoningText: String? { finalStep.reasoningText }

    /// Everything the final step produced, in order.
    public var content: [ModelContent] { finalStep.content }

    /// The tools the model asked for in the final step.
    ///
    /// Non-empty here together with ``finishReason`` of ``FinishReason/toolCalls`` means the model
    /// is waiting on a client-executed tool.
    public var toolCalls: [ToolCallPart] { finalStep.toolCalls }

    /// The results of tools that ran in the final step.
    public var toolResults: [ToolResultPart] { finalStep.toolResults }

    /// Files the model produced in the final step.
    public var files: [FilePart] { finalStep.files }

    /// Sources the model cited in the final step.
    public var sources: [SourcePart] { finalStep.sources }

    /// Why the final step stopped.
    public var finishReason: FinishReason { finalStep.finishReason }

    /// Token usage for the final step.
    public var usage: Usage { finalStep.usage }

    /// Token usage across every step. This is what the call cost.
    public var totalUsage: Usage {
        steps.reduce(Usage.none) { $0.adding($1.usage) }
    }

    /// Settings no provider could honor, across every step.
    public var warnings: [CallWarning] { steps.flatMap(\.warnings) }

    /// Provider-specific values from the final step.
    public var providerMetadata: ProviderMetadata? { finalStep.providerMetadata }

    /// What was sent in the final step, for debugging.
    public var request: RequestInfo? { finalStep.request }

    /// Metadata about the final response.
    public var response: ResponseInfo? { finalStep.response }

    /// Every message the generation added to the conversation.
    ///
    /// Append these to your history to continue where the call left off. Reconstructing them by
    /// hand loses reasoning signatures and tool call identifiers that providers require:
    ///
    /// ```swift
    /// history.append(.user(question))
    /// let result = try await generateText(model: model, prompt: Prompt(history))
    /// history += result.responseMessages
    /// ```
    public var responseMessages: [ModelMessage] { steps.flatMap(\.messages) }
}

/// Generates text, optionally calling tools along the way.
///
/// The simplest form asks a question and reads the answer:
///
/// ```swift
/// let result = try await generateText(model: model, prompt: "Explain kinetic energy briefly.")
/// print(result.text)
/// ```
///
/// Supplying tools lets the model gather what it needs first. By default the call runs a single
/// step, so the model may *ask* for a tool but its result is not sent back. Pass a `stopWhen`
/// budget to make it a loop:
///
/// ```swift
/// let result = try await generateText(
///     model: model,
///     system: "You answer questions about the weather.",
///     prompt: "Should I take an umbrella in Malmö tomorrow?",
///     tools: [weatherTool],
///     stopWhen: [.stepCount(5)]
/// )
/// ```
///
/// Each pass runs the model, executes whatever tools it asked for concurrently, and feeds the
/// results back. The loop ends when the model answers without calling tools, when a stop condition
/// matches, or when the model calls a tool your application has to resolve.
///
/// - Parameters:
///   - model: The model to call.
///   - system: Instructions framing the conversation. Providers place these where they belong,
///     which is why they are passed separately from the prompt.
///   - prompt: The question, or the conversation so far.
///   - tools: Tools the model may call.
///   - toolChoice: How freely the model may call them. Ignored when `tools` is empty.
///   - settings: Temperature, token limits, retries, and provider-specific options.
///   - stopWhen: When to stop looping. Defaults to a single step.
///   - prepareStep: Adjusts the model, tools, or conversation before each step.
///   - onStepFinish: Called after each step, including any tools it ran. Useful for progress
///     reporting and for logging a long-running agent.
/// - Returns: The final answer, plus every step that led to it.
/// - Throws: ``APICallError`` for upstream failures, ``RetryError`` when retries are exhausted,
///   ``InvalidPromptError`` for a malformed prompt, or `CancellationError` if the task is
///   cancelled. A tool that throws does *not* fail the call: the error is reported to the model
///   as a tool result so it can recover.
public func generateText(
    model: any LanguageModel,
    system: String? = nil,
    prompt: Prompt,
    tools: [any Tool] = [],
    toolChoice: ToolChoice = .auto,
    settings: GenerationSettings = GenerationSettings(),
    stopWhen: [StopCondition] = [.stepCount(1)],
    prepareStep: (@Sendable (PrepareStepContext) async throws -> PrepareStepAdjustments?)? = nil,
    onStepFinish: (@Sendable (StepResult) async -> Void)? = nil
) async throws -> GenerateTextResult {
    let loop = ToolLoop(
        model: model,
        tools: tools,
        toolChoice: toolChoice,
        settings: settings,
        stopWhen: stopWhen,
        responseFormat: nil,
        activeTools: nil,
        prepareStep: prepareStep,
        observer: onStepFinish.map(CallbackObserver.init(onStepFinish:))
    )

    let steps = try await loop.run(messages: try prompt.resolved(system: system)) {
        model, options, retryPolicy, _ in
        // A buffered call produces nothing observable until it succeeds, so the whole thing can
        // be retried safely.
        try await withRetries(policy: retryPolicy) { _ in
            let response = try await model.generate(options)
            return RawStep(
                content: response.content,
                finishReason: response.finishReason,
                usage: response.usage,
                warnings: response.warnings,
                providerMetadata: response.providerMetadata,
                request: response.request,
                response: response.response
            )
        }
    }

    return GenerateTextResult(steps: steps)
}

/// Adapts an `onStepFinish` closure to the loop's observer protocol.
struct CallbackObserver: ToolLoopObserver {
    var onStepFinish: @Sendable (StepResult) async -> Void

    func stepWillBegin(stepNumber: Int) async {}
    func toolWillRun(call: ToolCallPart) async {}
    func toolDidRun(result: ToolResultPart) async {}
    func stepDidFinish(_ step: StepResult) async { await onStepFinish(step) }
}

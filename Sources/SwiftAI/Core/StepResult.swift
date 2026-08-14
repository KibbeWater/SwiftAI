import AIProviderSpec

/// One round trip to the model, plus any tools that ran as a result.
///
/// A generation with tools is a loop: the model answers, tools run, the results go back, the
/// model answers again. Each pass is a step, and every step is recorded so the whole exchange can
/// be inspected — for debugging, for token accounting, or for showing a user what happened.
public struct StepResult: Sendable {
    /// Everything the model produced this step, in order.
    public var content: [ModelContent]

    /// Why the model stopped this step.
    public var finishReason: FinishReason

    /// Token usage for this step alone.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values from this step.
    public var providerMetadata: ProviderMetadata?

    /// What was sent, for debugging.
    public var request: RequestInfo?

    /// Metadata about the response.
    public var response: ResponseInfo?

    /// The messages this step contributed to the conversation.
    ///
    /// One assistant message, plus a tool message when tools ran. Append these to your history to
    /// continue the conversation with everything the provider needs — including reasoning
    /// signatures and tool call identifiers that a reconstructed message would lose.
    public var messages: [ModelMessage]

    public init(
        content: [ModelContent],
        finishReason: FinishReason,
        usage: Usage = .none,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        request: RequestInfo? = nil,
        response: ResponseInfo? = nil,
        messages: [ModelMessage] = []
    ) {
        self.content = content
        self.finishReason = finishReason
        self.usage = usage
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.request = request
        self.response = response
        self.messages = messages
    }

    /// The visible text produced this step.
    public var text: String { content.text }

    /// The reasoning produced this step, if the model exposed any.
    public var reasoningText: String? { content.reasoningText }

    /// The tools the model asked to call this step.
    public var toolCalls: [ToolCallPart] { content.toolCalls }

    /// The results of the tools that ran this step.
    public var toolResults: [ToolResultPart] { content.toolResults }

    /// Files the model produced this step.
    public var files: [FilePart] { content.files }

    /// Sources the model cited this step.
    public var sources: [SourcePart] { content.sources }
}

/// Decides when a multi-step generation should stop.
///
/// Without one, ``generateText(model:system:prompt:tools:toolChoice:settings:stopWhen:prepareStep:onStepFinish:)``
/// runs a single step: the model may ask for tools, but the results are not sent back. Supplying a
/// stop condition is what turns a single call into an agentic loop.
///
/// ```swift
/// // Let the model use tools until it answers, but never more than five rounds.
/// stopWhen: [.stepCount(5)]
///
/// // Stop as soon as a particular tool has been used.
/// stopWhen: [.hasToolCall("submitAnswer")]
/// ```
///
/// Conditions are evaluated after every step and combined with logical *or*: the first one to
/// match ends the loop. The loop also ends on its own whenever a step finishes for any reason
/// other than ``FinishReason/toolCalls``, since there is nothing left to feed back.
///
/// A step budget is not optional in practice. A model that keeps calling tools will keep going
/// until something stops it, and that something should be a number you chose.
public struct StopCondition: Sendable {
    private let predicate: @Sendable ([StepResult]) async -> Bool

    /// Creates a condition from a predicate over the steps so far.
    public init(_ predicate: @escaping @Sendable ([StepResult]) async -> Bool) {
        self.predicate = predicate
    }

    func isSatisfied(by steps: [StepResult]) async -> Bool {
        await predicate(steps)
    }

    /// Stops once this many steps have run.
    public static func stepCount(_ count: Int) -> StopCondition {
        StopCondition { steps in steps.count >= count }
    }

    /// Stops once a particular tool has been called.
    ///
    /// The canonical use is an "answer" tool: give the model a tool whose only job is to submit
    /// its final result, and stop when it does.
    public static func hasToolCall(_ name: String) -> StopCondition {
        StopCondition { steps in
            steps.contains { step in step.toolCalls.contains { $0.toolName == name } }
        }
    }

    /// Stops once the accumulated token usage reaches a budget.
    public static func totalTokens(_ limit: Int) -> StopCondition {
        StopCondition { steps in
            let total = steps.reduce(Usage.none) { $0.adding($1.usage) }
            return (total.resolvedTotalTokens ?? 0) >= limit
        }
    }
}

// MARK: - Per-step preparation

/// The state of a generation at the start of a step.
public struct PrepareStepContext: Sendable {
    /// The steps that have already run.
    public var steps: [StepResult]

    /// Which step is about to run, counting from zero.
    public var stepNumber: Int

    /// The conversation as it will be sent, unless overridden.
    public var messages: [ModelMessage]
}

/// Adjustments to apply to a single step.
///
/// Every field is optional; anything left `nil` keeps the value the call was configured with.
///
/// ```swift
/// prepareStep: { context in
///     // Use a cheaper model once the plan is settled.
///     context.stepNumber == 0 ? nil : PrepareStepAdjustments(model: fastModel)
/// }
/// ```
public struct PrepareStepAdjustments: Sendable {
    /// Use a different model for this step.
    public var model: (any LanguageModel)?

    /// Change how freely the model may call tools this step.
    public var toolChoice: ToolChoice?

    /// Restrict which tools are offered this step, by name.
    ///
    /// The tools themselves stay registered, so results from earlier steps still resolve. This is
    /// the mechanism behind staged workflows, where a later phase should not revisit an earlier
    /// phase's tools.
    public var activeTools: [String]?

    /// Replace the conversation for this step.
    ///
    /// Use it to compress history once it grows long, which keeps a long-running agent inside its
    /// context window.
    public var messages: [ModelMessage]?

    /// Override generation settings for this step.
    public var settings: GenerationSettings?

    public init(
        model: (any LanguageModel)? = nil,
        toolChoice: ToolChoice? = nil,
        activeTools: [String]? = nil,
        messages: [ModelMessage]? = nil,
        settings: GenerationSettings? = nil
    ) {
        self.model = model
        self.toolChoice = toolChoice
        self.activeTools = activeTools
        self.messages = messages
        self.settings = settings
    }
}

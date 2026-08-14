import AIProviderSpec

/// A model, its instructions, and its tools, configured once and used many times.
///
/// An agent holds everything that stays the same between calls, so the call site carries only what
/// changes — usually just the prompt. Beyond convenience, that keeps a system prompt and its tools
/// together, which is where they belong: a tool is only useful if the instructions tell the model
/// when to reach for it.
///
/// ```swift
/// let researcher = Agent(
///     model: model,
///     system: """
///         You answer questions about the company's documentation. Search before answering, \
///         and cite the page you used.
///         """,
///     tools: [searchTool, fetchPageTool],
///     stopWhen: [.stepCount(8)]
/// )
///
/// let answer = try await researcher.generate(prompt: "What is our refund policy?")
/// ```
///
/// The default step budget is 20, because an agent is expected to loop; the free functions default
/// to a single step, because they are not.
///
/// Agents are values. Making a variant is a copy with one thing changed, which composes better
/// than mutating shared configuration:
///
/// ```swift
/// let careful = researcher.with(settings: .deterministic)
/// ```
public struct Agent: Sendable {
    /// The model calls are sent to.
    public var model: any LanguageModel

    /// Instructions framing every conversation.
    public var system: String?

    /// The tools the model may call.
    public var tools: [any Tool]

    /// How freely the model may call them.
    public var toolChoice: ToolChoice

    /// When to stop looping.
    public var stopWhen: [StopCondition]

    /// Temperature, token limits, retries, and provider-specific options.
    public var settings: GenerationSettings

    /// Adjusts the model, tools, or conversation before each step.
    public var prepareStep: (@Sendable (PrepareStepContext) async throws -> PrepareStepAdjustments?)?

    /// Creates an agent.
    ///
    /// - Parameters:
    ///   - model: The model to call.
    ///   - system: Instructions framing every conversation.
    ///   - tools: The tools the model may call.
    ///   - toolChoice: How freely the model may call them.
    ///   - stopWhen: When to stop looping. Defaults to a budget of 20 steps.
    ///   - settings: Temperature, token limits, retries, and provider-specific options.
    ///   - prepareStep: Adjusts the model, tools, or conversation before each step.
    public init(
        model: any LanguageModel,
        system: String? = nil,
        tools: [any Tool] = [],
        toolChoice: ToolChoice = .auto,
        stopWhen: [StopCondition] = [.stepCount(20)],
        settings: GenerationSettings = GenerationSettings(),
        prepareStep: (@Sendable (PrepareStepContext) async throws -> PrepareStepAdjustments?)? = nil
    ) {
        self.model = model
        self.system = system
        self.tools = tools
        self.toolChoice = toolChoice
        self.stopWhen = stopWhen
        self.settings = settings
        self.prepareStep = prepareStep
    }

    /// Runs the agent to completion.
    ///
    /// - Parameters:
    ///   - prompt: The question, or the conversation so far.
    ///   - onStepFinish: Called after each step. Useful for surfacing progress during a long run.
    /// - Returns: The final answer, plus every step that led to it.
    public func generate(
        prompt: Prompt,
        onStepFinish: (@Sendable (StepResult) async -> Void)? = nil
    ) async throws -> GenerateTextResult {
        try await generateText(
            model: model,
            system: system,
            prompt: prompt,
            tools: tools,
            toolChoice: toolChoice,
            settings: settings,
            stopWhen: stopWhen,
            prepareStep: prepareStep,
            onStepFinish: onStepFinish
        )
    }

    /// Runs the agent, publishing output as it arrives.
    ///
    /// - Parameters:
    ///   - prompt: The question, or the conversation so far.
    ///   - onStepFinish: Called after each step.
    ///   - onFinish: Called once with the complete result.
    ///   - onError: Called if the run fails.
    /// - Returns: The in-progress run.
    public func stream(
        prompt: Prompt,
        onStepFinish: (@Sendable (StepResult) async -> Void)? = nil,
        onFinish: (@Sendable (GenerateTextResult) async -> Void)? = nil,
        onError: (@Sendable (any Error) async -> Void)? = nil
    ) -> StreamTextResult {
        streamText(
            model: model,
            system: system,
            prompt: prompt,
            tools: tools,
            toolChoice: toolChoice,
            settings: settings,
            stopWhen: stopWhen,
            prepareStep: prepareStep,
            onStepFinish: onStepFinish,
            onFinish: onFinish,
            onError: onError
        )
    }

    /// Returns a copy with the given properties replaced.
    ///
    /// Anything omitted is carried over unchanged.
    public func with(
        model: (any LanguageModel)? = nil,
        system: String? = nil,
        tools: [any Tool]? = nil,
        toolChoice: ToolChoice? = nil,
        stopWhen: [StopCondition]? = nil,
        settings: GenerationSettings? = nil
    ) -> Agent {
        var copy = self
        if let model { copy.model = model }
        if let system { copy.system = system }
        if let tools { copy.tools = tools }
        if let toolChoice { copy.toolChoice = toolChoice }
        if let stopWhen { copy.stopWhen = stopWhen }
        if let settings { copy.settings = settings }
        return copy
    }

    /// Returns a copy with additional tools.
    public func addingTools(_ additional: [any Tool]) -> Agent {
        var copy = self
        copy.tools += additional
        return copy
    }
}

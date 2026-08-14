import AIProviderSpec
import AIProviderUtils
import Foundation

/// What one round trip to a model produced, before tools have run.
struct RawStep: Sendable {
    var content: [ModelContent]
    var finishReason: FinishReason
    var usage: Usage = .none
    var warnings: [CallWarning] = []
    var providerMetadata: ProviderMetadata?
    var request: RequestInfo?
    var response: ResponseInfo?
}

/// Events the loop reports as it runs, so a streaming caller can forward them.
///
/// Buffered generation ignores these; streaming turns each into a part on the public stream. The
/// loop itself is identical either way, which is the point: there is one implementation of the
/// agentic loop, not one per calling convention.
protocol ToolLoopObserver: Sendable {
    /// A step is about to run.
    func stepWillBegin(stepNumber: Int) async

    /// A tool is about to be invoked.
    func toolWillRun(call: ToolCallPart) async

    /// A tool finished, successfully or not.
    func toolDidRun(result: ToolResultPart) async

    /// A step finished, including any tools it ran.
    func stepDidFinish(_ step: StepResult) async
}

/// Runs the model, executes the tools it asks for, and feeds the results back until a stop
/// condition is met.
///
/// The loop is deliberately separate from how a single step is performed. `performStep` is
/// supplied by the caller: ``generateText(model:system:prompt:tools:toolChoice:settings:stopWhen:prepareStep:onStepFinish:)``
/// passes one that calls the model's buffered path, while `streamText` passes one that consumes
/// the model's stream and republishes each part. Everything else — retries, tool dispatch, message
/// assembly, stop conditions — happens here, once.
struct ToolLoop: Sendable {
    var model: any LanguageModel
    var tools: [any Tool]
    var toolChoice: ToolChoice
    var settings: GenerationSettings
    var stopWhen: [StopCondition]
    var responseFormat: ResponseFormat?
    var activeTools: [String]?
    var prepareStep: (@Sendable (PrepareStepContext) async throws -> PrepareStepAdjustments?)?
    var observer: (any ToolLoopObserver)?

    /// Performs a single model call. See ``ToolLoop`` for why this is injected.
    ///
    /// Retrying is the performer's responsibility rather than the loop's, because how much of a
    /// call may safely be retried depends on how it is made. A buffered call can be retried
    /// wholesale; a streaming one can only be retried up to the point where the first part has
    /// been published, since replaying a stream would duplicate what the consumer already saw.
    typealias StepPerformer = @Sendable (
        _ model: any LanguageModel,
        _ options: LanguageModelCallOptions,
        _ retryPolicy: RetryPolicy,
        _ stepNumber: Int
    ) async throws -> RawStep

    /// Runs the loop to completion.
    ///
    /// - Parameters:
    ///   - messages: The starting conversation, system message included.
    ///   - performStep: How to perform one model call.
    /// - Returns: Every step that ran, in order. Never empty.
    func run(
        messages: [ModelMessage],
        performStep: StepPerformer
    ) async throws -> [StepResult] {
        let toolsByName = Dictionary(tools.map { ($0.name, $0) }, uniquingKeysWith: { _, last in last })
        var steps: [StepResult] = []
        var conversation = messages
        var stepNumber = 0

        while true {
            try Task.checkCancellation()
            await observer?.stepWillBegin(stepNumber: stepNumber)

            // 1. Let the caller adjust this step.
            let adjustments = try await prepareStep?(
                PrepareStepContext(steps: steps, stepNumber: stepNumber, messages: conversation)
            )
            let stepModel = adjustments?.model ?? model
            let stepMessages = adjustments?.messages ?? conversation
            let stepSettings = settings.merging(adjustments?.settings)
            let stepToolChoice = adjustments?.toolChoice ?? toolChoice
            let offeredNames = adjustments?.activeTools ?? activeTools
            let offeredTools = offeredNames.map { names in
                tools.filter { names.contains($0.name) }
            } ?? tools

            // 2. Substitute bytes for any file URL the model cannot fetch itself.
            let resolvedMessages = try await FileResolver.resolve(
                messages: stepMessages,
                for: stepModel,
                transport: stepSettings.fileTransport ?? URLSessionTransport.shared
            )

            let options = stepSettings.callOptions(
                prompt: resolvedMessages,
                tools: offeredTools.map { $0.languageModelTool() },
                // Sending a tool choice with no tools confuses several providers.
                toolChoice: offeredTools.isEmpty ? nil : stepToolChoice,
                responseFormat: responseFormat
            )

            // 3. Call the model.
            let raw = try await performStep(stepModel, options, stepSettings.retryPolicy, stepNumber)

            // 4. Run whatever tools the model asked for.
            let outcome = await runTools(
                calls: raw.content.toolCalls,
                toolsByName: toolsByName,
                conversation: resolvedMessages,
                stepNumber: stepNumber
            )

            // 5. Record the step.
            let content = raw.content + outcome.results.map(ModelContent.toolResult)
            let step = StepResult(
                content: content,
                finishReason: raw.finishReason,
                usage: raw.usage,
                warnings: raw.warnings,
                providerMetadata: raw.providerMetadata,
                request: raw.request,
                response: raw.response,
                messages: Self.conversationMessages(from: raw.content, toolResults: outcome.results)
            )
            steps.append(step)
            await observer?.stepDidFinish(step)

            // 6. Decide whether to go around again.
            //
            // Another step is only warranted when the model stopped to wait for tools *and*
            // something new came back for it to read. A pending client-executed call ends the
            // loop by design: the application has to supply that result itself.
            let shouldContinue = raw.finishReason == .toolCalls
                && !outcome.results.isEmpty
                && !outcome.hasPendingClientCall
            guard shouldContinue else { return steps }

            for condition in stopWhen where await condition.isSatisfied(by: steps) {
                return steps
            }

            conversation += step.messages
            stepNumber += 1
        }
    }

    // MARK: - Tool execution

    private struct ToolOutcome: Sendable {
        var results: [ToolResultPart] = []
        /// Whether the model called a tool the application has to resolve.
        var hasPendingClientCall = false
    }

    /// Runs every locally executable tool the model called, concurrently.
    ///
    /// Failures become error results rather than thrown errors. A model that receives "the city
    /// was not found" can try a different city; one that receives nothing at all cannot do
    /// anything useful, and neither can the caller.
    private func runTools(
        calls: [ToolCallPart],
        toolsByName: [String: any Tool],
        conversation: [ModelMessage],
        stepNumber: Int
    ) async -> ToolOutcome {
        var outcome = ToolOutcome()
        guard !calls.isEmpty else { return outcome }

        var executable: [(offset: Int, call: ToolCallPart, tool: any Tool)] = []

        for (offset, call) in calls.enumerated() {
            // A provider-executed call arrives with its result already attached.
            guard !call.providerExecuted else { continue }

            guard let tool = toolsByName[call.toolName] else {
                let error = NoSuchToolError(toolName: call.toolName, availableTools: Array(toolsByName.keys))
                outcome.results.append(
                    ToolResultPart(
                        toolCallID: call.toolCallID,
                        toolName: call.toolName,
                        output: .errorText(error.message),
                        isDynamic: call.isDynamic
                    )
                )
                continue
            }

            switch tool.execution {
            case .client:
                outcome.hasPendingClientCall = true
            case .provider:
                // The provider owns these; nothing to do locally.
                continue
            case .local:
                executable.append((offset, call, tool))
            }
        }

        guard !executable.isEmpty else { return outcome }

        // Tools run concurrently: a model that asks for three lookups at once should not wait for
        // them serially. Results are reordered afterwards so the model always sees them in the
        // order it requested them, which keeps replayed conversations stable.
        let executed = await withTaskGroup(
            of: (Int, ToolResultPart).self,
            returning: [(Int, ToolResultPart)].self
        ) { group in
            for (offset, call, tool) in executable {
                group.addTask {
                    await observer?.toolWillRun(call: call)
                    let context = ToolContext(
                        toolCallID: call.toolCallID,
                        messages: conversation,
                        stepNumber: stepNumber
                    )
                    let output: ToolOutput
                    do {
                        output = try await tool.invoke(rawInput: call.input, context: context)
                    } catch is CancellationError {
                        output = .errorText("The tool call was cancelled.")
                    } catch let error as InvalidToolInputError {
                        output = .errorText(error.message)
                    } catch {
                        output = .errorText("The tool failed: \(error)")
                    }
                    let result = ToolResultPart(
                        toolCallID: call.toolCallID,
                        toolName: call.toolName,
                        output: output,
                        isDynamic: call.isDynamic
                    )
                    await observer?.toolDidRun(result: result)
                    return (offset, result)
                }
            }
            var collected: [(Int, ToolResultPart)] = []
            for await element in group { collected.append(element) }
            return collected
        }

        outcome.results.append(contentsOf: executed.sorted { $0.0 < $1.0 }.map(\.1))
        return outcome
    }

    // MARK: - Message assembly

    /// Builds the messages a step contributes to the conversation.
    ///
    /// Citations are dropped: no provider accepts them as input, and replaying them would be
    /// rejected. Everything else is preserved verbatim, including reasoning signatures that some
    /// providers require on the next turn.
    static func conversationMessages(
        from content: [ModelContent],
        toolResults: [ToolResultPart]
    ) -> [ModelMessage] {
        var messages: [ModelMessage] = []

        let assistantContent = content.filter { part in
            if case .source = part { return false }
            return true
        }
        if !assistantContent.isEmpty {
            messages.append(.assistant(assistantContent))
        }
        if !toolResults.isEmpty {
            messages.append(.tool(toolResults))
        }
        return messages
    }
}

// MARK: - File resolution

/// Substitutes inline bytes for file URLs a model cannot fetch itself.
///
/// Providers differ: some accept an image URL and fetch it, others require base64. Rather than
/// making that the caller's problem, a URL the model declines is downloaded here and passed on as
/// data, so the same prompt works against every provider.
enum FileResolver {
    static func resolve(
        messages: [ModelMessage],
        for model: any LanguageModel,
        transport: any HTTPTransport
    ) async throws -> [ModelMessage] {
        // The common case is no remote files at all, so check before doing any work.
        guard messages.contains(where: hasRemoteFile) else { return messages }

        var resolved: [ModelMessage] = []
        resolved.reserveCapacity(messages.count)

        for message in messages {
            guard case .user(var userMessage) = message else {
                resolved.append(message)
                continue
            }
            for (index, part) in userMessage.content.enumerated() {
                guard case .file(let file) = part,
                      case .url(let url) = file.source,
                      !model.supportsNativeURL(url, mediaType: file.mediaType)
                else { continue }

                var downloaded = file
                downloaded.source = .data(try await download(url, using: transport))
                userMessage.content[index] = .file(downloaded)
            }
            resolved.append(.user(userMessage))
        }
        return resolved
    }

    private static func hasRemoteFile(_ message: ModelMessage) -> Bool {
        guard case .user(let userMessage) = message else { return false }
        return userMessage.content.contains { part in
            guard case .file(let file) = part, case .url = file.source else { return false }
            return true
        }
    }

    private static func download(_ url: URL, using transport: any HTTPTransport) async throws -> Data {
        let response: HTTPResponse
        do {
            response = try await transport.send(HTTPRequest(url: url, method: .get))
        } catch {
            throw DownloadError(url: url, underlyingError: error)
        }
        guard response.head.isSuccess else {
            throw DownloadError(
                url: url,
                statusCode: response.statusCode,
                message: "Downloading '\(url.absoluteString)' failed with status \(response.statusCode)."
            )
        }
        return response.body
    }
}

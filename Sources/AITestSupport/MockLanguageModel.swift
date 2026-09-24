import AIProviderSpec
import Foundation

/// A scripted ``LanguageModel`` for testing everything above the provider layer.
///
/// Where ``MockHTTPTransport`` exercises a real provider against recorded bytes, this replaces the
/// provider entirely. It is the right tool for testing the tool loop, middleware, and streaming
/// machinery, none of which should care which provider produced a response.
///
/// ```swift
/// let model = MockLanguageModel(responses: [
///     .toolCall(name: "weather", input: ["city": "Malmö"]),
///     .text("It is raining."),
/// ])
/// let result = try await generateText(model: model, prompt: "Weather?", tools: [weather])
/// #expect(model.recordedCalls.count == 2)
/// ```
public final class MockLanguageModel: LanguageModel, @unchecked Sendable {
    public let provider: String
    public let modelID: String

    /// One scripted response.
    public struct Response: Sendable {
        public var content: [ModelContent]
        public var finishReason: FinishReason
        public var usage: Usage
        public var warnings: [CallWarning]
        public var providerMetadata: ProviderMetadata?

        /// An error to throw instead of responding.
        public var error: (any Error)?

        /// For streaming, how to split text into deltas. `nil` emits the text in one delta.
        public var textChunkSize: Int?

        public init(
            content: [ModelContent] = [],
            finishReason: FinishReason = .stop,
            usage: Usage = Usage(inputTokens: 10, outputTokens: 5),
            warnings: [CallWarning] = [],
            providerMetadata: ProviderMetadata? = nil,
            error: (any Error)? = nil,
            textChunkSize: Int? = nil
        ) {
            self.content = content
            self.finishReason = finishReason
            self.usage = usage
            self.warnings = warnings
            self.providerMetadata = providerMetadata
            self.error = error
            self.textChunkSize = textChunkSize
        }

        /// A plain text response.
        public static func text(
            _ text: String,
            finishReason: FinishReason = .stop,
            usage: Usage = Usage(inputTokens: 10, outputTokens: 5),
            textChunkSize: Int? = nil
        ) -> Response {
            Response(
                content: [.text(TextPart(text))],
                finishReason: finishReason,
                usage: usage,
                textChunkSize: textChunkSize
            )
        }

        /// A response that reasons before answering.
        public static func reasoningThenText(
            reasoning: String,
            text: String,
            usage: Usage = Usage(inputTokens: 10, outputTokens: 5, reasoningTokens: 20)
        ) -> Response {
            Response(
                content: [.reasoning(ReasoningPart(reasoning)), .text(TextPart(text))],
                finishReason: .stop,
                usage: usage
            )
        }

        /// A response asking to call one tool.
        public static func toolCall(
            id: String = "call_1",
            name: String,
            input: JSONValue,
            usage: Usage = Usage(inputTokens: 10, outputTokens: 5)
        ) -> Response {
            Response(
                content: [.toolCall(ToolCallPart(toolCallID: id, toolName: name, input: input))],
                finishReason: .toolCalls,
                usage: usage
            )
        }

        /// A response asking to call several tools at once.
        public static func toolCalls(_ calls: [ToolCallPart]) -> Response {
            Response(content: calls.map(ModelContent.toolCall), finishReason: .toolCalls)
        }

        /// A response that fails.
        public static func failure(_ error: any Error) -> Response {
            Response(error: error)
        }
    }

    /// A record of one call the model received.
    public struct RecordedCall: Sendable {
        public var options: LanguageModelCallOptions
        public var wasStreaming: Bool
    }

    private let lock = NSLock()
    private var queue: [Response]
    private var recorded: [RecordedCall] = []

    /// Whether the final response repeats once the queue is exhausted.
    public let repeatsLastResponse: Bool

    /// Called before each response is produced. Use it to assert on call ordering, or to inject
    /// a delay so cancellation can be tested.
    public var onCall: (@Sendable (LanguageModelCallOptions) async throws -> Void)?

    public init(
        provider: String = "mock",
        modelID: String = "mock-model",
        responses: [Response],
        repeatsLastResponse: Bool = false
    ) {
        self.provider = provider
        self.modelID = modelID
        self.queue = responses
        self.repeatsLastResponse = repeatsLastResponse
    }

    /// Creates a model that answers every call with the same text.
    public convenience init(provider: String = "mock", modelID: String = "mock-model", text: String) {
        self.init(
            provider: provider,
            modelID: modelID,
            responses: [.text(text)],
            repeatsLastResponse: true
        )
    }

    /// Every call the model received, in order.
    public var recordedCalls: [RecordedCall] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// The options from a recorded call.
    public func recordedOptions(at index: Int = 0) -> LanguageModelCallOptions? {
        let calls = recordedCalls
        return calls.indices.contains(index) ? calls[index].options : nil
    }

    private func nextResponse(for options: LanguageModelCallOptions, streaming: Bool) throws -> Response {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(RecordedCall(options: options, wasStreaming: streaming))

        if queue.count > 1 || !repeatsLastResponse {
            guard !queue.isEmpty else {
                throw MockModelError.responsesExhausted(callCount: recorded.count)
            }
            return queue.removeFirst()
        }
        guard let last = queue.first else {
            throw MockModelError.responsesExhausted(callCount: recorded.count)
        }
        return last
    }

    // MARK: - LanguageModel

    public func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        try await onCall?(options)
        let response = try nextResponse(for: options, streaming: false)
        if let error = response.error { throw error }
        return LanguageModelResponse(
            content: response.content,
            finishReason: response.finishReason,
            usage: response.usage,
            warnings: response.warnings,
            providerMetadata: response.providerMetadata,
            request: RequestInfo(body: "{}"),
            response: ResponseInfo(id: "mock-response", modelID: modelID)
        )
    }

    public func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        try await onCall?(options)
        let response = try nextResponse(for: options, streaming: true)
        if let error = response.error, response.content.isEmpty { throw error }

        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()
        continuation.yield(.streamStart(warnings: response.warnings))
        continuation.yield(.responseMetadata(id: "mock-response", modelID: modelID))

        for (index, part) in response.content.enumerated() {
            let blockID = "block_\(index)"
            switch part {
            case .text(let text):
                continuation.yield(.textStart(id: blockID))
                for chunk in MockLanguageModel.split(text.text, by: response.textChunkSize) {
                    continuation.yield(.textDelta(id: blockID, delta: chunk))
                }
                continuation.yield(.textEnd(id: blockID))

            case .reasoning(let reasoning):
                continuation.yield(.reasoningStart(id: blockID))
                for chunk in MockLanguageModel.split(reasoning.text, by: response.textChunkSize) {
                    continuation.yield(.reasoningDelta(id: blockID, delta: chunk))
                }
                // A reasoning part's options are what a real provider would have attached as
                // metadata, so they travel on the end part the way a provider's would.
                continuation.yield(.reasoningEnd(
                    id: blockID,
                    providerMetadata: reasoning.providerOptions.map { ProviderMetadata($0.namespaces) }
                ))

            case .toolCall(let call):
                // Real providers stream tool arguments as JSON fragments, so the mock does too.
                continuation.yield(.toolInputStart(id: call.toolCallID, toolName: call.toolName))
                for chunk in MockLanguageModel.split(call.input.serialized(sortedKeys: true), by: response.textChunkSize) {
                    continuation.yield(.toolInputDelta(id: call.toolCallID, delta: chunk))
                }
                continuation.yield(.toolInputEnd(id: call.toolCallID))
                continuation.yield(.toolCall(call))

            case .toolResult(let result):
                continuation.yield(.toolResult(result))

            case .file(let file):
                continuation.yield(.file(file))

            case .source(let source):
                continuation.yield(.source(source))
            }
        }

        if let error = response.error {
            continuation.finish(throwing: error)
        } else {
            continuation.yield(
                .finish(
                    finishReason: response.finishReason,
                    usage: response.usage,
                    providerMetadata: response.providerMetadata
                )
            )
            continuation.finish()
        }

        return LanguageModelStreamResponse(
            stream: stream,
            request: RequestInfo(body: "{}"),
            response: ResponseInfo(id: "mock-response", modelID: modelID)
        )
    }

    private static func split(_ text: String, by chunkSize: Int?) -> [String] {
        guard let chunkSize, chunkSize > 0, !text.isEmpty else {
            return text.isEmpty ? [] : [text]
        }
        var chunks: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            let end = text.index(index, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
            chunks.append(String(text[index..<end]))
            index = end
        }
        return chunks
    }
}

/// Failures raised by ``MockLanguageModel`` itself.
public enum MockModelError: Error, CustomStringConvertible {
    /// More calls were made than the test scripted responses for.
    case responsesExhausted(callCount: Int)

    public var description: String {
        switch self {
        case .responsesExhausted(let callCount):
            return """
                MockLanguageModel ran out of scripted responses on call \(callCount). \
                Queue another response, or pass repeatsLastResponse: true.
                """
        }
    }
}

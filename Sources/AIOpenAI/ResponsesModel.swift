import AIProviderSpec
import AIProviderUtils
import Foundation

// MARK: - Wire types

/// A Responses API response.
struct OpenAIResponse: Decodable {
    var id: String?
    var model: String?
    var createdAt: Double?
    var status: String?
    var output: [OutputItem]?
    var usage: TokenUsage?
    var incompleteDetails: IncompleteDetails?
    var error: ErrorPayload?

    struct OutputItem: Decodable {
        var type: String
        var id: String?
        var role: String?
        var content: [ContentPart]?

        // Present on `function_call` items.
        var callId: String?
        var name: String?
        var arguments: String?

        // Present on `reasoning` items.
        var summary: [SummaryPart]?
        var encryptedContent: String?
    }

    struct ContentPart: Decodable {
        var type: String
        var text: String?
        var annotations: [Annotation]?
    }

    struct Annotation: Decodable {
        var type: String?
        var url: String?
        var title: String?
    }

    struct SummaryPart: Decodable {
        var type: String?
        var text: String?
    }

    struct IncompleteDetails: Decodable {
        var reason: String?
    }

    struct ErrorPayload: Decodable {
        var code: String?
        var message: String?
    }

    struct TokenUsage: Decodable {
        var inputTokens: Int?
        var outputTokens: Int?
        var totalTokens: Int?
        var inputTokensDetails: InputDetails?
        var outputTokensDetails: OutputDetails?

        struct InputDetails: Decodable {
            var cachedTokens: Int?
        }

        struct OutputDetails: Decodable {
            var reasoningTokens: Int?
        }

        var normalized: Usage {
            Usage(
                inputTokens: inputTokens,
                outputTokens: outputTokens,
                totalTokens: totalTokens,
                reasoningTokens: outputTokensDetails?.reasoningTokens,
                cachedInputTokens: inputTokensDetails?.cachedTokens
            )
        }
    }
}

extension OpenAIResponse {
    /// The output items, converted to the shared representation.
    func modelContent() -> [ModelContent] {
        var content: [ModelContent] = []

        for item in output ?? [] {
            switch item.type {
            case "reasoning":
                let text = (item.summary ?? []).compactMap(\.text).joined(separator: "\n\n")
                // The identifier and encrypted payload have to travel back on the next turn, or
                // the model loses its chain of thought.
                var options: [String: JSONValue] = [:]
                if let id = item.id { options["itemId"] = .string(id) }
                if let encrypted = item.encryptedContent { options["encryptedContent"] = .string(encrypted) }
                content.append(
                    .reasoning(
                        ReasoningPart(
                            text,
                            providerOptions: options.isEmpty ? nil : ProviderOptions(["openai": options])
                        )
                    )
                )

            case "message":
                for part in item.content ?? [] where part.type == "output_text" {
                    if let text = part.text, !text.isEmpty {
                        content.append(.text(TextPart(text)))
                    }
                    for annotation in part.annotations ?? [] {
                        guard annotation.type == "url_citation",
                              let uri = annotation.url,
                              let url = URL(string: uri)
                        else { continue }
                        content.append(
                            .source(
                                SourcePart(
                                    id: IdentifierGenerator.generate(prefix: "source"),
                                    kind: .url(url),
                                    title: annotation.title
                                )
                            )
                        )
                    }
                }

            case "function_call":
                guard let name = item.name else { continue }
                content.append(
                    .toolCall(
                        ToolCallPart(
                            toolCallID: item.callId ?? item.id ?? IdentifierGenerator.generate(prefix: "call"),
                            toolName: name,
                            input: (try? ProviderJSON.parseEmbeddedJSON(
                                item.arguments ?? "",
                                context: "the arguments for '\(name)'"
                            )) ?? .object([:])
                        )
                    )
                )

            default:
                // Provider-executed tool items such as `web_search_call` carry their results in
                // the following message, so there is nothing to surface separately.
                continue
            }
        }
        return content
    }

    var normalizedFinishReason: FinishReason {
        if output?.contains(where: { $0.type == "function_call" }) == true { return .toolCalls }
        switch status {
        case "completed": return .stop
        case "incomplete":
            switch incompleteDetails?.reason {
            case "max_output_tokens": return .length
            case "content_filter": return .contentFilter
            default: return .other
            }
        case "failed": return .error
        case nil: return .unknown
        default: return .other
        }
    }
}

// MARK: - Streaming events

/// One event from a streamed Responses call.
///
/// The API emits a named event per state transition. Only the ones carrying content are decoded;
/// the rest — `response.created`, `response.in_progress`, part boundaries — are bookkeeping this
/// SDK reconstructs from the deltas themselves.
struct OpenAIStreamEvent: Decodable {
    var type: String
    var itemId: String?
    var outputIndex: Int?
    var contentIndex: Int?
    var delta: String?
    var text: String?
    var arguments: String?
    var item: OpenAIResponse.OutputItem?
    var response: OpenAIResponse?
    var sequenceNumber: Int?
}

/// Turns Responses stream events into spec stream parts.
struct ResponsesStreamDecoder {
    private var openTextBlocks: Set<String> = []
    private var openReasoningBlocks: Set<String> = []
    private var toolNames: [String: String] = [:]
    private var finishReason: FinishReason = .unknown
    private var usage: Usage = .none
    private var sawFunctionCall = false

    mutating func consume(_ event: OpenAIStreamEvent) throws -> [LanguageModelStreamPart] {
        switch event.type {
        case "response.created":
            guard let response = event.response else { return [] }
            return [
                .responseMetadata(
                    id: response.id,
                    modelID: response.model,
                    timestamp: response.createdAt.map { Date(timeIntervalSince1970: $0) }
                )
            ]

        case "response.output_item.added":
            guard let item = event.item else { return [] }
            switch item.type {
            case "function_call":
                guard let name = item.name else { return [] }
                let id = item.callId ?? item.id ?? IdentifierGenerator.generate(prefix: "call")
                sawFunctionCall = true
                // Keyed by the item identifier, which is what the argument deltas reference.
                toolNames[item.id ?? id] = name
                toolCallIDs[item.id ?? id] = id
                return [.toolInputStart(id: id, toolName: name)]
            default:
                return []
            }

        case "response.output_text.delta":
            guard let id = event.itemId, let delta = event.delta else { return [] }
            var parts: [LanguageModelStreamPart] = []
            if openTextBlocks.insert(id).inserted {
                parts.append(.textStart(id: id))
            }
            parts.append(.textDelta(id: id, delta: delta))
            return parts

        case "response.output_text.done":
            guard let id = event.itemId, openTextBlocks.remove(id) != nil else { return [] }
            return [.textEnd(id: id)]

        case "response.reasoning_summary_text.delta":
            guard let id = event.itemId, let delta = event.delta else { return [] }
            var parts: [LanguageModelStreamPart] = []
            if openReasoningBlocks.insert(id).inserted {
                parts.append(.reasoningStart(id: id))
            }
            parts.append(.reasoningDelta(id: id, delta: delta))
            return parts

        case "response.reasoning_summary_text.done":
            guard let id = event.itemId, openReasoningBlocks.remove(id) != nil else { return [] }
            return [.reasoningEnd(id: id)]

        case "response.function_call_arguments.delta":
            guard let itemID = event.itemId,
                  let callID = toolCallIDs[itemID],
                  let delta = event.delta
            else { return [] }
            return [.toolInputDelta(id: callID, delta: delta)]

        case "response.function_call_arguments.done":
            guard let itemID = event.itemId,
                  let callID = toolCallIDs[itemID],
                  let name = toolNames[itemID]
            else { return [] }
            let input = (try? ProviderJSON.parseEmbeddedJSON(
                event.arguments ?? "",
                context: "the arguments for '\(name)'"
            )) ?? .object([:])
            return [
                .toolInputEnd(id: callID),
                .toolCall(ToolCallPart(toolCallID: callID, toolName: name, input: input)),
            ]

        case "response.completed", "response.incomplete", "response.failed":
            guard let response = event.response else { return [] }
            usage = response.usage?.normalized ?? usage
            finishReason = response.normalizedFinishReason
            return []

        case "error":
            throw APICallError(
                message: event.response?.error?.message ?? "The OpenAI stream reported an error.",
                url: URL(string: "https://api.openai.com/v1/responses")!,
                isRetryable: false
            )

        default:
            return []
        }
    }

    private var toolCallIDs: [String: String] = [:]

    /// Closes anything still open and emits the finish part.
    mutating func finish() -> [LanguageModelStreamPart] {
        var parts: [LanguageModelStreamPart] = []
        for id in openReasoningBlocks.sorted() { parts.append(.reasoningEnd(id: id)) }
        for id in openTextBlocks.sorted() { parts.append(.textEnd(id: id)) }
        openReasoningBlocks.removeAll()
        openTextBlocks.removeAll()

        parts.append(
            .finish(
                finishReason: sawFunctionCall && finishReason == .stop ? .toolCalls : finishReason,
                usage: usage
            )
        )
        return parts
    }
}

// MARK: - Model

/// A language model served by OpenAI's Responses API.
///
/// The Responses API is preferred over Chat Completions because it exposes reasoning items —
/// which have to be replayed to preserve a reasoning model's chain of thought across turns — and
/// provider-executed tools such as web search. Use ``OpenAIProvider/chatModel(_:)`` when you need
/// a setting only Chat Completions supports.
public struct OpenAIResponsesModel: LanguageModel {
    public let provider: String
    public let modelID: String

    let client: ProviderHTTPClient

    /// OpenAI fetches image URLs itself.
    public func supportsNativeURL(_ url: URL, mediaType: String) -> Bool {
        url.scheme == "https" && mediaType.hasPrefix("image/")
    }

    public func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        let request = try ResponsesRequestBuilder(modelID: modelID).build(options, stream: false)

        let (response, head, requestBody) = try await client.postJSON(
            path: "responses",
            body: request.body,
            additionalHeaders: options.headers,
            as: OpenAIResponse.self
        )

        // A failed response arrives with a 200 and an error payload inside it.
        if response.status == "failed", let error = response.error {
            throw APICallError(
                message: "openai reported a failed response: \(error.message ?? "no message")",
                url: try client.url(path: "responses"),
                statusCode: head.statusCode,
                responseHeaders: head.headers,
                isRetryable: false
            )
        }

        return LanguageModelResponse(
            content: response.modelContent(),
            finishReason: response.normalizedFinishReason,
            usage: response.usage?.normalized ?? .none,
            warnings: request.warnings,
            request: RequestInfo(body: requestBody),
            response: ResponseInfo(
                id: response.id,
                modelID: response.model,
                timestamp: response.createdAt.map { Date(timeIntervalSince1970: $0) },
                headers: head.headers
            )
        )
    }

    public func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        let request = try ResponsesRequestBuilder(modelID: modelID).build(options, stream: true)

        let (events, head, requestBody) = try await client.postJSONForServerSentEvents(
            path: "responses",
            body: request.body,
            additionalHeaders: options.headers,
            stopAtDoneSentinel: true
        )

        let warnings = request.warnings
        let includesRawChunks = options.includesRawChunks
        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()

        let task = Task {
            var decoder = ResponsesStreamDecoder()
            continuation.yield(.streamStart(warnings: warnings))

            do {
                for try await sseEvent in events {
                    try Task.checkCancellation()
                    guard !sseEvent.data.isEmpty else { continue }

                    if includesRawChunks, let raw = try? JSONValue.parse(sseEvent.data) {
                        continuation.yield(.raw(raw))
                    }

                    let event = try ProviderJSON.decode(
                        OpenAIStreamEvent.self,
                        from: Data(sseEvent.data.utf8),
                        context: "a streamed event",
                        decoder: ProviderJSON.snakeCaseDecoder
                    )
                    for part in try decoder.consume(event) {
                        continuation.yield(part)
                    }
                }
                for part in decoder.finish() {
                    continuation.yield(part)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }

        return LanguageModelStreamResponse(
            stream: stream,
            request: RequestInfo(body: requestBody),
            response: ResponseInfo(headers: head.headers)
        )
    }
}

import AIProviderSpec
import AIProviderUtils
import Foundation

/// Applies default settings that a call did not specify.
///
/// Anything the call sets wins; this only fills gaps. Useful for pinning a house style — a
/// temperature, a token ceiling, a set of provider options — without repeating it at every call
/// site, and without preventing a caller from overriding it when they need to.
///
/// ```swift
/// let model = wrapLanguageModel(
///     model: base,
///     middleware: [DefaultSettingsMiddleware(temperature: 0.2, maxOutputTokens: 2_000)]
/// )
/// ```
public struct DefaultSettingsMiddleware: LanguageModelMiddleware {
    private let defaults: GenerationSettings

    /// Creates middleware from a settings value.
    public init(_ defaults: GenerationSettings) {
        self.defaults = defaults
    }

    /// Creates middleware from the settings most often defaulted.
    public init(
        temperature: Double? = nil,
        maxOutputTokens: Int? = nil,
        topP: Double? = nil,
        providerOptions: ProviderOptions? = nil
    ) {
        self.defaults = GenerationSettings(
            maxOutputTokens: maxOutputTokens,
            temperature: temperature,
            topP: topP,
            providerOptions: providerOptions
        )
    }

    public func transformOptions(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        kind: ModelCallKind
    ) async throws -> LanguageModelCallOptions {
        var adjusted = options
        adjusted.temperature = options.temperature ?? defaults.temperature
        adjusted.maxOutputTokens = options.maxOutputTokens ?? defaults.maxOutputTokens
        adjusted.topP = options.topP ?? defaults.topP
        adjusted.topK = options.topK ?? defaults.topK
        adjusted.presencePenalty = options.presencePenalty ?? defaults.presencePenalty
        adjusted.frequencyPenalty = options.frequencyPenalty ?? defaults.frequencyPenalty
        adjusted.seed = options.seed ?? defaults.seed
        if options.stopSequences.isEmpty { adjusted.stopSequences = defaults.stopSequences }
        adjusted.headers = defaults.headers.merging(options.headers) { _, callSite in callSite }
        // Defaults sit underneath, so a per-call provider option of the same name wins.
        adjusted.providerOptions = (defaults.providerOptions ?? ProviderOptions())
            .merging(options.providerOptions ?? ProviderOptions())
        return adjusted
    }
}

/// Turns a model's buffered output into a stream.
///
/// Some models — and some middleware — have no real streaming implementation, but callers may
/// still want to use ``streamText(model:system:prompt:tools:toolChoice:settings:stopWhen:prepareStep:onStepFinish:onFinish:onError:)``
/// uniformly. This produces a well-formed stream from a single buffered call, so the calling code
/// does not have to branch.
///
/// The output does not arrive any sooner: everything is emitted at once when the call completes.
public struct SimulateStreamingMiddleware: LanguageModelMiddleware {
    public init() {}

    public func stream(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelStreamResponse
    ) async throws -> LanguageModelStreamResponse {
        let response = try await model.generate(options)

        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()
        continuation.yield(.streamStart(warnings: response.warnings))
        if let info = response.response {
            continuation.yield(.responseMetadata(id: info.id, modelID: info.modelID, timestamp: info.timestamp))
        }

        for (index, part) in response.content.enumerated() {
            let blockID = "simulated_\(index)"
            switch part {
            case .text(let text):
                continuation.yield(.textStart(id: blockID))
                continuation.yield(.textDelta(id: blockID, delta: text.text))
                continuation.yield(.textEnd(id: blockID))
            case .reasoning(let reasoning):
                continuation.yield(.reasoningStart(id: blockID))
                continuation.yield(.reasoningDelta(id: blockID, delta: reasoning.text))
                continuation.yield(.reasoningEnd(id: blockID))
            case .toolCall(let call):
                continuation.yield(.toolInputStart(id: call.toolCallID, toolName: call.toolName))
                continuation.yield(.toolInputDelta(id: call.toolCallID, delta: call.input.serialized()))
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

        continuation.yield(
            .finish(
                finishReason: response.finishReason,
                usage: response.usage,
                providerMetadata: response.providerMetadata
            )
        )
        continuation.finish()

        return LanguageModelStreamResponse(
            stream: stream,
            request: response.request,
            response: response.response
        )
    }
}

/// Lifts inline reasoning tags out of a model's text and into proper reasoning parts.
///
/// Several models — particularly open-weight ones served through OpenAI-compatible endpoints —
/// emit their chain of thought inside `<think>…</think>` in the ordinary text channel rather than
/// through a dedicated reasoning field. Without this, that text lands in
/// ``GenerateTextResult/text`` and gets shown to users.
///
/// ```swift
/// let model = wrapLanguageModel(
///     model: ollama.languageModel("deepseek-r1"),
///     middleware: [ExtractReasoningMiddleware()]
/// )
/// let result = try await generateText(model: model, prompt: "…")
/// result.text            // the answer only
/// result.reasoningText   // what was inside the tags
/// ```
///
/// Works for both buffered and streaming calls. When streaming, text is held back only as far as
/// needed to recognize a tag that may be split across deltas.
public struct ExtractReasoningMiddleware: LanguageModelMiddleware {
    /// The tag name enclosing the reasoning, without angle brackets.
    public let tagName: String

    /// Whether the response is assumed to begin inside the tag.
    ///
    /// Some models emit a closing tag without ever having emitted an opening one, because the
    /// prompt template already opened it.
    public let startsInsideReasoning: Bool

    public init(tagName: String = "think", startsInsideReasoning: Bool = false) {
        self.tagName = tagName
        self.startsInsideReasoning = startsInsideReasoning
    }

    private var openingTag: String { "<\(tagName)>" }
    private var closingTag: String { "</\(tagName)>" }

    // MARK: Buffered

    public func generate(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse
    ) async throws -> LanguageModelResponse {
        var response = try await next(options)
        response.content = response.content.flatMap { part -> [ModelContent] in
            guard case .text(let text) = part else { return [part] }
            return split(text.text).map { piece in
                piece.isReasoning ? .reasoning(ReasoningPart(piece.text)) : .text(TextPart(piece.text))
            }
        }
        return response
    }

    /// Splits text into alternating answer and reasoning runs.
    private func split(_ text: String) -> [(text: String, isReasoning: Bool)] {
        var pieces: [(String, Bool)] = []
        var remainder = Substring(text)
        var isReasoning = startsInsideReasoning

        while !remainder.isEmpty {
            let marker = isReasoning ? closingTag : openingTag
            guard let range = remainder.range(of: marker) else {
                pieces.append((String(remainder), isReasoning))
                break
            }
            let before = remainder[remainder.startIndex..<range.lowerBound]
            if !before.isEmpty { pieces.append((String(before), isReasoning)) }
            remainder = remainder[range.upperBound...]
            isReasoning.toggle()
        }

        // Tags are usually followed by a newline that would otherwise appear as leading blank
        // space in the answer.
        return pieces
            .map { (text: $0.0.trimmingCharacters(in: .newlines), isReasoning: $0.1) }
            .filter { !$0.text.isEmpty }
    }

    // MARK: Streaming

    public func stream(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelStreamResponse
    ) async throws -> LanguageModelStreamResponse {
        let upstream = try await next(options)
        let openingTag = openingTag
        let closingTag = closingTag
        let startsInsideReasoning = startsInsideReasoning

        let (stream, continuation) = AsyncThrowingStream<LanguageModelStreamPart, any Error>.makeStream()
        let task = Task {
            var splitter = ReasoningTagSplitter(
                openingTag: openingTag,
                closingTag: closingTag,
                isReasoning: startsInsideReasoning
            )
            var reasoningBlockIsOpen = false

            do {
                for try await part in upstream.stream {
                    guard case .textDelta(let id, let delta) = part else {
                        continuation.yield(part)
                        continue
                    }
                    for piece in splitter.consume(delta) {
                        emit(piece, id: id, into: continuation, reasoningIsOpen: &reasoningBlockIsOpen)
                    }
                }
                // Anything held back waiting for a tag that never arrived is still real output.
                for piece in splitter.flush() {
                    emit(piece, id: "reasoning_flush", into: continuation, reasoningIsOpen: &reasoningBlockIsOpen)
                }
                if reasoningBlockIsOpen { continuation.yield(.reasoningEnd(id: "reasoning")) }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }

        return LanguageModelStreamResponse(
            stream: stream,
            request: upstream.request,
            response: upstream.response
        )
    }

    private func emit(
        _ piece: ReasoningTagSplitter.Piece,
        id: String,
        into continuation: AsyncThrowingStream<LanguageModelStreamPart, any Error>.Continuation,
        reasoningIsOpen: inout Bool
    ) {
        if piece.isReasoning {
            if !reasoningIsOpen {
                continuation.yield(.reasoningStart(id: "reasoning"))
                reasoningIsOpen = true
            }
            continuation.yield(.reasoningDelta(id: "reasoning", delta: piece.text))
        } else {
            if reasoningIsOpen {
                continuation.yield(.reasoningEnd(id: "reasoning"))
                reasoningIsOpen = false
            }
            continuation.yield(.textDelta(id: id, delta: piece.text))
        }
    }
}

/// Splits a stream of text deltas around reasoning tags.
///
/// The difficulty is that a tag can be split across deltas: `<th` in one chunk and `ink>` in the
/// next. Any trailing text that could still turn out to be the start of a tag is held back until
/// the next delta resolves it, so a tag is never emitted as visible text and text is never
/// delayed longer than necessary.
struct ReasoningTagSplitter {
    struct Piece {
        var text: String
        var isReasoning: Bool
    }

    let openingTag: String
    let closingTag: String
    private(set) var isReasoning: Bool
    private var buffer = ""

    init(openingTag: String, closingTag: String, isReasoning: Bool) {
        self.openingTag = openingTag
        self.closingTag = closingTag
        self.isReasoning = isReasoning
    }

    mutating func consume(_ delta: String) -> [Piece] {
        buffer += delta
        var pieces: [Piece] = []

        while true {
            let marker = isReasoning ? closingTag : openingTag
            if let range = buffer.range(of: marker) {
                let before = String(buffer[buffer.startIndex..<range.lowerBound])
                if !before.isEmpty { pieces.append(Piece(text: before, isReasoning: isReasoning)) }
                buffer = String(buffer[range.upperBound...])
                isReasoning.toggle()
                continue
            }

            // Emit everything that cannot be the beginning of the marker, and keep the rest.
            let safeLength = buffer.count - longestSuffixThatCouldStart(marker, of: buffer)
            if safeLength > 0 {
                let index = buffer.index(buffer.startIndex, offsetBy: safeLength)
                pieces.append(Piece(text: String(buffer[buffer.startIndex..<index]), isReasoning: isReasoning))
                buffer = String(buffer[index...])
            }
            break
        }
        return pieces
    }

    /// Emits whatever is left, once no more input is coming.
    mutating func flush() -> [Piece] {
        defer { buffer = "" }
        guard !buffer.isEmpty else { return [] }
        return [Piece(text: buffer, isReasoning: isReasoning)]
    }

    /// The length of the longest suffix of `text` that is a prefix of `marker`.
    private func longestSuffixThatCouldStart(_ marker: String, of text: String) -> Int {
        let maximum = min(marker.count - 1, text.count)
        guard maximum > 0 else { return 0 }
        for length in stride(from: maximum, through: 1, by: -1) {
            if marker.hasPrefix(text.suffix(length)) { return length }
        }
        return 0
    }
}

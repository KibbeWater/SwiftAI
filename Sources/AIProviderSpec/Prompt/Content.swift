import Foundation

// MARK: - Text

/// A run of plain text.
public struct TextPart: Sendable, Hashable {
    /// The text itself.
    public var text: String

    /// Provider-specific settings scoped to this part.
    public var providerOptions: ProviderOptions?

    public init(_ text: String, providerOptions: ProviderOptions? = nil) {
        self.text = text
        self.providerOptions = providerOptions
    }
}

extension TextPart: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self.init(value) }
}

// MARK: - Reasoning

/// Intermediate reasoning produced by a model before its answer.
///
/// Reasoning is exposed separately from the answer so applications can choose to display, hide,
/// or persist it. Some providers require that reasoning be replayed verbatim — including its
/// ``signature`` — when continuing a conversation, so preserve these parts in message history
/// rather than discarding them.
public struct ReasoningPart: Sendable, Hashable {
    /// The reasoning text. Empty when the provider returned only an opaque, redacted block.
    public var text: String

    /// A provider-issued signature attesting to the reasoning block.
    ///
    /// Anthropic returns one and rejects continuations where it is missing or altered.
    public var signature: String?

    /// Provider-specific settings or values scoped to this part.
    public var providerOptions: ProviderOptions?

    public init(_ text: String, signature: String? = nil, providerOptions: ProviderOptions? = nil) {
        self.text = text
        self.signature = signature
        self.providerOptions = providerOptions
    }
}

// MARK: - Files

/// Binary or remote content attached to a message: an image, a document, an audio clip.
///
/// Content is carried either inline as bytes or as a URL. Whether a URL reaches the provider
/// untouched depends on ``LanguageModel/supportedURLPatterns``; when a provider cannot fetch a
/// URL itself, the SDK downloads it and substitutes the bytes.
public struct FilePart: Sendable, Hashable {
    /// Where the file's bytes come from.
    public enum Source: Sendable, Hashable {
        /// Bytes carried inline. Providers encode these as base64 or multipart as required.
        case data(Data)
        /// A location the provider may fetch directly.
        case url(URL)
    }

    public var source: Source

    /// The IANA media type, such as `image/png` or `application/pdf`.
    ///
    /// Required: providers cannot reliably infer it, and several reject requests without it.
    public var mediaType: String

    /// An optional display name, used by providers that surface documents by name.
    public var filename: String?

    /// Provider-specific settings scoped to this part.
    public var providerOptions: ProviderOptions?

    public init(
        source: Source,
        mediaType: String,
        filename: String? = nil,
        providerOptions: ProviderOptions? = nil
    ) {
        self.source = source
        self.mediaType = mediaType
        self.filename = filename
        self.providerOptions = providerOptions
    }

    /// Creates a part from inline bytes.
    public static func data(
        _ data: Data,
        mediaType: String,
        filename: String? = nil,
        providerOptions: ProviderOptions? = nil
    ) -> FilePart {
        FilePart(source: .data(data), mediaType: mediaType, filename: filename, providerOptions: providerOptions)
    }

    /// Creates a part referring to a remote location.
    public static func url(
        _ url: URL,
        mediaType: String,
        filename: String? = nil,
        providerOptions: ProviderOptions? = nil
    ) -> FilePart {
        FilePart(source: .url(url), mediaType: mediaType, filename: filename, providerOptions: providerOptions)
    }

    /// The inline bytes, or `nil` when the part refers to a URL.
    public var data: Data? {
        guard case .data(let data) = source else { return nil }
        return data
    }

    /// The referenced URL, or `nil` when the part carries inline bytes.
    public var url: URL? {
        guard case .url(let url) = source else { return nil }
        return url
    }

    /// The inline bytes as a base64 string, or `nil` when the part refers to a URL.
    public var base64EncodedString: String? { data?.base64EncodedString() }

    /// A `data:` URI for the inline bytes, which several providers accept in place of a URL.
    public var dataURI: String? {
        guard let base64EncodedString else { return nil }
        return "data:\(mediaType);base64,\(base64EncodedString)"
    }

    /// Whether the media type denotes an image.
    public var isImage: Bool { mediaType.hasPrefix("image/") }
}

// MARK: - Sources

/// A citation the model attributed part of its answer to.
///
/// Produced by providers with built-in retrieval, such as web search or document grounding.
public struct SourcePart: Sendable, Hashable {
    /// What kind of source is being cited.
    public enum Kind: Sendable, Hashable {
        /// A web page.
        case url(URL)
        /// A document supplied to the model, identified by media type and optional filename.
        case document(mediaType: String, filename: String?)
    }

    /// A provider-assigned identifier, unique within the response.
    public var id: String

    public var kind: Kind

    /// A human-readable title for the source.
    public var title: String?

    /// Provider-specific values scoped to this source, such as relevance scores.
    public var providerMetadata: ProviderMetadata?

    public init(id: String, kind: Kind, title: String? = nil, providerMetadata: ProviderMetadata? = nil) {
        self.id = id
        self.kind = kind
        self.title = title
        self.providerMetadata = providerMetadata
    }
}

// MARK: - Tool calls and results

/// A model's request to invoke a tool.
public struct ToolCallPart: Sendable, Hashable {
    /// The provider-assigned identifier correlating this call with its result.
    public var toolCallID: String

    /// The name of the tool the model chose.
    public var toolName: String

    /// The arguments, decoded from the JSON the model produced.
    public var input: JSONValue

    /// Whether the provider executed this tool itself.
    ///
    /// Provider-executed tools — web search, code interpreters — arrive already resolved, with a
    /// matching ``ToolResultPart``. The SDK must not attempt to run them locally.
    public var providerExecuted: Bool

    /// Whether the tool was supplied without a compile-time Swift type.
    ///
    /// Dynamic calls carry untyped input, so callers must inspect ``input`` directly rather than
    /// decoding it into a known `Arguments` type.
    public var isDynamic: Bool

    /// Provider-specific settings or values scoped to this call.
    public var providerOptions: ProviderOptions?

    public init(
        toolCallID: String,
        toolName: String,
        input: JSONValue,
        providerExecuted: Bool = false,
        isDynamic: Bool = false,
        providerOptions: ProviderOptions? = nil
    ) {
        self.toolCallID = toolCallID
        self.toolName = toolName
        self.input = input
        self.providerExecuted = providerExecuted
        self.isDynamic = isDynamic
        self.providerOptions = providerOptions
    }
}

/// What a tool returned, in a form the model can consume.
///
/// The distinction between success and failure matters: models respond differently when they can
/// see that a tool errored, and providers that support it mark error results specially.
public enum ToolResultOutput: Sendable, Hashable {
    /// A plain text result.
    case text(String)

    /// A structured result, serialized to JSON for the model.
    case json(JSONValue)

    /// A failure, described in text.
    case errorText(String)

    /// A structured failure.
    case errorJSON(JSONValue)

    /// A multimodal result, such as a screenshot returned by a browser tool.
    ///
    /// Not every provider accepts non-text tool results; those that do not receive a textual
    /// placeholder plus a ``CallWarning``.
    case content([UserContent])

    /// Whether the output represents a failure.
    public var isError: Bool {
        switch self {
        case .errorText, .errorJSON: return true
        case .text, .json, .content: return false
        }
    }
}

/// The outcome of a tool invocation, paired with the call that produced it.
public struct ToolResultPart: Sendable, Hashable {
    /// The identifier of the ``ToolCallPart`` this result answers.
    public var toolCallID: String

    /// The name of the tool that ran.
    public var toolName: String

    /// What the tool returned.
    public var output: ToolResultOutput

    /// Whether the provider produced this result itself rather than the SDK running a local tool.
    public var providerExecuted: Bool

    /// Whether the originating call was dynamic.
    public var isDynamic: Bool

    /// Provider-specific settings or values scoped to this result.
    public var providerOptions: ProviderOptions?

    public init(
        toolCallID: String,
        toolName: String,
        output: ToolResultOutput,
        providerExecuted: Bool = false,
        isDynamic: Bool = false,
        providerOptions: ProviderOptions? = nil
    ) {
        self.toolCallID = toolCallID
        self.toolName = toolName
        self.output = output
        self.providerExecuted = providerExecuted
        self.isDynamic = isDynamic
        self.providerOptions = providerOptions
    }
}

// MARK: - Content collections

/// Content a user may send to a model.
public enum UserContent: Sendable, Hashable {
    case text(TextPart)
    case file(FilePart)

    /// Convenience for plain text.
    public static func text(_ text: String) -> UserContent { .text(TextPart(text)) }
}

extension UserContent: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .text(TextPart(value)) }
}

/// Content a model may produce, and which may be replayed back to it as assistant history.
///
/// The same type serves both roles so that a result can be appended to a conversation without
/// conversion. ``ModelContent/source(_:)`` parts are informational: providers drop them when
/// serializing history, since no provider accepts citations as input.
public enum ModelContent: Sendable, Hashable {
    case text(TextPart)
    case reasoning(ReasoningPart)
    case file(FilePart)
    case source(SourcePart)
    case toolCall(ToolCallPart)
    case toolResult(ToolResultPart)

    /// Convenience for plain text.
    public static func text(_ text: String) -> ModelContent { .text(TextPart(text)) }

    /// The text of this part, when it is a text part.
    public var text: String? {
        guard case .text(let part) = self else { return nil }
        return part.text
    }

    /// The tool call, when this part is one.
    public var toolCall: ToolCallPart? {
        guard case .toolCall(let part) = self else { return nil }
        return part
    }

    /// The tool result, when this part is one.
    public var toolResult: ToolResultPart? {
        guard case .toolResult(let part) = self else { return nil }
        return part
    }
}

extension ModelContent: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .text(TextPart(value)) }
}

extension Array where Element == ModelContent {
    /// The concatenated text of every text part, in order.
    ///
    /// Reasoning, files, and tool activity are excluded — this is the answer as the user would
    /// read it.
    public var text: String {
        compactMap(\.text).joined()
    }

    /// The concatenated text of every reasoning part, in order.
    public var reasoningText: String? {
        let parts = compactMap { part -> String? in
            guard case .reasoning(let reasoning) = part else { return nil }
            return reasoning.text
        }
        return parts.isEmpty ? nil : parts.joined()
    }

    /// Every tool call, in order.
    public var toolCalls: [ToolCallPart] { compactMap(\.toolCall) }

    /// Every tool result, in order.
    public var toolResults: [ToolResultPart] { compactMap(\.toolResult) }

    /// Every file part, in order.
    public var files: [FilePart] {
        compactMap { part in
            guard case .file(let file) = part else { return nil }
            return file
        }
    }

    /// Every cited source, in order.
    public var sources: [SourcePart] {
        compactMap { part in
            guard case .source(let source) = part else { return nil }
            return source
        }
    }
}

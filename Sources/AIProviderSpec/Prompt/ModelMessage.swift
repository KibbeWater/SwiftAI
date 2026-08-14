/// A message in the normalized conversation format that every provider receives.
///
/// This is the SDK's lingua franca for prompts. Application code builds messages once, and each
/// provider translates them into its own wire format. Construct them with the static helpers,
/// which accept either a plain string or an array of content parts:
///
/// ```swift
/// let messages: [ModelMessage] = [
///     .system("You answer in one sentence."),
///     .user("What is in this picture?"),
///     .user([.file(.data(imageBytes, mediaType: "image/png"))]),
/// ]
/// ```
///
/// Assistant and tool messages usually come from a previous result rather than being written by
/// hand — append `result.response.messages` to continue a conversation, which preserves reasoning
/// signatures and tool call identifiers that providers require.
public enum ModelMessage: Sendable, Hashable {
    /// Instructions that frame the conversation.
    case system(SystemMessage)

    /// Input from the user.
    case user(UserMessage)

    /// Output from the model.
    case assistant(AssistantMessage)

    /// Results of tools the model asked to run.
    case tool(ToolMessage)
}

// MARK: - Message payloads

/// Instructions that frame the conversation.
public struct SystemMessage: Sendable, Hashable {
    public var content: String
    public var providerOptions: ProviderOptions?

    public init(_ content: String, providerOptions: ProviderOptions? = nil) {
        self.content = content
        self.providerOptions = providerOptions
    }
}

/// Input from the user, as one or more content parts.
public struct UserMessage: Sendable, Hashable {
    public var content: [UserContent]
    public var providerOptions: ProviderOptions?

    public init(_ content: [UserContent], providerOptions: ProviderOptions? = nil) {
        self.content = content
        self.providerOptions = providerOptions
    }

    public init(_ text: String, providerOptions: ProviderOptions? = nil) {
        self.init([.text(TextPart(text))], providerOptions: providerOptions)
    }

    /// The concatenated text of every text part.
    public var text: String {
        content.compactMap { part in
            guard case .text(let text) = part else { return nil }
            return text.text
        }.joined()
    }
}

/// Output from the model, as one or more content parts.
public struct AssistantMessage: Sendable, Hashable {
    public var content: [ModelContent]
    public var providerOptions: ProviderOptions?

    public init(_ content: [ModelContent], providerOptions: ProviderOptions? = nil) {
        self.content = content
        self.providerOptions = providerOptions
    }

    public init(_ text: String, providerOptions: ProviderOptions? = nil) {
        self.init([.text(TextPart(text))], providerOptions: providerOptions)
    }

    /// The concatenated text of every text part.
    public var text: String { content.text }
}

/// Results of tools the model asked to run, returned to it for the next step.
public struct ToolMessage: Sendable, Hashable {
    public var content: [ToolResultPart]
    public var providerOptions: ProviderOptions?

    public init(_ content: [ToolResultPart], providerOptions: ProviderOptions? = nil) {
        self.content = content
        self.providerOptions = providerOptions
    }
}

// MARK: - Construction

extension ModelMessage {
    /// A system message.
    public static func system(_ content: String, providerOptions: ProviderOptions? = nil) -> ModelMessage {
        .system(SystemMessage(content, providerOptions: providerOptions))
    }

    /// A user message containing plain text.
    public static func user(_ text: String, providerOptions: ProviderOptions? = nil) -> ModelMessage {
        .user(UserMessage(text, providerOptions: providerOptions))
    }

    /// A user message containing arbitrary content parts.
    public static func user(_ content: [UserContent], providerOptions: ProviderOptions? = nil) -> ModelMessage {
        .user(UserMessage(content, providerOptions: providerOptions))
    }

    /// An assistant message containing plain text.
    public static func assistant(_ text: String, providerOptions: ProviderOptions? = nil) -> ModelMessage {
        .assistant(AssistantMessage(text, providerOptions: providerOptions))
    }

    /// An assistant message containing arbitrary content parts.
    public static func assistant(_ content: [ModelContent], providerOptions: ProviderOptions? = nil) -> ModelMessage {
        .assistant(AssistantMessage(content, providerOptions: providerOptions))
    }

    /// A tool message carrying results back to the model.
    public static func tool(_ results: [ToolResultPart], providerOptions: ProviderOptions? = nil) -> ModelMessage {
        .tool(ToolMessage(results, providerOptions: providerOptions))
    }
}

// MARK: - Inspection

extension ModelMessage {
    /// The role of the message, as providers name it.
    public enum Role: String, Sendable, Hashable, CaseIterable {
        case system, user, assistant, tool
    }

    public var role: Role {
        switch self {
        case .system: return .system
        case .user: return .user
        case .assistant: return .assistant
        case .tool: return .tool
        }
    }

    /// Options attached to the whole message, if any.
    public var providerOptions: ProviderOptions? {
        switch self {
        case .system(let message): return message.providerOptions
        case .user(let message): return message.providerOptions
        case .assistant(let message): return message.providerOptions
        case .tool(let message): return message.providerOptions
        }
    }

    /// The text of the message, ignoring non-text content.
    ///
    /// Useful for logging and for building simple transcripts; do not use it to reconstruct a
    /// prompt, since it discards files, reasoning, and tool activity.
    public var text: String {
        switch self {
        case .system(let message): return message.content
        case .user(let message): return message.text
        case .assistant(let message): return message.text
        case .tool: return ""
        }
    }
}

extension ModelMessage: ExpressibleByStringLiteral {
    /// Creates a user message. Strings appear as user input far more often than any other role.
    public init(stringLiteral value: String) {
        self = .user(value)
    }
}

import AIProviderSpec

/// What to send the model.
///
/// A prompt is a conversation. The common case is a single question, so a string literal works
/// directly:
///
/// ```swift
/// let result = try await generateText(model: model, prompt: "Why is the sky blue?")
/// ```
///
/// A continuing conversation is an array of messages, which is what a previous result's
/// ``GenerateTextResult/responseMessages`` produces:
///
/// ```swift
/// var history: [ModelMessage] = [.user("Why is the sky blue?")]
/// let first = try await generateText(model: model, prompt: Prompt(history))
/// history += first.responseMessages
/// history.append(.user("And at sunset?"))
/// let second = try await generateText(model: model, prompt: Prompt(history))
/// ```
///
/// A system message is passed separately, as the `system` argument, rather than being written
/// into the array. Providers place system instructions differently — some as a top-level field,
/// some as the first message — and keeping it separate lets each do the right thing.
public struct Prompt: Sendable, Hashable {
    /// The conversation, in order.
    public var messages: [ModelMessage]

    /// Creates a prompt from a conversation.
    public init(_ messages: [ModelMessage]) {
        self.messages = messages
    }

    /// Creates a prompt containing a single user message.
    public init(_ text: String) {
        self.messages = [.user(text)]
    }

    /// Creates a prompt containing a single user message with mixed content.
    ///
    /// Use this to attach images or documents to a question:
    ///
    /// ```swift
    /// Prompt([
    ///     .text("What is in this picture?"),
    ///     .file(.data(imageBytes, mediaType: "image/png")),
    /// ])
    /// ```
    public init(_ content: [UserContent]) {
        self.messages = [.user(content)]
    }

    /// Whether the prompt has no messages.
    public var isEmpty: Bool { messages.isEmpty }
}

extension Prompt: ExpressibleByStringLiteral, ExpressibleByStringInterpolation {
    public init(stringLiteral value: String) {
        self.init(value)
    }
}

extension Prompt: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: ModelMessage...) {
        self.init(elements)
    }
}

// MARK: - Normalization

extension Prompt {
    /// Produces the message list sent to a provider.
    ///
    /// - Parameter system: Instructions to prepend as a system message.
    /// - Returns: The complete conversation.
    /// - Throws: ``InvalidPromptError`` when the prompt has no messages, or when a system message
    ///   appears anywhere but first.
    func resolved(system: String?) throws -> [ModelMessage] {
        var resolved = messages

        if let system, !system.isEmpty {
            // A system message supplied both ways is a genuine ambiguity rather than something to
            // silently merge, so it is reported instead.
            if case .system = resolved.first {
                throw InvalidPromptError(
                    message: """
                        The prompt already begins with a system message, and a 'system' argument \
                        was also supplied. Use one or the other.
                        """
                )
            }
            resolved.insert(.system(system), at: 0)
        }

        guard !resolved.isEmpty else {
            throw InvalidPromptError(message: "A prompt must contain at least one message.")
        }

        // Providers place system instructions in a dedicated field, so one appearing mid-stream
        // would be silently dropped or reordered. Better to say so.
        for (index, message) in resolved.enumerated() where index > 0 {
            if case .system = message {
                throw InvalidPromptError(
                    message: """
                        A system message may only appear first. Found one at position \(index). \
                        Pass instructions as the 'system' argument instead.
                        """
                )
            }
        }

        return resolved
    }
}

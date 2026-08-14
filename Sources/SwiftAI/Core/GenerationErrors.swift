import AIProviderSpec
import Foundation

/// The model asked for a tool that was not supplied.
///
/// Usually means the model hallucinated a tool name, which happens most often when two tools have
/// similar descriptions. It can also mean a tool was removed between turns while the conversation
/// history still refers to it.
public struct NoSuchToolError: AISDKError, LocalizedError {
    public var name: String { "AI_NoSuchToolError" }
    public var message: String

    /// The name the model asked for.
    public var toolName: String

    /// The tools that were available.
    public var availableTools: [String]

    public init(toolName: String, availableTools: [String]) {
        self.toolName = toolName
        self.availableTools = availableTools.sorted()
        self.message = availableTools.isEmpty
            ? "The model called '\(toolName)', but no tools were supplied."
            : """
                The model called '\(toolName)', which is not among the available tools: \
                \(self.availableTools.joined(separator: ", ")).
                """
    }
}

/// The arguments a model produced for a tool did not match its schema.
///
/// The loop reports this back to the model as a tool error rather than failing the generation, so
/// the model gets a chance to correct itself. It surfaces as a thrown error only when a tool is
/// invoked directly.
public struct InvalidToolInputError: AISDKError, LocalizedError {
    public var name: String { "AI_InvalidToolInputError" }
    public var message: String

    /// The tool that was called.
    public var toolName: String

    /// The arguments the model produced.
    public var rawInput: JSONValue

    /// The validation failure underneath.
    public var cause: any Error

    public init(toolName: String, rawInput: JSONValue, cause: any Error) {
        self.toolName = toolName
        self.rawInput = rawInput
        self.cause = cause
        self.message = "The arguments for '\(toolName)' did not match its schema: \(cause)"
    }
}

/// The model did not produce an object matching the requested schema.
///
/// The text the model produced is preserved, which is usually enough to see what went wrong: a
/// refusal, a truncated response, or JSON wrapped in prose.
public struct NoObjectGeneratedError: AISDKError, LocalizedError {
    public var name: String { "AI_NoObjectGeneratedError" }
    public var message: String

    /// What the model actually produced.
    public var text: String?

    /// Why generation stopped. ``FinishReason/length`` means the object was cut off and a larger
    /// `maxOutputTokens` may be all that is needed.
    public var finishReason: FinishReason?

    /// Token usage for the failed attempt, which is billed regardless.
    public var usage: Usage?

    /// The parsing or validation failure underneath.
    public var cause: (any Error)?

    /// Metadata about the response, for support requests.
    public var response: ResponseInfo?

    public init(
        message: String? = nil,
        text: String? = nil,
        finishReason: FinishReason? = nil,
        usage: Usage? = nil,
        cause: (any Error)? = nil,
        response: ResponseInfo? = nil
    ) {
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
        self.cause = cause
        self.response = response
        self.message = message ?? NoObjectGeneratedError.describe(finishReason: finishReason, cause: cause)
    }

    private static func describe(finishReason: FinishReason?, cause: (any Error)?) -> String {
        switch finishReason {
        case .length:
            return """
                The model ran out of output tokens before completing the object. Raise \
                'maxOutputTokens', or ask for a smaller structure.
                """
        case .contentFilter:
            return "The provider's safety systems stopped generation before an object was produced."
        default:
            guard let cause else { return "The model did not produce an object matching the schema." }
            return "The model's output did not match the schema: \(cause)"
        }
    }

    public var description: String {
        var description = "\(name): \(message)"
        if let text, !text.isEmpty {
            description += "\nModel output: \(text.prefix(2048))"
        }
        return description
    }
}

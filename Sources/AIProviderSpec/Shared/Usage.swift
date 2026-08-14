/// Token counts reported by a model for a single call.
///
/// Every field is optional because providers differ in what they report, and some omit counts
/// entirely when streaming. Treat `nil` as "not reported" rather than zero.
public struct Usage: Sendable, Hashable {
    /// Tokens consumed by the prompt.
    public var inputTokens: Int?

    /// Tokens produced by the model, including reasoning tokens where the provider bills them
    /// as output.
    public var outputTokens: Int?

    /// The total billed tokens, when the provider reports one directly.
    ///
    /// Prefer ``resolvedTotalTokens`` for display, which falls back to summing the input and
    /// output counts.
    public var totalTokens: Int?

    /// Tokens spent on reasoning that was not returned to the caller.
    ///
    /// Reported by models with extended thinking, such as OpenAI's reasoning models and
    /// Anthropic's models with thinking enabled.
    public var reasoningTokens: Int?

    /// Input tokens served from the provider's prompt cache, which are usually billed at a
    /// reduced rate.
    public var cachedInputTokens: Int?

    public init(
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        totalTokens: Int? = nil,
        reasoningTokens: Int? = nil,
        cachedInputTokens: Int? = nil
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.totalTokens = totalTokens
        self.reasoningTokens = reasoningTokens
        self.cachedInputTokens = cachedInputTokens
    }

    /// A usage record with nothing reported.
    public static let none = Usage()

    /// The total token count, falling back to the sum of input and output when the provider did
    /// not report a total.
    public var resolvedTotalTokens: Int? {
        if let totalTokens { return totalTokens }
        switch (inputTokens, outputTokens) {
        case (let input?, let output?): return input + output
        case (let input?, nil): return input
        case (nil, let output?): return output
        case (nil, nil): return nil
        }
    }

    /// Returns the field-wise sum of two usage records.
    ///
    /// A field that neither record reports stays `nil`. A field reported by only one of the two
    /// is carried through, treating the unreported side as zero — this keeps multi-step totals
    /// useful when a provider omits a count on some steps, at the cost of a total that may
    /// undercount rather than being absent entirely.
    public func adding(_ other: Usage) -> Usage {
        Usage(
            inputTokens: Usage.sum(inputTokens, other.inputTokens),
            outputTokens: Usage.sum(outputTokens, other.outputTokens),
            totalTokens: Usage.sum(totalTokens, other.totalTokens),
            reasoningTokens: Usage.sum(reasoningTokens, other.reasoningTokens),
            cachedInputTokens: Usage.sum(cachedInputTokens, other.cachedInputTokens)
        )
    }

    private static func sum(_ lhs: Int?, _ rhs: Int?) -> Int? {
        switch (lhs, rhs) {
        case (nil, nil): return nil
        case (let lhs?, let rhs?): return lhs + rhs
        case (let value?, nil), (nil, let value?): return value
        }
    }

    /// Adds `rhs` into `lhs` field-wise. See ``adding(_:)`` for how missing counts are handled.
    public static func += (lhs: inout Usage, rhs: Usage) {
        lhs = lhs.adding(rhs)
    }
}

extension Usage: CustomStringConvertible {
    public var description: String {
        var parts: [String] = []
        if let inputTokens { parts.append("in: \(inputTokens)") }
        if let outputTokens { parts.append("out: \(outputTokens)") }
        if let reasoningTokens { parts.append("reasoning: \(reasoningTokens)") }
        if let cachedInputTokens { parts.append("cached: \(cachedInputTokens)") }
        if let total = resolvedTotalTokens { parts.append("total: \(total)") }
        return parts.isEmpty ? "Usage(not reported)" : "Usage(\(parts.joined(separator: ", ")))"
    }
}

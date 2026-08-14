import AIProviderSpec
import AIProviderUtils

/// The knobs common to every generation call.
///
/// These are collected into one value rather than spread across a dozen parameters so that a
/// configuration can be defined once and reused, and so adding a setting does not change every
/// call site.
///
/// ```swift
/// let precise = GenerationSettings(temperature: 0, maxOutputTokens: 500)
/// let result = try await generateText(model: model, prompt: "…", settings: precise)
/// ```
///
/// Every field is optional. An omitted setting is left out of the request entirely, so the
/// provider's own default applies — which is usually what you want, since defaults differ between
/// models and hard-coding one provider's would be wrong for the others.
public struct GenerationSettings: Sendable {
    /// The maximum number of tokens to generate.
    ///
    /// Leaving this unset lets the model run to its own limit. Setting it too low truncates the
    /// answer, which surfaces as ``FinishReason/length``.
    public var maxOutputTokens: Int?

    /// How much randomness to allow, typically between `0` and `2`.
    ///
    /// Set this or ``topP``, not both: they are two ways of narrowing the same distribution, and
    /// combining them makes the effect of either hard to reason about.
    public var temperature: Double?

    /// Restricts sampling to the most likely tokens whose probabilities sum to this value.
    public var topP: Double?

    /// Restricts sampling to the `k` most likely tokens.
    ///
    /// Not supported by every provider; those that lack it report a ``CallWarning``.
    public var topK: Int?

    /// Discourages reusing tokens that have already appeared.
    public var presencePenalty: Double?

    /// Discourages reusing tokens in proportion to how often they have appeared.
    public var frequencyPenalty: Double?

    /// A seed for reproducible sampling, where the provider supports it.
    public var seed: Int?

    /// Sequences that end the response once generated.
    public var stopSequences: [String]

    /// How transient failures are retried. Defaults to two retries with exponential backoff.
    public var retryPolicy: RetryPolicy

    /// Extra HTTP headers to merge into the request.
    public var headers: [String: String]

    /// Provider-specific settings, namespaced by provider name.
    ///
    /// ```swift
    /// settings.providerOptions = ["anthropic": ["thinking": ["type": "enabled", "budget_tokens": 4096]]]
    /// ```
    public var providerOptions: ProviderOptions?

    /// Whether providers should emit their undecoded stream events as
    /// ``LanguageModelStreamPart/raw(_:)``.
    public var includesRawChunks: Bool

    /// How to fetch file URLs that the model cannot fetch for itself.
    ///
    /// When a prompt references a remote image or document and the provider requires inline
    /// bytes, the SDK downloads it. Supply a transport here to route those downloads through your
    /// own client, or to intercept them in tests. Defaults to a shared `URLSession` transport.
    public var fileTransport: (any HTTPTransport)?

    public init(
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        seed: Int? = nil,
        stopSequences: [String] = [],
        retryPolicy: RetryPolicy = .default,
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil,
        includesRawChunks: Bool = false,
        fileTransport: (any HTTPTransport)? = nil
    ) {
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.seed = seed
        self.stopSequences = stopSequences
        self.retryPolicy = retryPolicy
        self.headers = headers
        self.providerOptions = providerOptions
        self.includesRawChunks = includesRawChunks
        self.fileTransport = fileTransport
    }

    /// Settings that ask for the most deterministic output a provider can give.
    ///
    /// Useful for extraction and classification, where variation is a defect rather than a
    /// feature. Determinism is still not guaranteed — most providers make no such promise even at
    /// temperature zero.
    public static let deterministic = GenerationSettings(temperature: 0)

    /// Returns a copy with any setting that `overrides` specifies applied on top.
    ///
    /// Used by middleware and by ``Agent`` to layer per-call settings over stored defaults.
    /// Collections merge rather than replace, so a default header survives an override that does
    /// not mention it.
    public func merging(_ overrides: GenerationSettings?) -> GenerationSettings {
        guard let overrides else { return self }
        var merged = self
        merged.maxOutputTokens = overrides.maxOutputTokens ?? maxOutputTokens
        merged.temperature = overrides.temperature ?? temperature
        merged.topP = overrides.topP ?? topP
        merged.topK = overrides.topK ?? topK
        merged.presencePenalty = overrides.presencePenalty ?? presencePenalty
        merged.frequencyPenalty = overrides.frequencyPenalty ?? frequencyPenalty
        merged.seed = overrides.seed ?? seed
        merged.stopSequences = overrides.stopSequences.isEmpty ? stopSequences : overrides.stopSequences
        merged.retryPolicy = overrides.retryPolicy
        merged.headers = headers.merging(overrides.headers) { _, new in new }
        merged.providerOptions = providerOptions.map { $0.merging(overrides.providerOptions ?? ProviderOptions()) }
            ?? overrides.providerOptions
        merged.includesRawChunks = includesRawChunks || overrides.includesRawChunks
        merged.fileTransport = overrides.fileTransport ?? fileTransport
        return merged
    }

    /// Builds the provider-facing call options.
    func callOptions(
        prompt: [ModelMessage],
        tools: [LanguageModelTool] = [],
        toolChoice: ToolChoice? = nil,
        responseFormat: ResponseFormat? = nil
    ) -> LanguageModelCallOptions {
        LanguageModelCallOptions(
            prompt: prompt,
            maxOutputTokens: maxOutputTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            seed: seed,
            stopSequences: stopSequences,
            responseFormat: responseFormat,
            tools: tools,
            toolChoice: toolChoice,
            includesRawChunks: includesRawChunks,
            headers: headers,
            providerOptions: providerOptions
        )
    }
}

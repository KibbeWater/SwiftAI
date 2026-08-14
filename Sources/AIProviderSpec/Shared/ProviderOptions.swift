/// Provider-specific settings passed *into* a model call, namespaced by provider.
///
/// Every provider exposes knobs that have no cross-provider equivalent — reasoning effort,
/// thinking budgets, cache control, safety thresholds. Rather than growing a union of every
/// provider's settings on the shared options type, those are carried here under the provider's
/// own name and read only by that provider. A provider silently ignores namespaces that are not
/// its own, so a single set of options can travel with a prompt across models.
///
/// ```swift
/// var settings = GenerationSettings()
/// settings.providerOptions = [
///     "anthropic": ["thinking": ["type": "enabled", "budget_tokens": 4096]],
///     "openai": ["reasoningEffort": "high"],
/// ]
/// ```
///
/// Options can be attached at three levels — the call, an individual message, and an individual
/// content part — so that, for example, Anthropic cache breakpoints can be placed precisely.
///
/// - SeeAlso: ``ProviderMetadata``, which carries provider-specific values back *out* of a call.
public struct ProviderOptions: Sendable, Hashable, ExpressibleByDictionaryLiteral {
    /// The namespaced settings, keyed by provider name such as `"openai"` or `"anthropic"`.
    public var namespaces: [String: [String: JSONValue]]

    public init(_ namespaces: [String: [String: JSONValue]] = [:]) {
        self.namespaces = namespaces
    }

    public init(dictionaryLiteral elements: (String, [String: JSONValue])...) {
        self.namespaces = Dictionary(elements, uniquingKeysWith: { _, last in last })
    }

    /// Whether any options are present.
    public var isEmpty: Bool { namespaces.isEmpty }

    /// The settings registered for a provider.
    public subscript(provider: String) -> [String: JSONValue]? {
        get { namespaces[provider] }
        set { namespaces[provider] = newValue }
    }

    /// Reads a single setting from a provider's namespace.
    ///
    /// Providers use this when translating a call into their wire format.
    public func value(_ key: String, for provider: String) -> JSONValue? {
        namespaces[provider]?[key]
    }

    /// Returns a copy with `other`'s settings merged in, preferring `other` on conflicts.
    ///
    /// Merging is per-key within each namespace rather than wholesale namespace replacement, so
    /// defaults supplied by middleware combine with per-call overrides instead of erasing them.
    public func merging(_ other: ProviderOptions) -> ProviderOptions {
        var merged = namespaces
        for (provider, settings) in other.namespaces {
            merged[provider] = merged[provider]?.merging(settings) { _, new in new } ?? settings
        }
        return ProviderOptions(merged)
    }
}

/// Provider-specific values returned *from* a model call, namespaced by provider.
///
/// Providers use this to surface information that has no place on the shared result types:
/// Anthropic's cache-creation token counts, OpenAI's response identifiers, Google's safety
/// ratings. Read it with the provider's own name as the key.
///
/// ```swift
/// let cacheWrites = result.providerMetadata?["anthropic"]?["cacheCreationInputTokens"]?.intValue
/// ```
///
/// - SeeAlso: ``ProviderOptions``, which carries provider-specific settings *into* a call.
public struct ProviderMetadata: Sendable, Hashable, ExpressibleByDictionaryLiteral {
    /// The namespaced values, keyed by provider name.
    public var namespaces: [String: [String: JSONValue]]

    public init(_ namespaces: [String: [String: JSONValue]] = [:]) {
        self.namespaces = namespaces
    }

    public init(dictionaryLiteral elements: (String, [String: JSONValue])...) {
        self.namespaces = Dictionary(elements, uniquingKeysWith: { _, last in last })
    }

    /// Whether any metadata is present.
    public var isEmpty: Bool { namespaces.isEmpty }

    /// The values reported by a provider.
    public subscript(provider: String) -> [String: JSONValue]? {
        get { namespaces[provider] }
        set { namespaces[provider] = newValue }
    }

    /// Reads a single value from a provider's namespace.
    public func value(_ key: String, for provider: String) -> JSONValue? {
        namespaces[provider]?[key]
    }

    /// Returns a copy with `other`'s values merged in, preferring `other` on conflicts.
    ///
    /// Used to accumulate metadata across the steps of a multi-step call.
    public func merging(_ other: ProviderMetadata) -> ProviderMetadata {
        var merged = namespaces
        for (provider, values) in other.namespaces {
            merged[provider] = merged[provider]?.merging(values) { _, new in new } ?? values
        }
        return ProviderMetadata(merged)
    }

    /// Merges two optional metadata values, returning `nil` only when both are absent.
    public static func merging(_ lhs: ProviderMetadata?, _ rhs: ProviderMetadata?) -> ProviderMetadata? {
        switch (lhs, rhs) {
        case (nil, nil): return nil
        case (let lhs?, nil): return lhs
        case (nil, let rhs?): return rhs
        case (let lhs?, let rhs?): return lhs.merging(rhs)
        }
    }
}

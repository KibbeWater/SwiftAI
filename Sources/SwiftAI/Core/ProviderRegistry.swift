import AIProviderSpec
import Foundation

/// Resolves models by a single `"provider:model"` identifier.
///
/// Useful wherever a model has to be named by a string rather than chosen in code — a
/// configuration file, an environment variable, a request parameter — which is most server
/// deployments.
///
/// ```swift
/// let registry = ProviderRegistry([
///     "openai": OpenAIProvider(apiKey: openAIKey),
///     "anthropic": AnthropicProvider(apiKey: anthropicKey),
/// ])
///
/// let model = try registry.languageModel("anthropic:claude-sonnet-4-5")
/// ```
///
/// A registry is itself an ``AIProvider``, so it can be nested inside another one.
public struct ProviderRegistry: AIProvider, Sendable {
    public let name: String

    private let providers: [String: any AIProvider]
    private let separator: String

    /// Creates a registry.
    ///
    /// - Parameters:
    ///   - providers: The providers, keyed by the prefix that selects them.
    ///   - separator: What divides the prefix from the model identifier. Defaults to `":"`.
    ///   - name: The registry's own name, reported by ``AIProvider/name``.
    public init(
        _ providers: [String: any AIProvider],
        separator: String = ":",
        name: String = "registry"
    ) {
        self.providers = providers
        self.separator = separator
        self.name = name
    }

    /// The registered prefixes, sorted.
    public var providerIDs: [String] { providers.keys.sorted() }

    public func languageModel(_ id: String) throws -> any LanguageModel {
        let (provider, modelID) = try split(id, kind: .language)
        return try provider.languageModel(modelID)
    }

    public func embeddingModel(_ id: String) throws -> any EmbeddingModel {
        let (provider, modelID) = try split(id, kind: .embedding)
        return try provider.embeddingModel(modelID)
    }

    public func imageModel(_ id: String) throws -> any ImageModel {
        let (provider, modelID) = try split(id, kind: .image)
        return try provider.imageModel(modelID)
    }

    public func speechModel(_ id: String) throws -> any SpeechModel {
        let (provider, modelID) = try split(id, kind: .speech)
        return try provider.speechModel(modelID)
    }

    public func transcriptionModel(_ id: String) throws -> any TranscriptionModel {
        let (provider, modelID) = try split(id, kind: .transcription)
        return try provider.transcriptionModel(modelID)
    }

    /// Splits a qualified identifier into its provider and model parts.
    ///
    /// Only the first separator divides them, so model identifiers containing the separator —
    /// which several providers use — still resolve.
    private func split(
        _ id: String,
        kind: NoSuchModelError.ModelKind
    ) throws -> (provider: any AIProvider, modelID: String) {
        guard let range = id.range(of: separator) else {
            throw NoSuchModelError(
                modelID: id,
                modelKind: kind,
                message: """
                    '\(id)' is not a qualified model identifier. Write it as \
                    'provider\(separator)model', for example 'openai\(separator)gpt-5'.
                    """
            )
        }
        let providerID = String(id[id.startIndex..<range.lowerBound])
        let modelID = String(id[range.upperBound...])

        guard let provider = providers[providerID] else {
            throw NoSuchProviderError(providerID: providerID, availableProviders: Array(providers.keys))
        }
        return (provider, modelID)
    }
}

/// Presents a set of named models as a provider.
///
/// Use it to give configured models stable, meaningful names, so that swapping the model behind a
/// name is a one-line change rather than a search across the codebase:
///
/// ```swift
/// let models = CustomProvider(
///     name: "app",
///     languageModels: [
///         "fast": openai.languageModel("gpt-5-mini"),
///         "smart": wrapLanguageModel(
///             model: anthropic.languageModel("claude-sonnet-4-5"),
///             middleware: [DefaultSettingsMiddleware(temperature: 0.2)]
///         ),
///     ],
///     fallback: openai
/// )
///
/// let model = try models.languageModel("fast")
/// ```
public struct CustomProvider: AIProvider, Sendable {
    public let name: String

    private let languageModels: [String: any LanguageModel]
    private let embeddingModels: [String: any EmbeddingModel]
    private let imageModels: [String: any ImageModel]
    private let fallback: (any AIProvider)?

    /// Creates a provider from named models.
    ///
    /// - Parameters:
    ///   - name: The provider's name.
    ///   - languageModels: Language models by alias.
    ///   - embeddingModels: Embedding models by alias.
    ///   - imageModels: Image models by alias.
    ///   - fallback: Consulted for any name not listed above, so a custom provider can add aliases
    ///     to an existing one rather than replacing it.
    public init(
        name: String = "custom",
        languageModels: [String: any LanguageModel] = [:],
        embeddingModels: [String: any EmbeddingModel] = [:],
        imageModels: [String: any ImageModel] = [:],
        fallback: (any AIProvider)? = nil
    ) {
        self.name = name
        self.languageModels = languageModels
        self.embeddingModels = embeddingModels
        self.imageModels = imageModels
        self.fallback = fallback
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        if let model = languageModels[modelID] { return model }
        guard let fallback else {
            throw NoSuchModelError(modelID: modelID, modelKind: .language)
        }
        return try fallback.languageModel(modelID)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        if let model = embeddingModels[modelID] { return model }
        guard let fallback else {
            throw NoSuchModelError(modelID: modelID, modelKind: .embedding)
        }
        return try fallback.embeddingModel(modelID)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        if let model = imageModels[modelID] { return model }
        guard let fallback else {
            throw NoSuchModelError(modelID: modelID, modelKind: .image)
        }
        return try fallback.imageModel(modelID)
    }
}

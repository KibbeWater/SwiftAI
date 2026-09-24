/// A source of models, usually one vendor's API.
///
/// Conforming types are the entry point users reach for first:
///
/// ```swift
/// let openai = OpenAIProvider(apiKey: key)
/// let model = openai.languageModel("gpt-5")
/// ```
///
/// Every model kind has a default implementation that throws ``NoSuchModelError``, so a provider
/// implements only the modalities it actually offers.
public protocol AIProvider: Sendable {
    /// The provider's short name, such as `"openai"`.
    ///
    /// This is the namespace used for ``ProviderOptions``, ``ProviderMetadata``, and registry
    /// identifiers, so it must match the `provider` reported by the models this type vends.
    var name: String { get }

    /// Returns the language model with the given identifier.
    ///
    /// - Throws: ``NoSuchModelError`` if the provider offers no such model.
    func languageModel(_ modelID: String) throws -> any LanguageModel

    /// Returns the text embedding model with the given identifier.
    ///
    /// - Throws: ``NoSuchModelError`` if the provider offers no such model.
    func embeddingModel(_ modelID: String) throws -> any EmbeddingModel

    /// Returns the image generation model with the given identifier.
    ///
    /// - Throws: ``NoSuchModelError`` if the provider offers no such model.
    func imageModel(_ modelID: String) throws -> any ImageModel

    /// Returns the speech synthesis model with the given identifier.
    ///
    /// - Throws: ``NoSuchModelError`` if the provider offers no such model.
    func speechModel(_ modelID: String) throws -> any SpeechModel

    /// Returns the transcription model with the given identifier.
    ///
    /// - Throws: ``NoSuchModelError`` if the provider offers no such model.
    func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel

    /// Returns the evaluation model with the given identifier.
    ///
    /// - Important: Experimental. Evaluation models may change in a minor release.
    /// - Throws: ``NoSuchModelError`` if the provider offers no such model.
    func evaluationModel(_ modelID: String) throws -> any EvaluationModel
}

extension AIProvider {
    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        throw NoSuchModelError(modelID: modelID, modelKind: .language, message: unsupportedMessage(modelID, "language"))
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        throw NoSuchModelError(modelID: modelID, modelKind: .embedding, message: unsupportedMessage(modelID, "embedding"))
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        throw NoSuchModelError(modelID: modelID, modelKind: .image, message: unsupportedMessage(modelID, "image"))
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw NoSuchModelError(modelID: modelID, modelKind: .speech, message: unsupportedMessage(modelID, "speech"))
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw NoSuchModelError(
            modelID: modelID,
            modelKind: .transcription,
            message: unsupportedMessage(modelID, "transcription")
        )
    }

    public func evaluationModel(_ modelID: String) throws -> any EvaluationModel {
        throw NoSuchModelError(
            modelID: modelID,
            modelKind: .evaluation,
            message: unsupportedMessage(modelID, "evaluation")
        )
    }

    private func unsupportedMessage(_ modelID: String, _ kind: String) -> String {
        "The '\(name)' provider does not offer \(kind) models (requested '\(modelID)')."
    }
}

/// A vector embedding of a value.
public typealias Embedding = [Double]

/// The contract every embedding provider implements.
///
/// Implementations handle exactly one request. Batching a large set of values across several
/// requests is the core's job, driven by ``maxEmbeddingsPerCall`` and ``supportsParallelCalls``,
/// so providers do not have to reimplement it.
public protocol EmbeddingModelV2: Sendable {
    /// The provider's short name, such as `"openai"`.
    var provider: String { get }

    /// The provider's identifier for this model.
    var modelID: String { get }

    /// The most values the provider accepts in one request, or `nil` when there is no limit.
    ///
    /// The core splits larger inputs into chunks of this size.
    var maxEmbeddingsPerCall: Int? { get }

    /// Whether chunks may be sent concurrently.
    ///
    /// Providers with strict rate limits return `false` to force sequential requests.
    var supportsParallelCalls: Bool { get }

    /// Embeds a batch of values.
    ///
    /// - Parameter options: The values to embed and any provider-specific settings.
    /// - Returns: One embedding per input value, in the same order.
    /// - Throws: ``APICallError`` for upstream failures.
    func embed(_ options: EmbeddingModelCallOptions) async throws -> EmbeddingModelResponse
}

extension EmbeddingModelV2 {
    public var maxEmbeddingsPerCall: Int? { nil }
    public var supportsParallelCalls: Bool { true }
}

/// The current embedding model specification.
public typealias EmbeddingModel = EmbeddingModelV2

/// A request to embed a batch of values.
public struct EmbeddingModelCallOptions: Sendable {
    /// The values to embed. Never larger than ``EmbeddingModelV2/maxEmbeddingsPerCall``.
    public var values: [String]

    /// Extra HTTP headers to merge into the request.
    public var headers: [String: String]

    /// Provider-specific settings, such as an output dimension count.
    public var providerOptions: ProviderOptions?

    public init(
        values: [String],
        headers: [String: String] = [:],
        providerOptions: ProviderOptions? = nil
    ) {
        self.values = values
        self.headers = headers
        self.providerOptions = providerOptions
    }
}

/// The response from an embedding request.
public struct EmbeddingModelResponse: Sendable {
    /// One embedding per input value, in the same order as the request.
    public var embeddings: [Embedding]

    /// Token usage. Embedding providers report input tokens only, in ``Usage/inputTokens``.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about the response.
    public var response: ResponseInfo?

    public init(
        embeddings: [Embedding],
        usage: Usage = .none,
        warnings: [CallWarning] = [],
        providerMetadata: ProviderMetadata? = nil,
        response: ResponseInfo? = nil
    ) {
        self.embeddings = embeddings
        self.usage = usage
        self.warnings = warnings
        self.providerMetadata = providerMetadata
        self.response = response
    }
}

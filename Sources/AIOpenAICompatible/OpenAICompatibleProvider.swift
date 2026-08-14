import AIProviderSpec
import AIProviderUtils
import Foundation

/// Where a particular OpenAI-compatible server departs from OpenAI's own behaviour.
///
/// "OpenAI-compatible" describes a family rather than a specification. Servers agree on the shape
/// of a request and disagree on the details: which token-limit field to use, whether schemas can
/// be enforced, whether streamed responses report usage. Collecting those differences here means
/// one implementation serves all of them, and adding support for a new server is a value rather
/// than a subclass.
public struct OpenAICompatibleQuirks: Sendable {
    /// Whether to send `max_completion_tokens` instead of `max_tokens`.
    ///
    /// OpenAI has moved to the former and rejects the latter on newer models; most other servers
    /// still expect `max_tokens`.
    public var usesMaxCompletionTokens: Bool

    /// Whether the server can constrain output to a JSON Schema.
    ///
    /// When `false`, a schema is supplied as an instruction instead and a warning is reported,
    /// since conformance is then a matter of the model's cooperation rather than a guarantee.
    public var supportsStructuredOutputs: Bool

    /// Whether the server accepts `strict: true` on schemas.
    public var supportsStrictSchemas: Bool

    /// Whether the server reports usage on streamed responses when asked with `stream_options`.
    public var supportsStreamUsage: Bool

    /// Whether the server fetches image URLs itself rather than requiring inline bytes.
    public var fetchesImageURLs: Bool

    /// The role name for system messages.
    ///
    /// OpenAI's reasoning models expect `developer`; everything else expects `system`.
    public var systemRole: String

    /// The path of the chat completions endpoint, relative to the base URL.
    public var chatCompletionsPath: String

    /// The path of the embeddings endpoint, relative to the base URL.
    public var embeddingsPath: String

    /// The namespace this provider reads from ``ProviderOptions``.
    ///
    /// Anything under it is merged into the request body verbatim, which is the escape hatch for
    /// server-specific parameters this type does not model.
    public var providerOptionsNamespace: String

    public init(
        usesMaxCompletionTokens: Bool = false,
        supportsStructuredOutputs: Bool = true,
        supportsStrictSchemas: Bool = false,
        supportsStreamUsage: Bool = true,
        fetchesImageURLs: Bool = false,
        systemRole: String = "system",
        chatCompletionsPath: String = "chat/completions",
        embeddingsPath: String = "embeddings",
        providerOptionsNamespace: String = "openaiCompatible"
    ) {
        self.usesMaxCompletionTokens = usesMaxCompletionTokens
        self.supportsStructuredOutputs = supportsStructuredOutputs
        self.supportsStrictSchemas = supportsStrictSchemas
        self.supportsStreamUsage = supportsStreamUsage
        self.fetchesImageURLs = fetchesImageURLs
        self.systemRole = systemRole
        self.chatCompletionsPath = chatCompletionsPath
        self.embeddingsPath = embeddingsPath
        self.providerOptionsNamespace = providerOptionsNamespace
    }

    /// Conservative defaults that work against the widest range of servers.
    public static let `default` = OpenAICompatibleQuirks()

    /// OpenAI's own behaviour: strict schemas, `max_completion_tokens`, and URL fetching.
    public static let openAI = OpenAICompatibleQuirks(
        usesMaxCompletionTokens: true,
        supportsStructuredOutputs: true,
        supportsStrictSchemas: true,
        supportsStreamUsage: true,
        fetchesImageURLs: true,
        providerOptionsNamespace: "openai"
    )

    /// Ollama, which serves local models and does not enforce schemas.
    public static let ollama = OpenAICompatibleQuirks(
        supportsStructuredOutputs: false,
        supportsStreamUsage: false,
        providerOptionsNamespace: "ollama"
    )
}

/// A provider for any server that speaks the OpenAI Chat Completions API.
///
/// ```swift
/// // A local Ollama server.
/// let ollama = OpenAICompatibleProvider(
///     name: "ollama",
///     baseURL: URL(string: "http://localhost:11434/v1")!,
///     apiKey: "ollama",
///     quirks: .ollama
/// )
/// let model = ollama.languageModel("llama3.2")
///
/// // A hosted gateway.
/// let groq = OpenAICompatibleProvider(
///     name: "groq",
///     baseURL: URL(string: "https://api.groq.com/openai/v1")!,
///     apiKeyEnvironmentVariable: "GROQ_API_KEY"
/// )
/// ```
///
/// Credentials are resolved lazily, when the first request is made, so constructing a provider
/// never fails and an application that configures several providers does not require keys for the
/// ones it will not use.
public struct OpenAICompatibleProvider: AIProvider, Sendable {
    public let name: String

    let client: ProviderHTTPClient
    let quirks: OpenAICompatibleQuirks

    /// Creates a provider.
    ///
    /// - Parameters:
    ///   - name: The provider's short name. Used as the ``ProviderMetadata`` namespace and in
    ///     error messages, so give each configured server a distinct one.
    ///   - baseURL: The API root, including any version path such as `/v1`.
    ///   - apiKey: The key to send as a bearer token. When `nil`, the environment is consulted.
    ///   - apiKeyEnvironmentVariable: The variable to read when no key is supplied.
    ///   - headers: Extra headers sent with every request.
    ///   - transport: The HTTP transport. Substitute this in tests.
    ///   - quirks: How this server differs from OpenAI's own behaviour.
    public init(
        name: String,
        baseURL: URL,
        apiKey: String? = nil,
        apiKeyEnvironmentVariable: String? = nil,
        headers: [String: String] = [:],
        transport: any HTTPTransport = URLSessionTransport.shared,
        quirks: OpenAICompatibleQuirks = .default
    ) {
        self.name = name
        self.quirks = quirks
        self.client = ProviderHTTPClient(
            baseURL: baseURL,
            provider: name,
            headers: {
                var merged = headers
                // A missing key is only an error if the server actually needs one, so an absent
                // environment variable simply means no Authorization header.
                let resolved = apiKey
                    ?? apiKeyEnvironmentVariable.flatMap { ProcessInfo.processInfo.environment[$0] }
                if let resolved, !resolved.isEmpty {
                    merged["Authorization"] = "Bearer \(resolved)"
                }
                return merged
            },
            transport: transport,
            decoder: ProviderJSON.snakeCaseDecoder
        )
    }

    /// Creates a provider from a pre-configured HTTP client.
    ///
    /// Used by ``AIOpenAI`` to reuse this implementation for its chat endpoint.
    public init(name: String, client: ProviderHTTPClient, quirks: OpenAICompatibleQuirks) {
        self.name = name
        self.client = client
        self.quirks = quirks
    }

    /// Returns a chat model.
    public func languageModel(_ modelID: String) -> any LanguageModel {
        ChatCompletionsModel(provider: name, modelID: modelID, client: client, quirks: quirks)
    }

    /// Returns an embedding model.
    ///
    /// Uses the batch limit OpenAI documents. For a server with tighter bounds, use
    /// ``embeddingModel(_:maxEmbeddingsPerCall:supportsParallelCalls:)``.
    public func embeddingModel(_ modelID: String) -> any EmbeddingModel {
        embeddingModel(modelID, maxEmbeddingsPerCall: 2048)
    }

    /// Returns an embedding model with explicit batching limits.
    ///
    /// - Parameters:
    ///   - modelID: The model identifier.
    ///   - maxEmbeddingsPerCall: How many values one request may carry, or `nil` for no limit.
    ///     Larger inputs are split into this many values per request.
    ///   - supportsParallelCalls: Whether batches may be sent concurrently. Set `false` for
    ///     servers with strict rate limits.
    public func embeddingModel(
        _ modelID: String,
        maxEmbeddingsPerCall: Int?,
        supportsParallelCalls: Bool = true
    ) -> any EmbeddingModel {
        OpenAICompatibleEmbeddingModel(
            provider: name,
            modelID: modelID,
            client: client,
            path: quirks.embeddingsPath,
            maxEmbeddingsPerCall: maxEmbeddingsPerCall,
            supportsParallelCalls: supportsParallelCalls
        )
    }
}

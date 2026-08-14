import AIProviderSpec
import AIProviderUtils
import Foundation

/// The result of embedding one value.
public struct EmbedResult: Sendable {
    /// The value that was embedded.
    public var value: String

    /// Its vector representation.
    public var embedding: Embedding

    /// Token usage for the call.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Provider-specific values.
    public var providerMetadata: ProviderMetadata?

    /// Metadata about the response.
    public var response: ResponseInfo?
}

/// The result of embedding several values.
public struct EmbedManyResult: Sendable {
    /// The values that were embedded, in the order they were supplied.
    public var values: [String]

    /// Their vector representations, in the same order.
    public var embeddings: [Embedding]

    /// Token usage across every request made.
    public var usage: Usage

    /// Settings the provider could not honor.
    public var warnings: [CallWarning]

    /// Metadata about each response, one per request.
    public var responses: [ResponseInfo]

    /// The embeddings paired with their values.
    public var pairs: [(value: String, embedding: Embedding)] {
        Array(zip(values, embeddings)).map { (value: $0.0, embedding: $0.1) }
    }
}

/// Embeds a single value.
///
/// ```swift
/// let result = try await embed(model: openai.embeddingModel("text-embedding-3-small"), value: "hello")
/// ```
///
/// - Parameters:
///   - model: The embedding model.
///   - value: The text to embed.
///   - retryPolicy: How transient failures are retried.
///   - headers: Extra HTTP headers.
///   - providerOptions: Provider-specific settings, such as an output dimension count.
/// - Returns: The embedding, with usage.
/// - Throws: ``APICallError`` for upstream failures.
public func embed(
    model: any EmbeddingModel,
    value: String,
    retryPolicy: RetryPolicy = .default,
    headers: [String: String] = [:],
    providerOptions: ProviderOptions? = nil
) async throws -> EmbedResult {
    let response = try await withRetries(policy: retryPolicy) { _ in
        try await model.embed(
            EmbeddingModelCallOptions(values: [value], headers: headers, providerOptions: providerOptions)
        )
    }
    guard let embedding = response.embeddings.first else {
        throw TypeValidationError(message: "The provider returned no embedding for the supplied value.")
    }
    return EmbedResult(
        value: value,
        embedding: embedding,
        usage: response.usage,
        warnings: response.warnings,
        providerMetadata: response.providerMetadata,
        response: response.response
    )
}

/// Embeds many values, batching them to fit the provider's limits.
///
/// Providers cap how many values one request may carry, and the cap differs between them. Rather
/// than making that the caller's problem, the input is split into chunks the model accepts and the
/// chunks are sent concurrently — subject to `maximumParallelCalls` and to whether the provider
/// permits parallel requests at all. Results are reassembled in the original order.
///
/// ```swift
/// let result = try await embedMany(model: model, values: documents)
/// let index = zip(documents, result.embeddings)
/// ```
///
/// - Parameters:
///   - model: The embedding model.
///   - values: The texts to embed. May be far larger than one request allows.
///   - maximumParallelCalls: How many requests may be in flight at once. Ignored when the provider
///     reports that it does not support parallel calls.
///   - retryPolicy: How transient failures are retried. Applied per chunk.
///   - headers: Extra HTTP headers.
///   - providerOptions: Provider-specific settings.
/// - Returns: One embedding per input value, in order, with combined usage.
/// - Throws: ``APICallError`` for upstream failures, or ``InvalidArgumentError`` if the provider
///   returns a different number of embeddings than were requested.
public func embedMany(
    model: any EmbeddingModel,
    values: [String],
    maximumParallelCalls: Int = 4,
    retryPolicy: RetryPolicy = .default,
    headers: [String: String] = [:],
    providerOptions: ProviderOptions? = nil
) async throws -> EmbedManyResult {
    guard !values.isEmpty else {
        return EmbedManyResult(values: [], embeddings: [], usage: .none, warnings: [], responses: [])
    }

    let chunkSize = model.maxEmbeddingsPerCall ?? values.count
    let chunks = stride(from: 0, to: values.count, by: max(1, chunkSize)).map { start in
        Array(values[start..<min(start + max(1, chunkSize), values.count)])
    }
    let parallelism = model.supportsParallelCalls ? max(1, maximumParallelCalls) : 1

    /// Embeds one chunk, tagged with its position so results can be reordered afterwards.
    @Sendable func embedChunk(_ index: Int) async throws -> (Int, EmbeddingModelResponse) {
        let response = try await withRetries(policy: retryPolicy) { _ in
            try await model.embed(
                EmbeddingModelCallOptions(
                    values: chunks[index],
                    headers: headers,
                    providerOptions: providerOptions
                )
            )
        }
        guard response.embeddings.count == chunks[index].count else {
            throw InvalidArgumentError(
                argument: "values",
                message: """
                    The provider returned \(response.embeddings.count) embeddings for \
                    \(chunks[index].count) values.
                    """
            )
        }
        return (index, response)
    }

    // A bounded task group: start up to `parallelism` requests, then start another each time one
    // finishes. Sending every chunk at once would trip rate limits on any sizeable input.
    let responses = try await withThrowingTaskGroup(
        of: (Int, EmbeddingModelResponse).self,
        returning: [(Int, EmbeddingModelResponse)].self
    ) { group in
        var next = 0
        var collected: [(Int, EmbeddingModelResponse)] = []

        for _ in 0..<min(parallelism, chunks.count) {
            let index = next
            group.addTask { try await embedChunk(index) }
            next += 1
        }
        while let finished = try await group.next() {
            collected.append(finished)
            if next < chunks.count {
                let index = next
                group.addTask { try await embedChunk(index) }
                next += 1
            }
        }
        return collected
    }

    let ordered = responses.sorted { $0.0 < $1.0 }.map(\.1)
    return EmbedManyResult(
        values: values,
        embeddings: ordered.flatMap(\.embeddings),
        usage: ordered.reduce(Usage.none) { $0.adding($1.usage) },
        warnings: ordered.flatMap(\.warnings),
        responses: ordered.compactMap(\.response)
    )
}

/// The cosine of the angle between two vectors.
///
/// The standard similarity measure for embeddings: `1` means the same direction, `0` means
/// unrelated, `-1` means opposite. Because it ignores magnitude, it compares meaning rather than
/// length.
///
/// - Parameters:
///   - first: One embedding.
///   - second: Another, of the same length.
/// - Returns: A value in `-1...1`, or `0` when either vector has zero magnitude.
/// - Throws: ``InvalidArgumentError`` if the vectors have different lengths, which almost always
///   means two different models produced them.
public func cosineSimilarity(_ first: Embedding, _ second: Embedding) throws -> Double {
    guard first.count == second.count else {
        throw InvalidArgumentError(
            argument: "embeddings",
            message: """
                Cannot compare embeddings of different lengths (\(first.count) and \(second.count)). \
                They were probably produced by different models.
                """
        )
    }
    var dotProduct = 0.0
    var firstMagnitude = 0.0
    var secondMagnitude = 0.0
    for index in first.indices {
        dotProduct += first[index] * second[index]
        firstMagnitude += first[index] * first[index]
        secondMagnitude += second[index] * second[index]
    }
    let denominator = (firstMagnitude * secondMagnitude).squareRoot()
    return denominator == 0 ? 0 : dotProduct / denominator
}

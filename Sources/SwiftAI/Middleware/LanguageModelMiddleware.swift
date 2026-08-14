import AIProviderSpec
import Foundation

/// Which kind of call a middleware is being asked to transform.
public enum ModelCallKind: String, Sendable, Hashable {
    case generate
    case stream
}

/// Intercepts calls to a language model.
///
/// Middleware wraps a model at the provider boundary, below the tool loop and above the wire
/// format. That position is what makes it provider-agnostic: the same guardrail, cache, or logger
/// works against every provider, because all of them present the same interface here.
///
/// ```swift
/// struct RequestLogger: LanguageModelMiddleware {
///     func generate(
///         _ options: LanguageModelCallOptions,
///         model: any LanguageModel,
///         next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse
///     ) async throws -> LanguageModelResponse {
///         let started = ContinuousClock.now
///         let response = try await next(options)
///         logger.info("\(model.modelID) took \(started.duration(to: .now)), \(response.usage)")
///         return response
///     }
/// }
///
/// let observed = wrapLanguageModel(model: model, middleware: [RequestLogger()])
/// ```
///
/// Every requirement has a pass-through default, so implement only the one you need.
public protocol LanguageModelMiddleware: Sendable {
    /// Adjusts the request before it reaches the model.
    ///
    /// Use this for changes that do not depend on the response: injecting retrieved context,
    /// forcing a setting, adding provider options.
    func transformOptions(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        kind: ModelCallKind
    ) async throws -> LanguageModelCallOptions

    /// Wraps a buffered call.
    ///
    /// Call `next` to proceed, or return a response without calling it to short-circuit — which
    /// is how a cache is built.
    func generate(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse
    ) async throws -> LanguageModelResponse

    /// Wraps a streaming call.
    ///
    /// Transforming a stream means returning a new one that consumes the original. Remember that
    /// a stream must still be well formed: begin with a start part, end with a finish part.
    func stream(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelStreamResponse
    ) async throws -> LanguageModelStreamResponse
}

extension LanguageModelMiddleware {
    public func transformOptions(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        kind: ModelCallKind
    ) async throws -> LanguageModelCallOptions {
        options
    }

    public func generate(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse
    ) async throws -> LanguageModelResponse {
        try await next(options)
    }

    public func stream(
        _ options: LanguageModelCallOptions,
        model: any LanguageModel,
        next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelStreamResponse
    ) async throws -> LanguageModelStreamResponse {
        try await next(options)
    }
}

/// Wraps a model in middleware.
///
/// Middleware is applied outside-in: the first element is outermost and sees the call first, the
/// last is closest to the model. Reading the array top to bottom therefore traces a request's
/// path.
///
/// ```swift
/// let model = wrapLanguageModel(
///     model: openai.languageModel("gpt-5"),
///     middleware: [
///         RequestLogger(),                        // sees everything
///         DefaultSettingsMiddleware(.deterministic),
///         ExtractReasoningMiddleware(),           // closest to the model
///     ]
/// )
/// ```
///
/// The result is an ordinary ``LanguageModel``, so wrapped models can be passed anywhere a model
/// is expected, stored in a registry, or wrapped again.
///
/// - Parameters:
///   - model: The model to wrap.
///   - middleware: The middleware, outermost first. An empty array returns the model unchanged.
///   - provider: Overrides the reported provider name.
///   - modelID: Overrides the reported model identifier.
/// - Returns: The wrapped model.
public func wrapLanguageModel(
    model: any LanguageModel,
    middleware: [any LanguageModelMiddleware],
    provider: String? = nil,
    modelID: String? = nil
) -> any LanguageModel {
    guard !middleware.isEmpty || provider != nil || modelID != nil else { return model }
    return WrappedLanguageModel(
        base: model,
        middleware: middleware,
        providerOverride: provider,
        modelIDOverride: modelID
    )
}

/// A model with middleware applied. See ``wrapLanguageModel(model:middleware:provider:modelID:)``.
struct WrappedLanguageModel: LanguageModel {
    let base: any LanguageModel
    let middleware: [any LanguageModelMiddleware]
    let providerOverride: String?
    let modelIDOverride: String?

    var provider: String { providerOverride ?? base.provider }
    var modelID: String { modelIDOverride ?? base.modelID }

    func supportsNativeURL(_ url: URL, mediaType: String) -> Bool {
        base.supportsNativeURL(url, mediaType: mediaType)
    }

    func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse {
        var transformed = options
        for element in middleware {
            transformed = try await element.transformOptions(transformed, model: base, kind: .generate)
        }

        // Fold from the inside out so that `middleware[0]` ends up outermost.
        var next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelResponse = {
            try await base.generate($0)
        }
        for element in middleware.reversed() {
            let inner = next
            next = { try await element.generate($0, model: base, next: inner) }
        }
        return try await next(transformed)
    }

    func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse {
        var transformed = options
        for element in middleware {
            transformed = try await element.transformOptions(transformed, model: base, kind: .stream)
        }

        var next: @Sendable (LanguageModelCallOptions) async throws -> LanguageModelStreamResponse = {
            try await base.stream($0)
        }
        for element in middleware.reversed() {
            let inner = next
            next = { try await element.stream($0, model: base, next: inner) }
        }
        return try await next(transformed)
    }
}

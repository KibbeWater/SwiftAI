import Foundation

/// The root of the SDK's error hierarchy.
///
/// Every error thrown by SwiftAI and by first-party providers conforms to `AISDKError`. The
/// public API uses untyped `throws` rather than typed throws, because a single call aggregates
/// failures from many sources — networking, decoding, schema validation, user-supplied tool
/// bodies — and the set of possible failures needs room to grow without breaking source
/// compatibility. Recover from specific failures with a typed `catch`:
///
/// ```swift
/// do {
///     let result = try await generateText(model: model, prompt: "Hello")
/// } catch let error as APICallError where error.isRetryable {
///     // Transient upstream failure.
/// } catch let error as any AISDKError {
///     logger.error("\(error.description)")
/// }
/// ```
public protocol AISDKError: Error, Sendable, CustomStringConvertible {
    /// A stable, machine-readable identifier for the kind of failure, such as `"AI_APICallError"`.
    ///
    /// Useful for logging and metrics where matching on Swift types is inconvenient.
    var name: String { get }

    /// A human-readable explanation of what went wrong.
    var message: String { get }
}

extension AISDKError {
    public var description: String { "\(name): \(message)" }
}

extension AISDKError where Self: LocalizedError {
    public var errorDescription: String? { description }
}

// MARK: - Transport failures

/// A non-success response from a provider's HTTP API, or a failure to reach it.
///
/// Providers throw this for any response the upstream service rejects. The SDK's retry logic
/// keys off ``isRetryable``, so providers should classify failures accurately rather than
/// retrying themselves.
public struct APICallError: AISDKError, LocalizedError {
    public var name: String { "AI_APICallError" }
    public var message: String

    /// The URL that was requested.
    public var url: URL

    /// The HTTP method used, such as `"POST"`.
    public var method: String

    /// The HTTP status code, or `nil` if the request failed before a response arrived.
    public var statusCode: Int?

    /// Headers returned by the provider. Useful for reading rate-limit metadata.
    public var responseHeaders: [String: String]

    /// The raw response body, when one was received.
    public var responseBody: String?

    /// The request body that was sent, for debugging. Providers should omit credentials.
    public var requestBody: String?

    /// Whether retrying the identical request could plausibly succeed.
    ///
    /// Conventionally true for `408`, `409`, `429` and `5xx` responses, and for transport-level
    /// failures such as a dropped connection.
    public var isRetryable: Bool

    /// The provider's parsed error payload, when the body was JSON.
    public var data: JSONValue?

    /// The lower-level error that caused this failure, such as a `URLError`.
    public var underlyingError: (any Error)?

    public init(
        message: String,
        url: URL,
        method: String = "POST",
        statusCode: Int? = nil,
        responseHeaders: [String: String] = [:],
        responseBody: String? = nil,
        requestBody: String? = nil,
        isRetryable: Bool? = nil,
        data: JSONValue? = nil,
        underlyingError: (any Error)? = nil
    ) {
        self.message = message
        self.url = url
        self.method = method
        self.statusCode = statusCode
        self.responseHeaders = responseHeaders
        self.responseBody = responseBody
        self.requestBody = requestBody
        self.isRetryable = isRetryable ?? APICallError.defaultIsRetryable(statusCode: statusCode)
        self.data = data
        self.underlyingError = underlyingError
    }

    /// The default retry classification for a status code.
    ///
    /// `408` (request timeout), `409` (conflict), `429` (too many requests) and every `5xx`
    /// are treated as transient. A missing status code means the request never completed, which
    /// is also treated as transient.
    public static func defaultIsRetryable(statusCode: Int?) -> Bool {
        guard let statusCode else { return true }
        return statusCode == 408 || statusCode == 409 || statusCode == 429 || statusCode >= 500
    }

    public var description: String {
        var text = "\(name): \(message)"
        if let statusCode { text += " (status \(statusCode))" }
        if let responseBody, !responseBody.isEmpty {
            text += "\nResponse body: \(responseBody.prefix(2048))"
        }
        return text
    }
}

/// Every attempt at a retryable request failed.
///
/// Thrown only when the retry budget is exhausted. A failure classified as non-retryable — a
/// `401`, say — propagates unchanged instead of being wrapped, so the common `catch let error as
/// APICallError` still sees the response that actually caused the problem.
public struct RetryError: AISDKError, LocalizedError {
    public var name: String { "AI_RetryError" }
    public var message: String

    /// Every error encountered, in the order the attempts were made.
    public var errors: [any Error]

    /// The error from the final attempt.
    public var lastError: any Error

    /// How many attempts were made in total, including the first.
    public var attempts: Int { errors.count }

    public init(errors: [any Error], lastError: any Error) {
        self.errors = errors
        self.lastError = lastError
        self.message = "The request still failed after \(errors.count) attempt(s). Last error: \(lastError)"
    }
}

// MARK: - Data failures

/// JSON text could not be parsed.
public struct JSONParseError: AISDKError, LocalizedError {
    public var name: String { "AI_JSONParseError" }
    public var message: String

    /// The byte offset at which parsing failed, when known.
    public var offset: Int?

    /// The text that failed to parse, when it is small enough to be worth carrying.
    public var text: String?

    public init(message: String, offset: Int? = nil, text: String? = nil) {
        self.message = message
        self.offset = offset
        self.text = text
    }

    public var description: String {
        var description = "\(name): \(message)"
        if let offset { description += " (at offset \(offset))" }
        return description
    }
}

/// A JSON value did not match the shape a Swift type requires.
///
/// Thrown by ``StructuredOutput`` decoding when, for example, a required property is missing or
/// a string appears where a number was expected.
public struct TypeValidationError: AISDKError, LocalizedError {
    public var name: String { "AI_TypeValidationError" }
    public var message: String

    /// The dotted path to the offending value, such as `"ingredients.2.quantity"`.
    public var path: String?

    /// The value that failed validation.
    public var value: JSONValue?

    public init(message: String, path: String? = nil, value: JSONValue? = nil) {
        self.message = message
        self.path = path
        self.value = value
    }

    /// Returns a copy of the error with `component` prepended to the path.
    ///
    /// Decoders build the path from the inside out as the error propagates back up through
    /// nested containers.
    public func prependingPath(_ component: String) -> TypeValidationError {
        var copy = self
        copy.path = copy.path.map { "\(component).\($0)" } ?? component
        return copy
    }

    public var description: String {
        guard let path else { return "\(name): \(message)" }
        return "\(name): \(message) (at '\(path)')"
    }
}

// MARK: - Capability and configuration failures

/// A provider does not implement a requested capability.
///
/// Providers throw this only when a feature cannot be approximated at all. Settings that merely
/// have no equivalent should produce a ``CallWarning`` instead, so the call still succeeds.
public struct UnsupportedFunctionalityError: AISDKError, LocalizedError {
    public var name: String { "AI_UnsupportedFunctionalityError" }
    public var message: String

    /// The capability that is unavailable, such as `"image input"`.
    public var functionality: String

    public init(functionality: String, message: String? = nil) {
        self.functionality = functionality
        self.message = message ?? "'\(functionality)' is not supported by this model."
    }
}

/// A model identifier could not be resolved.
public struct NoSuchModelError: AISDKError, LocalizedError {
    public var name: String { "AI_NoSuchModelError" }
    public var message: String

    /// The kind of model that was requested.
    public enum ModelKind: String, Sendable, Hashable {
        case language, embedding, image, speech, transcription
    }

    public var modelID: String
    public var modelKind: ModelKind

    public init(modelID: String, modelKind: ModelKind, message: String? = nil) {
        self.modelID = modelID
        self.modelKind = modelKind
        self.message = message ?? "No \(modelKind.rawValue) model is registered for '\(modelID)'."
    }
}

/// A provider prefix in a registry lookup did not match any registered provider.
public struct NoSuchProviderError: AISDKError, LocalizedError {
    public var name: String { "AI_NoSuchProviderError" }
    public var message: String

    public var providerID: String
    public var availableProviders: [String]

    public init(providerID: String, availableProviders: [String]) {
        self.providerID = providerID
        self.availableProviders = availableProviders.sorted()
        self.message = """
            No provider is registered under '\(providerID)'. \
            Available providers: \(self.availableProviders.joined(separator: ", ")).
            """
    }
}

/// An argument supplied to the SDK was invalid.
public struct InvalidArgumentError: AISDKError, LocalizedError {
    public var name: String { "AI_InvalidArgumentError" }
    public var message: String

    /// The parameter at fault.
    public var argument: String

    public init(argument: String, message: String) {
        self.argument = argument
        self.message = message
    }
}

/// A prompt could not be converted into the normalized message list.
public struct InvalidPromptError: AISDKError, LocalizedError {
    public var name: String { "AI_InvalidPromptError" }
    public var message: String

    public init(message: String) {
        self.message = message
    }
}

/// A provider could not find the credentials it needs.
public struct MissingAPIKeyError: AISDKError, LocalizedError {
    public var name: String { "AI_MissingAPIKeyError" }
    public var message: String

    public var provider: String
    public var environmentVariable: String

    public init(provider: String, environmentVariable: String) {
        self.provider = provider
        self.environmentVariable = environmentVariable
        self.message = """
            No API key was supplied for the '\(provider)' provider. Pass one to the provider's \
            initializer, or set the \(environmentVariable) environment variable.
            """
    }
}

/// A referenced file or URL could not be fetched for inclusion in a prompt.
public struct DownloadError: AISDKError, LocalizedError {
    public var name: String { "AI_DownloadError" }
    public var message: String

    public var url: URL
    public var statusCode: Int?
    public var underlyingError: (any Error)?

    public init(url: URL, statusCode: Int? = nil, underlyingError: (any Error)? = nil, message: String? = nil) {
        self.url = url
        self.statusCode = statusCode
        self.underlyingError = underlyingError
        self.message = message ?? "Failed to download '\(url.absoluteString)'."
    }
}

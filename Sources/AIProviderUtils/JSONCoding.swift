import AIProviderSpec
import Foundation

/// Shared JSON coding configuration for provider wire types.
///
/// Providers model their request and response payloads as `Codable` structs and go through these
/// helpers, which keeps encoder configuration and error translation in one place instead of
/// scattered across every provider.
public enum ProviderJSON {
    /// The encoder used for request bodies.
    ///
    /// Keys are sorted. JSON object order is semantically irrelevant, but a deterministic body
    /// makes request assertions in tests stable and keeps hashes consistent for providers that
    /// cache on the request payload.
    public static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    /// The decoder used for response bodies whose keys already match their Swift names.
    ///
    /// Google's APIs use camel case, so their provider uses this directly.
    public static let decoder = JSONDecoder()

    /// A decoder that converts `snake_case` keys to camel case.
    ///
    /// OpenAI and Anthropic both use snake case throughout, and hand-writing `CodingKeys` for
    /// every wire type would be a great deal of noise for no benefit.
    public static let snakeCaseDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    /// Encodes a value as a request body.
    ///
    /// - Throws: ``InvalidArgumentError`` if the value cannot be represented as JSON, which
    ///   indicates a bug in the provider rather than a problem with the request.
    public static func encode(_ value: some Encodable) throws -> Data {
        do {
            return try encoder.encode(value)
        } catch {
            throw InvalidArgumentError(
                argument: "body",
                message: "Failed to encode the request body: \(error)"
            )
        }
    }

    /// Decodes a response body.
    ///
    /// - Parameters:
    ///   - type: The expected shape.
    ///   - data: The raw body.
    ///   - context: What was being decoded, used in the error message. For example
    ///     `"the chat completion response"`.
    /// - Throws: ``TypeValidationError`` describing where the payload diverged from expectations.
    ///   The message names the offending key path, because "the provider changed its response
    ///   shape" is otherwise an unpleasant thing to debug from a stack trace alone.
    public static func decode<T: Decodable>(
        _ type: T.Type,
        from data: Data,
        context: String,
        decoder: JSONDecoder = ProviderJSON.decoder
    ) throws -> T {
        do {
            return try decoder.decode(type, from: data)
        } catch let error as DecodingError {
            throw TypeValidationError(
                message: "Could not decode \(context): \(describe(error))",
                path: keyPath(of: error),
                value: try? JSONValue.parse(data)
            )
        } catch {
            throw TypeValidationError(message: "Could not decode \(context): \(error)")
        }
    }

    /// Decodes a JSON string that arrived inside another payload, such as a tool call's arguments.
    ///
    /// - Parameters:
    ///   - text: The JSON text.
    ///   - context: What the text represents, used in the error message.
    /// - Returns: The parsed value, or an empty object when the text is empty — several providers
    ///   send `""` rather than `"{}"` for a tool with no arguments.
    public static func parseEmbeddedJSON(_ text: String, context: String) throws -> JSONValue {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .object([:]) }
        do {
            return try JSONValue.parse(trimmed)
        } catch let error as JSONParseError {
            throw JSONParseError(
                message: "Could not parse \(context): \(error.message)",
                offset: error.offset,
                text: String(trimmed.prefix(1024))
            )
        }
    }

    // MARK: - Error description

    private static func describe(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, _):
            return "the required key '\(key.stringValue)' was missing"
        case .typeMismatch(let type, let context):
            return "expected \(type) but found something else\(context.debugDescription.isEmpty ? "" : " (\(context.debugDescription))")"
        case .valueNotFound(let type, _):
            return "a required \(type) value was null"
        case .dataCorrupted(let context):
            return context.debugDescription
        @unknown default:
            return String(describing: error)
        }
    }

    private static func keyPath(of error: DecodingError) -> String? {
        let path: [any CodingKey]
        switch error {
        case .keyNotFound(let key, let context): path = context.codingPath + [key]
        case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context):
            path = context.codingPath
        @unknown default: return nil
        }
        guard !path.isEmpty else { return nil }
        return path.map { $0.intValue.map(String.init) ?? $0.stringValue }.joined(separator: ".")
    }
}

// MARK: - JSONValue bridging

extension JSONValue {
    /// Creates a value by encoding any `Encodable`.
    ///
    /// Useful where a provider needs to splice a typed payload into a dynamic structure.
    public init(encoding value: some Encodable) throws {
        let data = try ProviderJSON.encode(value)
        self = try JSONValue.parse(data)
    }

    /// Decodes the value into a `Decodable` type.
    public func decoded<T: Decodable>(as type: T.Type, context: String = "a JSON value") throws -> T {
        try ProviderJSON.decode(type, from: serializedData(), context: context)
    }
}

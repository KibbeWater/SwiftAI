import AIProviderSpec
import Foundation

/// A type that can appear in a model-generated structure.
///
/// Conformance supplies three things: the JSON Schema describing the value, a way to decode it
/// from JSON, and a way to build a *partial* snapshot of it from incomplete JSON. The last is
/// what makes streaming structured output work — see ``Partial``.
///
/// The primitive types, `Optional`, `Array`, and `Dictionary` conform already. Your own types
/// conform by applying the ``Structured(_:)`` macro, which writes all three implementations from
/// the declaration:
///
/// ```swift
/// @Structured("A recipe.")
/// struct Recipe {
///     @Guidance("The name of the dish.")
///     var name: String
///     var ingredients: [String]
/// }
/// ```
///
/// Hand-written conformances are fully supported for shapes the macro cannot express.
public protocol StructuredValue: Sendable {
    /// The type produced while the value is still arriving.
    ///
    /// For primitives this is the value itself: a `String` is either present or not. For
    /// structures the macro generates a mirror type whose every property is optional, so a
    /// half-received object is still a valid, inspectable value rather than an error. This is the
    /// same shape Apple's FoundationModels framework produces, and it is what lets a SwiftUI view
    /// bind directly to a response that is still being written.
    associatedtype Partial: Sendable = Self

    /// The schema describing values of this type.
    static var jsonSchema: JSONSchema { get }

    /// Whether a missing value is acceptable, which is true only for optionals.
    ///
    /// Decoding uses this to tell "the model omitted a required field" from "the model correctly
    /// omitted an optional one".
    static var acceptsMissingValue: Bool { get }

    /// Decodes a complete value.
    ///
    /// - Parameter json: The JSON to decode.
    /// - Throws: ``TypeValidationError`` when the JSON does not match this type.
    init(structuredJSON json: JSONValue) throws

    /// Builds a snapshot from possibly incomplete JSON.
    ///
    /// - Parameter json: The JSON received so far.
    /// - Returns: A snapshot, or `nil` when nothing useful can be recovered yet.
    static func partialValue(from json: JSONValue) -> Partial?
}

extension StructuredValue {
    public static var acceptsMissingValue: Bool { false }
}

extension StructuredValue where Partial == Self {
    /// Primitives are their own partial: a value has either arrived or it has not.
    public static func partialValue(from json: JSONValue) -> Self? {
        try? Self(structuredJSON: json)
    }
}

/// A type a model can generate as a complete, top-level result.
///
/// This is the constraint on ``generateObject(model:of:system:prompt:settings:)`` and
/// ``streamObject(model:of:system:prompt:settings:onFinish:)``. The ``Structured(_:)`` macro adds
/// it to structs. Enumerations conform to ``StructuredValue`` only, because a partially received
/// enumeration has no meaningful snapshot — use `generateObject(model:enumOf:…)` for those.
public protocol StructuredOutput: StructuredValue {
    /// Builds a snapshot from possibly incomplete JSON.
    ///
    /// Unlike ``StructuredValue/partialValue(from:)`` this never fails: an object with no fields
    /// yet is still a valid snapshot with every property `nil`.
    static func partial(from json: JSONValue) -> Partial
}

extension StructuredOutput {
    public static func partialValue(from json: JSONValue) -> Partial? {
        // An explicit null means the model declined the value, not that it is still arriving.
        json.isNull ? nil : partial(from: json)
    }
}

/// The snapshot type for a value, used by generated partial structures.
///
/// `PartialOf<[Ingredient]>` is `[Ingredient.Partial]`, `PartialOf<String?>` is `String`, and
/// `PartialOf<String>` is `String`. Expressing it this way means the macro never has to interpret
/// a property's declared type — it writes `PartialOf<DeclaredType>?` and the type system works
/// out the rest.
public typealias PartialOf<Value: StructuredValue> = Value.Partial

// MARK: - Primitive conformances

extension String: StructuredValue {
    public static var jsonSchema: JSONSchema { .string() }

    public init(structuredJSON json: JSONValue) throws {
        guard let value = json.stringValue else {
            throw TypeValidationError(message: "Expected a string but found \(json.typeName).", value: json)
        }
        self = value
    }
}

extension Bool: StructuredValue {
    public static var jsonSchema: JSONSchema { .boolean() }

    public init(structuredJSON json: JSONValue) throws {
        guard let value = json.boolValue else {
            throw TypeValidationError(message: "Expected a boolean but found \(json.typeName).", value: json)
        }
        self = value
    }
}

extension Double: StructuredValue {
    public static var jsonSchema: JSONSchema { .number() }

    public init(structuredJSON json: JSONValue) throws {
        guard let value = json.numberValue else {
            throw TypeValidationError(message: "Expected a number but found \(json.typeName).", value: json)
        }
        self = value
    }
}

extension Float: StructuredValue {
    public static var jsonSchema: JSONSchema { .number() }

    public init(structuredJSON json: JSONValue) throws {
        guard let value = json.numberValue else {
            throw TypeValidationError(message: "Expected a number but found \(json.typeName).", value: json)
        }
        self = Float(value)
    }
}

/// Decodes an integer, accepting a whole-numbered double.
///
/// Models routinely emit `3.0` where an integer was asked for, and rejecting that would fail a
/// generation over a formatting detail the model got semantically right.
private func decodeInteger<T: BinaryInteger>(_ json: JSONValue, as type: T.Type) throws -> T {
    if let value = json.intValue, let converted = T(exactly: value) { return converted }
    if let value = json.doubleValue, value.rounded() == value, let converted = T(exactly: value) {
        return converted
    }
    guard json.numberValue != nil else {
        throw TypeValidationError(message: "Expected an integer but found \(json.typeName).", value: json)
    }
    throw TypeValidationError(
        message: "\(json) is not representable as \(T.self).",
        value: json
    )
}

extension Int: StructuredValue {
    public static var jsonSchema: JSONSchema { .integer() }
    public init(structuredJSON json: JSONValue) throws { self = try decodeInteger(json, as: Int.self) }
}

extension Int32: StructuredValue {
    public static var jsonSchema: JSONSchema { .integer() }
    public init(structuredJSON json: JSONValue) throws { self = try decodeInteger(json, as: Int32.self) }
}

extension Int64: StructuredValue {
    public static var jsonSchema: JSONSchema { .integer() }
    public init(structuredJSON json: JSONValue) throws { self = try decodeInteger(json, as: Int64.self) }
}

extension UInt: StructuredValue {
    public static var jsonSchema: JSONSchema { .integer(minimum: 0) }
    public init(structuredJSON json: JSONValue) throws { self = try decodeInteger(json, as: UInt.self) }
}

extension Date: StructuredValue {
    public static var jsonSchema: JSONSchema { .string(format: .dateTime) }

    public init(structuredJSON json: JSONValue) throws {
        guard let text = json.stringValue else {
            throw TypeValidationError(
                message: "Expected an ISO 8601 timestamp but found \(json.typeName).",
                value: json
            )
        }
        guard let date = ISO8601Parsing.date(from: text) else {
            throw TypeValidationError(message: "'\(text)' is not a valid ISO 8601 timestamp.", value: json)
        }
        self = date
    }
}

/// Parses ISO 8601 timestamps, with and without fractional seconds.
///
/// `ISO8601DateFormatter` is not `Sendable`, and while Foundation's formatters are thread-safe on
/// Darwin, swift-corelibs-foundation makes no such guarantee. A lock around a shared pair of
/// formatters keeps this correct on every platform without allocating a formatter per timestamp.
private enum ISO8601Parsing {
    nonisolated(unsafe) private static let formatters: [ISO8601DateFormatter] = {
        let withFractionalSeconds = ISO8601DateFormatter()
        withFractionalSeconds.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return [withFractionalSeconds, plain]
    }()

    private static let lock = NSLock()

    static func date(from text: String) -> Date? {
        lock.lock()
        defer { lock.unlock() }
        for formatter in formatters {
            if let date = formatter.date(from: text) { return date }
        }
        return nil
    }
}

extension URL: StructuredValue {
    public static var jsonSchema: JSONSchema { .string(format: .uri) }

    public init(structuredJSON json: JSONValue) throws {
        guard let text = json.stringValue else {
            throw TypeValidationError(message: "Expected a URL string but found \(json.typeName).", value: json)
        }
        guard let url = URL(string: text) else {
            throw TypeValidationError(message: "'\(text)' is not a valid URL.", value: json)
        }
        self = url
    }
}

extension UUID: StructuredValue {
    public static var jsonSchema: JSONSchema { .string(format: .uuid) }

    public init(structuredJSON json: JSONValue) throws {
        guard let text = json.stringValue else {
            throw TypeValidationError(message: "Expected a UUID string but found \(json.typeName).", value: json)
        }
        guard let uuid = UUID(uuidString: text) else {
            throw TypeValidationError(message: "'\(text)' is not a valid UUID.", value: json)
        }
        self = uuid
    }
}

/// A raw JSON value passes through unchanged, which is what makes the dynamic-schema and
/// schema-less generation paths compose with everything else.
extension JSONValue: StructuredValue {
    public static var jsonSchema: JSONSchema { .any }
    public init(structuredJSON json: JSONValue) { self = json }
}

// MARK: - Container conformances

extension Optional: StructuredValue where Wrapped: StructuredValue {
    public typealias Partial = Wrapped.Partial

    public static var jsonSchema: JSONSchema { .nullable(Wrapped.jsonSchema) }
    public static var acceptsMissingValue: Bool { true }

    public init(structuredJSON json: JSONValue) throws {
        self = json.isNull ? nil : try Wrapped(structuredJSON: json)
    }

    public static func partialValue(from json: JSONValue) -> Wrapped.Partial? {
        json.isNull ? nil : Wrapped.partialValue(from: json)
    }
}

extension Array: StructuredValue where Element: StructuredValue {
    public typealias Partial = [Element.Partial]

    public static var jsonSchema: JSONSchema { .array(of: Element.jsonSchema) }

    public init(structuredJSON json: JSONValue) throws {
        guard let elements = json.arrayValue else {
            throw TypeValidationError(message: "Expected an array but found \(json.typeName).", value: json)
        }
        self = try elements.enumerated().map { index, element in
            do {
                return try Element(structuredJSON: element)
            } catch let error as TypeValidationError {
                throw error.prependingPath(String(index))
            }
        }
    }

    public static func partialValue(from json: JSONValue) -> [Element.Partial]? {
        guard let elements = json.arrayValue else { return nil }
        // A trailing element that has not arrived yet is dropped rather than represented as a
        // hole, so every snapshot is a prefix of the eventual array.
        return elements.compactMap(Element.partialValue(from:))
    }
}

extension Dictionary: StructuredValue where Key == String, Value: StructuredValue {
    public typealias Partial = [String: Value.Partial]

    /// An open object: no named properties, but every value must match `Value`'s schema.
    ///
    /// - Note: Providers' strict structured output modes reject open objects. ``JSONSchema``
    ///   leaves these untouched when strictifying, so a dictionary in a strict-mode request will
    ///   be refused by the provider. Prefer a declared structure where strict mode matters.
    public static var jsonSchema: JSONSchema {
        .object(
            ObjectConstraintsBuilder.openObject(valueSchema: Value.jsonSchema)
        )
    }

    public init(structuredJSON json: JSONValue) throws {
        guard let object = json.objectValue else {
            throw TypeValidationError(message: "Expected an object but found \(json.typeName).", value: json)
        }
        self = try object.reduce(into: [:]) { result, pair in
            do {
                result[pair.key] = try Value(structuredJSON: pair.value)
            } catch let error as TypeValidationError {
                throw error.prependingPath(pair.key)
            }
        }
    }

    public static func partialValue(from json: JSONValue) -> [String: Value.Partial]? {
        guard let object = json.objectValue else { return nil }
        return object.compactMapValues(Value.partialValue(from:))
    }
}

/// Builds the object constraints for an open dictionary schema.
///
/// Extracted so the conditional `Dictionary` conformance stays readable.
private enum ObjectConstraintsBuilder {
    static func openObject(valueSchema: JSONSchema) -> JSONSchema.ObjectConstraints {
        JSONSchema.ObjectConstraints(
            properties: [:],
            required: [],
            additionalProperties: .schema(valueSchema)
        )
    }
}

// MARK: - Diagnostics

extension JSONValue {
    /// The JSON type name, for error messages.
    var typeName: String {
        switch self {
        case .null: return "null"
        case .bool: return "a boolean"
        case .int, .double: return "a number"
        case .string: return "a string"
        case .array: return "an array"
        case .object: return "an object"
        }
    }
}

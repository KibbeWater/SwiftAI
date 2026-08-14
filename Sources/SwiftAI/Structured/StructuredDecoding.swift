import AIProviderSpec

/// Decoding helpers used by macro-generated code.
///
/// These exist so the generated `init(structuredJSON:)` is one readable line per property, and so
/// error messages carry the property path without every generated type reimplementing that.
public enum StructuredDecoding {
    /// Decodes a required or optional property from an object.
    ///
    /// - Parameters:
    ///   - type: The property's declared type.
    ///   - container: The enclosing object.
    ///   - key: The property's JSON key.
    /// - Returns: The decoded value.
    /// - Throws: ``TypeValidationError`` when the key is missing and the type is not optional, or
    ///   when the value has the wrong shape. The error's path names the property.
    public static func decode<Value: StructuredValue>(
        _ type: Value.Type,
        from container: JSONValue,
        key: String
    ) throws -> Value {
        guard case .object(let fields) = container else {
            throw TypeValidationError(
                message: "Expected an object but found \(container.typeName).",
                value: container
            )
        }
        guard let raw = fields[key] else {
            // An optional property may legitimately be absent; a required one may not.
            guard Value.acceptsMissingValue else {
                throw TypeValidationError(
                    message: "The required property '\(key)' is missing.",
                    path: key,
                    value: container
                )
            }
            return try Value(structuredJSON: .null)
        }
        do {
            return try Value(structuredJSON: raw)
        } catch let error as TypeValidationError {
            throw error.prependingPath(key)
        }
    }

    /// Builds a property's snapshot from an object that may still be arriving.
    ///
    /// - Returns: The snapshot, or `nil` when the property has not arrived yet.
    public static func partial<Value: StructuredValue>(
        _ type: Value.Type,
        from container: JSONValue,
        key: String
    ) -> Value.Partial? {
        guard case .object(let fields) = container, let raw = fields[key] else { return nil }
        return Value.partialValue(from: raw)
    }

    /// Decodes a string-backed enumeration.
    public static func decodeRawRepresentable<Value>(
        _ type: Value.Type,
        from json: JSONValue
    ) throws -> Value where Value: RawRepresentable, Value.RawValue == String {
        guard let text = json.stringValue else {
            throw TypeValidationError(
                message: "Expected a string but found \(json.typeName).",
                value: json
            )
        }
        guard let value = Value(rawValue: text) else {
            throw TypeValidationError(
                message: "'\(text)' is not one of the permitted values.",
                value: json
            )
        }
        return value
    }

    /// Reports an unrecognized enumeration case.
    ///
    /// Called by generated code for enumerations without raw values, where the permitted values
    /// are the case names.
    public static func unknownEnumerationValue(
        _ json: JSONValue,
        permitted: [String]
    ) -> TypeValidationError {
        guard let text = json.stringValue else {
            return TypeValidationError(
                message: "Expected a string but found \(json.typeName).",
                value: json
            )
        }
        return TypeValidationError(
            message: "'\(text)' is not one of: \(permitted.joined(separator: ", ")).",
            value: json
        )
    }
}

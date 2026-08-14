/// A dynamically typed JSON value.
///
/// `JSONValue` is the currency type used wherever the SDK has to carry JSON whose shape is not
/// known at compile time: tool call arguments, provider-specific options, raw provider payloads,
/// and the output of schema-less object generation.
///
/// Integers and floating-point numbers are represented by distinct cases so that a value which
/// arrived as `1` is re-serialized as `1` rather than `1.0`. Some providers are sensitive to this
/// distinction (notably for `seed` and token-count fields), so the round trip is preserved
/// exactly.
///
/// ```swift
/// let options: JSONValue = [
///     "reasoning": ["effort": "high"],
///     "parallel_tool_calls": true,
/// ]
/// options["reasoning"]?["effort"]?.stringValue  // "high"
/// ```
///
/// - Note: Equality is structural and strict: `.int(1)` does not equal `.double(1.0)`. Use
///   ``numberValue`` when you want to compare numerically regardless of representation.
public enum JSONValue: Sendable, Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

// MARK: - Convenience accessors

extension JSONValue {
    /// The wrapped Boolean, or `nil` if the value is not a `bool`.
    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    /// The wrapped integer, or `nil` if the value is not an `int`.
    ///
    /// A `.double` that holds an exact integral value is *not* converted; use ``numberValue``
    /// if you want lenient numeric access.
    public var intValue: Int? {
        guard case .int(let value) = self else { return nil }
        return value
    }

    /// The wrapped floating-point value, or `nil` if the value is not a `double`.
    public var doubleValue: Double? {
        guard case .double(let value) = self else { return nil }
        return value
    }

    /// The numeric value, converting between `int` and `double` representations as needed.
    public var numberValue: Double? {
        switch self {
        case .int(let value): return Double(value)
        case .double(let value): return value
        default: return nil
        }
    }

    /// The wrapped string, or `nil` if the value is not a `string`.
    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    /// The wrapped array, or `nil` if the value is not an `array`.
    public var arrayValue: [JSONValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    /// The wrapped object, or `nil` if the value is not an `object`.
    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    /// Whether the value is ``JSONValue/null``.
    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    /// Looks up a key in an object value.
    ///
    /// Returns `nil` when the receiver is not an object or the key is absent. Note that a present
    /// key holding JSON `null` returns `.null`, which lets callers distinguish "absent" from
    /// "explicitly null".
    public subscript(key: String) -> JSONValue? {
        guard case .object(let object) = self else { return nil }
        return object[key]
    }

    /// Looks up an index in an array value.
    ///
    /// Returns `nil` when the receiver is not an array or the index is out of bounds. Unlike
    /// `Array`'s subscript this does not trap, because the shape of provider payloads is not
    /// guaranteed.
    public subscript(index: Int) -> JSONValue? {
        guard case .array(let array) = self, array.indices.contains(index) else { return nil }
        return array[index]
    }
}

// MARK: - Literal conformances

extension JSONValue: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) { self = .null }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}

extension JSONValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) { self = .int(value) }
}

extension JSONValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) { self = .double(value) }
}

extension JSONValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}

extension JSONValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
}

extension JSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(Dictionary(elements, uniquingKeysWith: { _, last in last }))
    }
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
            return
        }
        // Order matters. `Bool` is attempted before the numeric types because some JSON decoders
        // will happily coerce `1` into `true`; attempting `Bool` first means a genuine boolean is
        // never misread as a number. `Int` precedes `Double` so integral values keep their
        // representation.
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Value is not representable as JSON."
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

// MARK: - Description

extension JSONValue: CustomStringConvertible {
    /// A compact JSON rendering of the value, with object keys sorted for stability.
    public var description: String { serialized(sortedKeys: true) }
}

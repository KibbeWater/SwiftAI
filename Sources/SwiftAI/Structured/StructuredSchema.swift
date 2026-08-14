import AIProviderSpec

/// A constraint attached to a property by the ``Guidance(_:_:)`` macro.
///
/// Constraints narrow what the model may produce. Providers with constrained decoding enforce
/// some of them during generation; the rest reach the model as part of the schema, where they
/// still improve adherence noticeably. They are not validated locally after generation, because a
/// value that violates a hint is usually better returned than rejected.
///
/// ```swift
/// @Guidance("How many people it serves.", .range(1...12))
/// var servings: Int
///
/// @Guidance("An ISO 4217 currency code.", .pattern("^[A-Z]{3}$"))
/// var currency: String
/// ```
public enum SchemaConstraint: Sendable, Hashable {
    /// Numeric bounds, inclusive.
    case bounds(minimum: Double?, maximum: Double?)
    /// Numeric bounds, exclusive.
    case exclusiveBounds(minimum: Double?, maximum: Double?)
    /// The value must be a multiple of this.
    case multipleOf(Double)
    /// String length bounds, inclusive.
    case lengthBounds(minimum: Int?, maximum: Int?)
    /// A regular expression the string must match, in ECMA-262 syntax.
    case pattern(String)
    /// A semantic format annotation.
    case format(JSONSchema.StringFormat)
    /// Element count bounds for an array, inclusive.
    case countBounds(minimum: Int?, maximum: Int?)
    /// The complete set of permitted string values.
    case allowedValues([String])

    // MARK: Ergonomic constructors

    /// Inclusive numeric bounds.
    public static func range(_ range: ClosedRange<Int>) -> SchemaConstraint {
        .bounds(minimum: Double(range.lowerBound), maximum: Double(range.upperBound))
    }

    /// Inclusive numeric bounds.
    public static func range(_ range: ClosedRange<Double>) -> SchemaConstraint {
        .bounds(minimum: range.lowerBound, maximum: range.upperBound)
    }

    /// An inclusive lower bound.
    public static func minimum(_ value: Double) -> SchemaConstraint {
        .bounds(minimum: value, maximum: nil)
    }

    /// An inclusive upper bound.
    public static func maximum(_ value: Double) -> SchemaConstraint {
        .bounds(minimum: nil, maximum: value)
    }

    /// Inclusive bounds on a string's length.
    public static func length(_ range: ClosedRange<Int>) -> SchemaConstraint {
        .lengthBounds(minimum: range.lowerBound, maximum: range.upperBound)
    }

    /// An inclusive lower bound on a string's length.
    public static func minimumLength(_ value: Int) -> SchemaConstraint {
        .lengthBounds(minimum: value, maximum: nil)
    }

    /// An inclusive upper bound on a string's length.
    public static func maximumLength(_ value: Int) -> SchemaConstraint {
        .lengthBounds(minimum: nil, maximum: value)
    }

    /// Inclusive bounds on an array's element count.
    public static func count(_ range: ClosedRange<Int>) -> SchemaConstraint {
        .countBounds(minimum: range.lowerBound, maximum: range.upperBound)
    }

    /// An inclusive lower bound on an array's element count.
    public static func minimumCount(_ value: Int) -> SchemaConstraint {
        .countBounds(minimum: value, maximum: nil)
    }

    /// An inclusive upper bound on an array's element count.
    public static func maximumCount(_ value: Int) -> SchemaConstraint {
        .countBounds(minimum: nil, maximum: value)
    }

    /// Restricts a string to a fixed set of values.
    public static func anyOf(_ values: [String]) -> SchemaConstraint {
        .allowedValues(values)
    }
}

extension SchemaConstraint {
    /// Applies the constraint to a schema, ignoring it where it does not apply.
    ///
    /// A count bound on a string, say, is silently dropped rather than being a compile error: the
    /// macro cannot check the property's type, and failing the build over a harmless annotation
    /// would be worse than ignoring it.
    ///
    /// Applying to a ``JSONSchema/nullable(_:)`` reaches through the wrapper, so a constraint on
    /// an optional property still lands on the value's schema.
    public func applied(to schema: JSONSchema) -> JSONSchema {
        if case .nullable(let wrapped) = schema {
            return .nullable(applied(to: wrapped))
        }

        switch (self, schema) {
        case (.bounds(let minimum, let maximum), .number(var constraints)):
            constraints.minimum = minimum ?? constraints.minimum
            constraints.maximum = maximum ?? constraints.maximum
            return .number(constraints)

        case (.bounds(let minimum, let maximum), .integer(var constraints)):
            constraints.minimum = minimum ?? constraints.minimum
            constraints.maximum = maximum ?? constraints.maximum
            return .integer(constraints)

        case (.exclusiveBounds(let minimum, let maximum), .number(var constraints)):
            constraints.exclusiveMinimum = minimum ?? constraints.exclusiveMinimum
            constraints.exclusiveMaximum = maximum ?? constraints.exclusiveMaximum
            return .number(constraints)

        case (.exclusiveBounds(let minimum, let maximum), .integer(var constraints)):
            constraints.exclusiveMinimum = minimum ?? constraints.exclusiveMinimum
            constraints.exclusiveMaximum = maximum ?? constraints.exclusiveMaximum
            return .integer(constraints)

        case (.multipleOf(let value), .number(var constraints)):
            constraints.multipleOf = value
            return .number(constraints)

        case (.multipleOf(let value), .integer(var constraints)):
            constraints.multipleOf = value
            return .integer(constraints)

        case (.lengthBounds(let minimum, let maximum), .string(var constraints)):
            constraints.minLength = minimum ?? constraints.minLength
            constraints.maxLength = maximum ?? constraints.maxLength
            return .string(constraints)

        case (.pattern(let pattern), .string(var constraints)):
            constraints.pattern = pattern
            return .string(constraints)

        case (.format(let format), .string(var constraints)):
            constraints.format = format
            return .string(constraints)

        case (.countBounds(let minimum, let maximum), .array(var constraints)):
            constraints.minItems = minimum ?? constraints.minItems
            constraints.maxItems = maximum ?? constraints.maxItems
            return .array(constraints)

        case (.allowedValues(let values), .string(let constraints)):
            // Replaces the string schema outright: an enumeration is more specific.
            return .enumeration(
                JSONSchema.EnumerationConstraints(
                    metadata: constraints.metadata,
                    values: values.map(JSONValue.string)
                )
            )

        default:
            return schema
        }
    }
}

/// Assembles schemas for macro-generated types.
///
/// The ``Structured(_:)`` macro emits calls to these functions rather than building a
/// ``JSONSchema`` literal itself. That keeps the generated code short and readable, and — more
/// importantly — means the macro never has to interpret a property's declared type. It writes
/// `schema(for: [Ingredient].self)` and the type system resolves the rest.
public enum StructuredSchema {
    /// One property of a generated object.
    public struct Property: Sendable {
        /// The JSON key.
        public var name: String
        /// The property type's schema.
        public var schema: JSONSchema
        /// The description from ``Guidance(_:_:)``, if any.
        public var description: String?
        /// The constraints from ``Guidance(_:_:)``.
        public var constraints: [SchemaConstraint]

        public init(
            name: String,
            schema: JSONSchema,
            description: String? = nil,
            constraints: [SchemaConstraint] = []
        ) {
            self.name = name
            self.schema = schema
            self.description = description
            self.constraints = constraints
        }

        /// The property's schema with its description and constraints applied.
        public var resolvedSchema: JSONSchema {
            var resolved = constraints.reduce(schema) { $1.applied(to: $0) }
            if let description { resolved.schemaDescription = description }
            return resolved
        }
    }

    /// The schema of a type, as a function so the macro can name a type rather than a schema.
    public static func schema<Value: StructuredValue>(for type: Value.Type) -> JSONSchema {
        Value.jsonSchema
    }

    /// Builds an object schema from a type's properties.
    ///
    /// A property is required exactly when its type is not optional, which is read off the
    /// property's own schema rather than being tracked separately.
    public static func object(description: String?, properties: [Property]) -> JSONSchema {
        var schemas: [String: JSONSchema] = [:]
        var required: [String] = []
        for property in properties {
            let resolved = property.resolvedSchema
            schemas[property.name] = resolved
            if !resolved.isNullable { required.append(property.name) }
        }
        return .object(
            properties: schemas,
            required: required,
            description: description
        )
    }

    /// Builds an enumeration schema from a fixed set of string values.
    public static func enumeration(_ values: [String], description: String?) -> JSONSchema {
        .enumeration(values, description: description)
    }
}

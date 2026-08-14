/// A JSON Schema description of a value.
///
/// `JSONSchema` covers the subset of JSON Schema draft 2020-12 that language model providers
/// actually accept for tool parameters and structured output. It is a value type, so schemas can
/// be built up, stored, compared, and — importantly — rewritten: providers routinely need to
/// adapt a schema to their own dialect, which ``transformed(_:)`` makes straightforward.
///
/// Schemas are usually generated from a Swift type by the `@Structured` macro rather than written
/// by hand, but the hand-written form is a supported first-class path for shapes that are only
/// known at runtime:
///
/// ```swift
/// let schema = JSONSchema.object(
///     properties: [
///         "city": .string(description: "The city to look up."),
///         "days": .integer(description: "Forecast length.", minimum: 1, maximum: 7),
///     ],
///     required: ["city"],
///     description: "Arguments for a weather lookup."
/// )
/// ```
public indirect enum JSONSchema: Sendable, Hashable {
    /// An empty schema, which accepts any value.
    case any(Metadata)
    /// The JSON `null` value.
    case null(Metadata)
    /// A boolean.
    case boolean(Metadata)
    /// A string, optionally constrained by length, pattern, or format.
    case string(StringConstraints)
    /// A floating-point number.
    case number(NumberConstraints)
    /// An integer.
    case integer(NumberConstraints)
    /// An array of homogeneous elements.
    case array(ArrayConstraints)
    /// An object with named properties.
    case object(ObjectConstraints)
    /// A closed set of permitted values.
    case enumeration(EnumerationConstraints)
    /// A value matching at least one of several schemas.
    case anyOf(CompositionConstraints)
    /// A value matching exactly one of several schemas.
    case oneOf(CompositionConstraints)
    /// A reference to a schema defined elsewhere in the document, such as `"#/$defs/Address"`.
    case reference(String, Metadata)
    /// A schema whose value may also be JSON `null`.
    ///
    /// This is kept as a distinct case rather than being folded into a type array so that
    /// providers with differing conventions for optionality — a `["string", "null"]` type array,
    /// an `anyOf` with a null branch, or a `nullable: true` flag — can each render it natively.
    case nullable(JSONSchema)
}

// MARK: - Metadata

extension JSONSchema {
    /// Annotations that may accompany a schema of any type.
    public struct Metadata: Sendable, Hashable {
        /// A natural-language description of the value.
        ///
        /// This is the single highest-leverage field in a schema: models rely on it far more
        /// than on structural constraints when deciding what to put in a field.
        public var description: String?

        /// A short human-readable name for the schema.
        public var title: String?

        /// The value to assume when none is supplied.
        public var `default`: JSONValue?

        /// Illustrative values.
        public var examples: [JSONValue]

        /// Reusable subschemas, emitted as `$defs`. Normally present only on a root schema.
        public var definitions: [String: JSONSchema]

        /// Additional keywords to emit verbatim.
        ///
        /// An escape hatch for provider-specific vocabulary that this type does not model.
        /// Keys here overwrite generated keywords of the same name.
        public var extensions: [String: JSONValue]

        public init(
            description: String? = nil,
            title: String? = nil,
            default: JSONValue? = nil,
            examples: [JSONValue] = [],
            definitions: [String: JSONSchema] = [:],
            extensions: [String: JSONValue] = [:]
        ) {
            self.description = description
            self.title = title
            self.default = `default`
            self.examples = examples
            self.definitions = definitions
            self.extensions = extensions
        }
    }

    /// Constraints on a string value.
    public struct StringConstraints: Sendable, Hashable {
        public var metadata: Metadata
        /// A semantic format hint such as `date-time` or `uri`.
        public var format: StringFormat?
        /// A regular expression the value must match, in ECMA-262 syntax.
        public var pattern: String?
        public var minLength: Int?
        public var maxLength: Int?

        public init(
            metadata: Metadata = .init(),
            format: StringFormat? = nil,
            pattern: String? = nil,
            minLength: Int? = nil,
            maxLength: Int? = nil
        ) {
            self.metadata = metadata
            self.format = format
            self.pattern = pattern
            self.minLength = minLength
            self.maxLength = maxLength
        }
    }

    /// A standard `format` annotation for string values.
    public struct StringFormat: Sendable, Hashable, RawRepresentable, ExpressibleByStringLiteral {
        public var rawValue: String
        public init(rawValue: String) { self.rawValue = rawValue }
        public init(stringLiteral value: String) { self.rawValue = value }

        /// An RFC 3339 timestamp, such as `2026-08-14T09:41:00Z`.
        public static let dateTime = StringFormat(rawValue: "date-time")
        /// An RFC 3339 full-date, such as `2026-08-14`.
        public static let date = StringFormat(rawValue: "date")
        /// An RFC 3339 full-time, such as `09:41:00Z`.
        public static let time = StringFormat(rawValue: "time")
        /// An ISO 8601 duration.
        public static let duration = StringFormat(rawValue: "duration")
        /// An email address.
        public static let email = StringFormat(rawValue: "email")
        /// A URI.
        public static let uri = StringFormat(rawValue: "uri")
        /// A UUID.
        public static let uuid = StringFormat(rawValue: "uuid")
    }

    /// Constraints on a numeric value.
    public struct NumberConstraints: Sendable, Hashable {
        public var metadata: Metadata
        public var minimum: Double?
        public var maximum: Double?
        public var exclusiveMinimum: Double?
        public var exclusiveMaximum: Double?
        public var multipleOf: Double?

        public init(
            metadata: Metadata = .init(),
            minimum: Double? = nil,
            maximum: Double? = nil,
            exclusiveMinimum: Double? = nil,
            exclusiveMaximum: Double? = nil,
            multipleOf: Double? = nil
        ) {
            self.metadata = metadata
            self.minimum = minimum
            self.maximum = maximum
            self.exclusiveMinimum = exclusiveMinimum
            self.exclusiveMaximum = exclusiveMaximum
            self.multipleOf = multipleOf
        }
    }

    /// Constraints on an array value.
    public struct ArrayConstraints: Sendable, Hashable {
        public var metadata: Metadata
        /// The schema every element must satisfy.
        public var items: JSONSchema
        public var minItems: Int?
        public var maxItems: Int?
        public var uniqueItems: Bool?

        public init(
            metadata: Metadata = .init(),
            items: JSONSchema,
            minItems: Int? = nil,
            maxItems: Int? = nil,
            uniqueItems: Bool? = nil
        ) {
            self.metadata = metadata
            self.items = items
            self.minItems = minItems
            self.maxItems = maxItems
            self.uniqueItems = uniqueItems
        }
    }

    /// Constraints on an object value.
    public struct ObjectConstraints: Sendable, Hashable {
        public var metadata: Metadata
        /// The schema for each named property.
        public var properties: [String: JSONSchema]
        /// The properties that must be present.
        public var required: [String]
        /// How properties not named in ``properties`` are treated.
        public var additionalProperties: AdditionalProperties

        public init(
            metadata: Metadata = .init(),
            properties: [String: JSONSchema] = [:],
            required: [String] = [],
            additionalProperties: AdditionalProperties = .disallowed
        ) {
            self.metadata = metadata
            self.properties = properties
            self.required = required
            self.additionalProperties = additionalProperties
        }
    }

    /// How an object schema treats properties it does not name.
    public enum AdditionalProperties: Sendable, Hashable {
        /// Unnamed properties are permitted. Emitted as `additionalProperties: true`.
        case allowed
        /// Unnamed properties are rejected. Emitted as `additionalProperties: false`.
        ///
        /// This is the default because several providers require it for their strict structured
        /// output modes, and because a closed object is almost always what a Swift type means.
        case disallowed
        /// Unnamed properties are permitted if they match a schema.
        case schema(JSONSchema)
    }

    /// A closed set of permitted values.
    public struct EnumerationConstraints: Sendable, Hashable {
        public var metadata: Metadata
        /// The permitted values.
        public var values: [JSONValue]

        public init(metadata: Metadata = .init(), values: [JSONValue]) {
            self.metadata = metadata
            self.values = values
        }
    }

    /// A composition of several subschemas.
    public struct CompositionConstraints: Sendable, Hashable {
        public var metadata: Metadata
        public var subschemas: [JSONSchema]

        public init(metadata: Metadata = .init(), subschemas: [JSONSchema]) {
            self.metadata = metadata
            self.subschemas = subschemas
        }
    }
}

// MARK: - Ergonomic constructors

extension JSONSchema {
    /// An empty schema that accepts any value.
    public static var any: JSONSchema { .any(Metadata()) }

    /// A schema accepting only JSON `null`.
    public static var null: JSONSchema { .null(Metadata()) }

    /// A boolean schema.
    public static func boolean(description: String? = nil) -> JSONSchema {
        .boolean(Metadata(description: description))
    }

    /// A string schema.
    public static func string(
        description: String? = nil,
        format: StringFormat? = nil,
        pattern: String? = nil,
        minLength: Int? = nil,
        maxLength: Int? = nil
    ) -> JSONSchema {
        .string(
            StringConstraints(
                metadata: Metadata(description: description),
                format: format,
                pattern: pattern,
                minLength: minLength,
                maxLength: maxLength
            )
        )
    }

    /// A floating-point number schema.
    public static func number(
        description: String? = nil,
        minimum: Double? = nil,
        maximum: Double? = nil,
        multipleOf: Double? = nil
    ) -> JSONSchema {
        .number(
            NumberConstraints(
                metadata: Metadata(description: description),
                minimum: minimum,
                maximum: maximum,
                multipleOf: multipleOf
            )
        )
    }

    /// An integer schema.
    public static func integer(
        description: String? = nil,
        minimum: Double? = nil,
        maximum: Double? = nil,
        multipleOf: Double? = nil
    ) -> JSONSchema {
        .integer(
            NumberConstraints(
                metadata: Metadata(description: description),
                minimum: minimum,
                maximum: maximum,
                multipleOf: multipleOf
            )
        )
    }

    /// An array schema.
    public static func array(
        of items: JSONSchema,
        description: String? = nil,
        minItems: Int? = nil,
        maxItems: Int? = nil
    ) -> JSONSchema {
        .array(
            ArrayConstraints(
                metadata: Metadata(description: description),
                items: items,
                minItems: minItems,
                maxItems: maxItems
            )
        )
    }

    /// An object schema.
    public static func object(
        properties: [String: JSONSchema],
        required: [String],
        description: String? = nil,
        additionalProperties: AdditionalProperties = .disallowed
    ) -> JSONSchema {
        .object(
            ObjectConstraints(
                metadata: Metadata(description: description),
                properties: properties,
                required: required,
                additionalProperties: additionalProperties
            )
        )
    }

    /// A schema permitting only the given string values.
    public static func enumeration(_ values: [String], description: String? = nil) -> JSONSchema {
        .enumeration(
            EnumerationConstraints(
                metadata: Metadata(description: description),
                values: values.map(JSONValue.string)
            )
        )
    }

    /// A schema matching at least one of the given subschemas.
    public static func anyOf(_ subschemas: [JSONSchema], description: String? = nil) -> JSONSchema {
        .anyOf(CompositionConstraints(metadata: Metadata(description: description), subschemas: subschemas))
    }

    /// A schema matching exactly one of the given subschemas.
    public static func oneOf(_ subschemas: [JSONSchema], description: String? = nil) -> JSONSchema {
        .oneOf(CompositionConstraints(metadata: Metadata(description: description), subschemas: subschemas))
    }
}

// MARK: - Metadata access

extension JSONSchema {
    /// The annotations attached to this schema.
    ///
    /// Setting this on ``JSONSchema/nullable(_:)`` forwards to the wrapped schema, since the
    /// nullable wrapper carries no annotations of its own.
    public var metadata: Metadata {
        get {
            switch self {
            case .any(let metadata), .null(let metadata), .boolean(let metadata): return metadata
            case .string(let constraints): return constraints.metadata
            case .number(let constraints), .integer(let constraints): return constraints.metadata
            case .array(let constraints): return constraints.metadata
            case .object(let constraints): return constraints.metadata
            case .enumeration(let constraints): return constraints.metadata
            case .anyOf(let constraints), .oneOf(let constraints): return constraints.metadata
            case .reference(_, let metadata): return metadata
            case .nullable(let wrapped): return wrapped.metadata
            }
        }
        set {
            switch self {
            case .any: self = .any(newValue)
            case .null: self = .null(newValue)
            case .boolean: self = .boolean(newValue)
            case .string(var constraints): constraints.metadata = newValue; self = .string(constraints)
            case .number(var constraints): constraints.metadata = newValue; self = .number(constraints)
            case .integer(var constraints): constraints.metadata = newValue; self = .integer(constraints)
            case .array(var constraints): constraints.metadata = newValue; self = .array(constraints)
            case .object(var constraints): constraints.metadata = newValue; self = .object(constraints)
            case .enumeration(var constraints): constraints.metadata = newValue; self = .enumeration(constraints)
            case .anyOf(var constraints): constraints.metadata = newValue; self = .anyOf(constraints)
            case .oneOf(var constraints): constraints.metadata = newValue; self = .oneOf(constraints)
            case .reference(let pointer, _): self = .reference(pointer, newValue)
            case .nullable(var wrapped): wrapped.metadata = newValue; self = .nullable(wrapped)
            }
        }
    }

    /// The natural-language description of the value this schema describes.
    public var schemaDescription: String? {
        get { metadata.description }
        set { metadata.description = newValue }
    }

    /// Returns a copy of the schema with the given description.
    public func describing(_ description: String?) -> JSONSchema {
        var copy = self
        copy.schemaDescription = description
        return copy
    }

    /// Whether the schema permits JSON `null`.
    public var isNullable: Bool {
        if case .nullable = self { return true }
        if case .null = self { return true }
        return false
    }

    /// The schema with any nullable wrapper removed.
    public var unwrappingNullable: JSONSchema {
        if case .nullable(let wrapped) = self { return wrapped.unwrappingNullable }
        return self
    }
}

// MARK: - Transformation

extension JSONSchema {
    /// Rewrites the schema bottom-up, applying `transform` to every node.
    ///
    /// Children are transformed before their parent, so a transform sees already-rewritten
    /// subschemas. Providers use this to adapt schemas to their own dialect — for example,
    /// removing keywords a provider rejects, or forcing every object closed:
    ///
    /// ```swift
    /// let strict = schema.transformed { node in
    ///     guard case .object(var constraints) = node else { return node }
    ///     constraints.additionalProperties = .disallowed
    ///     constraints.required = constraints.properties.keys.sorted()
    ///     return .object(constraints)
    /// }
    /// ```
    ///
    /// - Parameter transform: A closure applied to each node after its children are rewritten.
    /// - Returns: The rewritten schema.
    public func transformed(_ transform: (JSONSchema) -> JSONSchema) -> JSONSchema {
        let rewritten: JSONSchema
        switch self {
        case .any, .null, .boolean, .string, .number, .integer, .enumeration, .reference:
            rewritten = self

        case .array(var constraints):
            constraints.items = constraints.items.transformed(transform)
            constraints.metadata.definitions = constraints.metadata.definitions
                .mapValues { $0.transformed(transform) }
            rewritten = .array(constraints)

        case .object(var constraints):
            constraints.properties = constraints.properties.mapValues { $0.transformed(transform) }
            if case .schema(let additional) = constraints.additionalProperties {
                constraints.additionalProperties = .schema(additional.transformed(transform))
            }
            constraints.metadata.definitions = constraints.metadata.definitions
                .mapValues { $0.transformed(transform) }
            rewritten = .object(constraints)

        case .anyOf(var constraints):
            constraints.subschemas = constraints.subschemas.map { $0.transformed(transform) }
            rewritten = .anyOf(constraints)

        case .oneOf(var constraints):
            constraints.subschemas = constraints.subschemas.map { $0.transformed(transform) }
            rewritten = .oneOf(constraints)

        case .nullable(let wrapped):
            rewritten = .nullable(wrapped.transformed(transform))
        }
        return transform(rewritten)
    }
}

/// How a schema renders optionality, which differs between providers.
public enum NullabilityStyle: Sendable, Hashable {
    /// Emit a type array, as in `{"type": ["string", "null"]}`.
    ///
    /// This is standard JSON Schema and what OpenAI's structured output mode expects.
    case typeArray

    /// Emit `{"anyOf": [<schema>, {"type": "null"}]}`.
    ///
    /// Useful for dialects that accept only a single string for `type`.
    case anyOfNull

    /// Emit a sibling `{"nullable": true}` flag, as OpenAPI 3.0 and Google's Gemini API expect.
    case nullableFlag

    /// Drop nullability entirely and emit the wrapped schema.
    ///
    /// Appropriate for dialects with no way to express it; combine with omitting the property
    /// from `required` so the field remains optional in practice.
    case omit
}

/// Options controlling how a ``JSONSchema`` is rendered to JSON.
///
/// The defaults produce standard draft 2020-12 output. Providers override individual options to
/// match their own dialect rather than maintaining separate rendering code.
public struct JSONSchemaEncodingOptions: Sendable, Hashable {
    /// How optionality is expressed. Defaults to ``NullabilityStyle/typeArray``.
    public var nullability: NullabilityStyle

    /// Whether to emit the `$schema` dialect declaration on the root schema. Defaults to `false`,
    /// because most providers reject it.
    public var includesDialectDeclaration: Bool

    /// Whether to emit `additionalProperties`. Defaults to `true`.
    ///
    /// Gemini rejects the keyword outright, so its provider disables it.
    public var includesAdditionalProperties: Bool

    /// Whether to emit `description`, `title`, `default`, and `examples`. Defaults to `true`.
    public var includesAnnotations: Bool

    /// Whether to emit numeric and string constraints such as `minimum` and `pattern`.
    ///
    /// Defaults to `true`. Providers that reject unknown keywords in strict modes turn this off.
    public var includesConstraints: Bool

    /// Keywords to drop from every node, by name.
    ///
    /// A blunt instrument for dialects that reject specific vocabulary.
    public var excludedKeywords: Set<String>

    public init(
        nullability: NullabilityStyle = .typeArray,
        includesDialectDeclaration: Bool = false,
        includesAdditionalProperties: Bool = true,
        includesAnnotations: Bool = true,
        includesConstraints: Bool = true,
        excludedKeywords: Set<String> = []
    ) {
        self.nullability = nullability
        self.includesDialectDeclaration = includesDialectDeclaration
        self.includesAdditionalProperties = includesAdditionalProperties
        self.includesAnnotations = includesAnnotations
        self.includesConstraints = includesConstraints
        self.excludedKeywords = excludedKeywords
    }

    /// Standard JSON Schema draft 2020-12 output.
    public static let standard = JSONSchemaEncodingOptions()

    /// Output accepted by OpenAI's strict structured output and strict function calling modes.
    ///
    /// Strict mode requires every property to appear in `required` and every object to set
    /// `additionalProperties: false`; optional fields are expressed by making them nullable
    /// instead. Callers should pair this with ``JSONSchema/strictified()``.
    public static let openAIStrict = JSONSchemaEncodingOptions(
        nullability: .typeArray,
        includesAdditionalProperties: true,
        includesConstraints: false
    )

    /// Output accepted by Google's Gemini API, which uses an OpenAPI 3.0 schema subset.
    public static let googleGenerativeAI = JSONSchemaEncodingOptions(
        nullability: .nullableFlag,
        includesAdditionalProperties: false,
        includesConstraints: false,
        excludedKeywords: ["$defs", "$ref", "additionalProperties", "examples", "default", "oneOf"]
    )
}

extension JSONSchema {
    /// Renders the schema as a ``JSONValue``.
    ///
    /// - Parameter options: Dialect options. Defaults to standard draft 2020-12 output.
    /// - Returns: The schema as JSON, ready to embed in a request body.
    public func jsonValue(options: JSONSchemaEncodingOptions = .standard) -> JSONValue {
        var object = encode(options: options, isRoot: true)
        if options.includesDialectDeclaration, case .object(var fields) = object {
            fields["$schema"] = .string("https://json-schema.org/draft/2020-12/schema")
            object = .object(fields)
        }
        return object
    }

    private func encode(options: JSONSchemaEncodingOptions, isRoot: Bool) -> JSONValue {
        var fields: [String: JSONValue] = [:]

        switch self {
        case .any:
            break

        case .null:
            fields["type"] = .string("null")

        case .boolean:
            fields["type"] = .string("boolean")

        case .string(let constraints):
            fields["type"] = .string("string")
            if options.includesConstraints {
                if let format = constraints.format { fields["format"] = .string(format.rawValue) }
                if let pattern = constraints.pattern { fields["pattern"] = .string(pattern) }
                if let minLength = constraints.minLength { fields["minLength"] = .int(minLength) }
                if let maxLength = constraints.maxLength { fields["maxLength"] = .int(maxLength) }
            }

        case .number(let constraints), .integer(let constraints):
            if case .integer = self { fields["type"] = .string("integer") }
            else { fields["type"] = .string("number") }
            if options.includesConstraints {
                if let minimum = constraints.minimum { fields["minimum"] = .double(minimum) }
                if let maximum = constraints.maximum { fields["maximum"] = .double(maximum) }
                if let value = constraints.exclusiveMinimum { fields["exclusiveMinimum"] = .double(value) }
                if let value = constraints.exclusiveMaximum { fields["exclusiveMaximum"] = .double(value) }
                if let value = constraints.multipleOf { fields["multipleOf"] = .double(value) }
            }

        case .array(let constraints):
            fields["type"] = .string("array")
            fields["items"] = constraints.items.encode(options: options, isRoot: false)
            if options.includesConstraints {
                if let minItems = constraints.minItems { fields["minItems"] = .int(minItems) }
                if let maxItems = constraints.maxItems { fields["maxItems"] = .int(maxItems) }
                if let unique = constraints.uniqueItems { fields["uniqueItems"] = .bool(unique) }
            }

        case .object(let constraints):
            fields["type"] = .string("object")
            fields["properties"] = .object(
                constraints.properties.mapValues { $0.encode(options: options, isRoot: false) }
            )
            // `required` is sorted so that identical schemas serialize identically, which keeps
            // request bodies stable for upstream prompt caching and makes tests deterministic.
            fields["required"] = .array(constraints.required.sorted().map(JSONValue.string))
            if options.includesAdditionalProperties {
                switch constraints.additionalProperties {
                case .allowed: fields["additionalProperties"] = .bool(true)
                case .disallowed: fields["additionalProperties"] = .bool(false)
                case .schema(let schema):
                    fields["additionalProperties"] = schema.encode(options: options, isRoot: false)
                }
            }

        case .enumeration(let constraints):
            fields["enum"] = .array(constraints.values)
            // A homogeneous string enumeration also gets an explicit type, which materially
            // improves adherence on several providers.
            if constraints.values.allSatisfy({ $0.stringValue != nil }) {
                fields["type"] = .string("string")
            }

        case .anyOf(let constraints):
            fields["anyOf"] = .array(constraints.subschemas.map { $0.encode(options: options, isRoot: false) })

        case .oneOf(let constraints):
            fields["oneOf"] = .array(constraints.subschemas.map { $0.encode(options: options, isRoot: false) })

        case .reference(let pointer, _):
            fields["$ref"] = .string(pointer)

        case .nullable(let wrapped):
            return encodeNullable(wrapped, options: options, isRoot: isRoot)
        }

        applyMetadata(metadata, to: &fields, options: options, isRoot: isRoot)
        for keyword in options.excludedKeywords { fields.removeValue(forKey: keyword) }
        return .object(fields)
    }

    private func encodeNullable(
        _ wrapped: JSONSchema,
        options: JSONSchemaEncodingOptions,
        isRoot: Bool
    ) -> JSONValue {
        let encoded = wrapped.encode(options: options, isRoot: isRoot)
        guard case .object(var fields) = encoded else { return encoded }

        switch options.nullability {
        case .omit:
            break

        case .nullableFlag:
            fields["nullable"] = .bool(true)

        case .typeArray:
            switch fields["type"] {
            case .string(let type):
                fields["type"] = .array([.string(type), .string("null")])
            case .array(let types):
                if !types.contains(.string("null")) { fields["type"] = .array(types + [.string("null")]) }
            default:
                // No `type` keyword to extend — `anyOf` is the only correct rendering.
                return .object(["anyOf": .array([encoded, .object(["type": .string("null")])])])
            }

        case .anyOfNull:
            var wrapper: [String: JSONValue] = [
                "anyOf": .array([encoded, .object(["type": .string("null")])])
            ]
            // Hoist the description so it stays visible on the property itself.
            if options.includesAnnotations, let description = fields["description"] {
                wrapper["description"] = description
            }
            return .object(wrapper)
        }

        for keyword in options.excludedKeywords { fields.removeValue(forKey: keyword) }
        return .object(fields)
    }

    private func applyMetadata(
        _ metadata: Metadata,
        to fields: inout [String: JSONValue],
        options: JSONSchemaEncodingOptions,
        isRoot: Bool
    ) {
        if options.includesAnnotations {
            if let description = metadata.description { fields["description"] = .string(description) }
            if let title = metadata.title { fields["title"] = .string(title) }
            if let value = metadata.default { fields["default"] = value }
            if !metadata.examples.isEmpty { fields["examples"] = .array(metadata.examples) }
        }
        if isRoot, !metadata.definitions.isEmpty {
            fields["$defs"] = .object(
                metadata.definitions.mapValues { $0.encode(options: options, isRoot: false) }
            )
        }
        // Extensions are applied last so they can override anything generated above.
        for (keyword, value) in metadata.extensions { fields[keyword] = value }
    }
}

// MARK: - Strict mode

extension JSONSchema {
    /// Returns a copy rewritten to satisfy providers' strict structured output rules.
    ///
    /// Strict modes — OpenAI's in particular — demand that every property of an object appear in
    /// `required` and that every object set `additionalProperties: false`. Genuinely optional
    /// properties are preserved by making them nullable, which conveys the same meaning: the
    /// model must emit the key, but may emit `null` for it.
    ///
    /// - Returns: An equivalent schema that satisfies strict-mode constraints.
    public func strictified() -> JSONSchema {
        transformed { node in
            guard case .object(var constraints) = node else { return node }
            let optionalKeys = Set(constraints.properties.keys).subtracting(constraints.required)
            for key in optionalKeys {
                guard let property = constraints.properties[key], !property.isNullable else { continue }
                constraints.properties[key] = .nullable(property)
            }
            constraints.required = constraints.properties.keys.sorted()
            constraints.additionalProperties = .disallowed
            return .object(constraints)
        }
    }
}

import Testing

@testable import AIProviderSpec

@Suite("JSONSchema")
struct JSONSchemaTests {
    /// A representative schema exercising objects, arrays, enumerations, and optionality.
    static let address = JSONSchema.object(
        properties: [
            "street": .string(description: "Street and number."),
            "country": .enumeration(["SE", "US", "GB"], description: "ISO 3166-1 alpha-2 code."),
        ],
        required: ["street", "country"]
    )

    static let person = JSONSchema.object(
        properties: [
            "name": .string(description: "Full name.", minLength: 1),
            "age": .integer(description: "Age in years.", minimum: 0, maximum: 130),
            "nickname": .nullable(.string(description: "What they go by.")),
            "addresses": .array(of: address, description: "Known addresses.", minItems: 1),
        ],
        required: ["name", "age", "addresses"],
        description: "A person."
    )

    // MARK: - Encoding

    @Test("Encodes an object schema in standard form")
    func encodesStandardForm() {
        let json = Self.person.jsonValue()

        #expect(json["type"]?.stringValue == "object")
        #expect(json["description"]?.stringValue == "A person.")
        #expect(json["additionalProperties"]?.boolValue == false)
        // `required` is sorted so identical schemas serialize identically.
        #expect(json["required"] == .array(["addresses", "age", "name"]))
        #expect(json["properties"]?["name"]?["type"]?.stringValue == "string")
        #expect(json["properties"]?["name"]?["minLength"]?.intValue == 1)
        #expect(json["properties"]?["age"]?["maximum"]?.numberValue == 130)
        #expect(json["properties"]?["addresses"]?["items"]?["type"]?.stringValue == "object")
    }

    @Test("Encodes a string enumeration with an explicit type")
    func encodesEnumeration() {
        let json = JSONSchema.enumeration(["a", "b"]).jsonValue()
        #expect(json["enum"] == .array(["a", "b"]))
        // The explicit type materially improves adherence on several providers.
        #expect(json["type"]?.stringValue == "string")
    }

    @Test("Omits the dialect declaration unless requested")
    func dialectDeclarationIsOptOut() {
        #expect(JSONSchema.string().jsonValue()["$schema"] == nil)

        var options = JSONSchemaEncodingOptions.standard
        options.includesDialectDeclaration = true
        #expect(JSONSchema.string().jsonValue(options: options)["$schema"] != nil)
    }

    @Test("Emits $defs only on the root")
    func definitionsOnlyOnRoot() {
        var root = JSONSchema.object(properties: ["a": .reference("#/$defs/A", .init())], required: ["a"])
        root.metadata.definitions = ["A": .string(description: "A value.")]

        let json = root.jsonValue()
        #expect(json["$defs"]?["A"]?["type"]?.stringValue == "string")
        #expect(json["properties"]?["a"]?["$ref"]?.stringValue == "#/$defs/A")
        #expect(json["properties"]?["a"]?["$defs"] == nil)
    }

    // MARK: - Nullability

    @Test("Renders nullability as a type array by default")
    func nullabilityAsTypeArray() {
        let json = JSONSchema.nullable(.string(description: "Maybe.")).jsonValue()
        #expect(json["type"] == .array(["string", "null"]))
        #expect(json["description"]?.stringValue == "Maybe.")
    }

    @Test("Renders nullability as an anyOf branch")
    func nullabilityAsAnyOf() {
        var options = JSONSchemaEncodingOptions.standard
        options.nullability = .anyOfNull

        let json = JSONSchema.nullable(.string(description: "Maybe.")).jsonValue(options: options)
        #expect(json["anyOf"]?[1]?["type"]?.stringValue == "null")
        // The description is hoisted so it stays visible on the property itself.
        #expect(json["description"]?.stringValue == "Maybe.")
    }

    @Test("Renders nullability as an OpenAPI flag")
    func nullabilityAsFlag() {
        var options = JSONSchemaEncodingOptions.standard
        options.nullability = .nullableFlag

        let json = JSONSchema.nullable(.string()).jsonValue(options: options)
        #expect(json["nullable"]?.boolValue == true)
        #expect(json["type"]?.stringValue == "string")
    }

    @Test("Drops nullability entirely when the dialect cannot express it")
    func nullabilityOmitted() {
        var options = JSONSchemaEncodingOptions.standard
        options.nullability = .omit

        let json = JSONSchema.nullable(.string()).jsonValue(options: options)
        #expect(json["type"]?.stringValue == "string")
        #expect(json["nullable"] == nil)
    }

    @Test("Falls back to anyOf when there is no type keyword to extend")
    func nullabilityFallsBackToAnyOf() {
        // An `anyOf` node has no `type`, so a type array cannot express its nullability.
        let composed = JSONSchema.anyOf([.string(), .integer()])
        let json = JSONSchema.nullable(composed).jsonValue()
        #expect(json["anyOf"]?[1]?["type"]?.stringValue == "null")
    }

    @Test("Nullable reports through the wrapper")
    func nullableInspection() {
        let schema = JSONSchema.nullable(.string(description: "Text."))
        #expect(schema.isNullable)
        #expect(schema.schemaDescription == "Text.")
        #expect(schema.unwrappingNullable.isNullable == false)
    }

    // MARK: - Provider dialects

    @Test("Google dialect drops keywords Gemini rejects")
    func googleDialect() {
        let json = Self.person.jsonValue(options: .googleGenerativeAI)

        #expect(json["additionalProperties"] == nil)
        #expect(json["properties"]?["name"]?["minLength"] == nil)
        #expect(json["properties"]?["nickname"]?["nullable"]?.boolValue == true)
        // Descriptions survive: they are the highest-value part of a schema.
        #expect(json["properties"]?["name"]?["description"]?.stringValue == "Full name.")
    }

    @Test("OpenAI strict dialect drops constraint keywords")
    func openAIStrictDialect() {
        let json = Self.person.jsonValue(options: .openAIStrict)
        #expect(json["properties"]?["age"]?["minimum"] == nil)
        #expect(json["additionalProperties"]?.boolValue == false)
    }

    // MARK: - Strict mode

    @Test("Strictifying makes optional properties nullable and required")
    func strictifyingMakesOptionalsNullable() {
        let json = Self.person.strictified().jsonValue()

        // Every property is now required, including the previously optional `nickname`.
        #expect(json["required"] == .array(["addresses", "age", "name", "nickname"]))
        // `nickname` was already nullable and is not double-wrapped.
        #expect(json["properties"]?["nickname"]?["type"] == .array(["string", "null"]))
        #expect(json["additionalProperties"]?.boolValue == false)
    }

    @Test("Strictifying recurses into nested objects")
    func strictifyingRecurses() {
        let nested = JSONSchema.object(
            properties: [
                "inner": .object(properties: ["a": .string(), "b": .integer()], required: ["a"])
            ],
            required: ["inner"]
        )
        let json = nested.strictified().jsonValue()
        #expect(json["properties"]?["inner"]?["required"] == .array(["a", "b"]))
        #expect(json["properties"]?["inner"]?["properties"]?["b"]?["type"] == .array(["integer", "null"]))
    }

    @Test("Strictifying is idempotent")
    func strictifyingIsIdempotent() {
        let once = Self.person.strictified()
        #expect(once.strictified() == once)
    }

    // MARK: - Transformation

    @Test("Transform visits children before their parent")
    func transformIsBottomUp() {
        var visited: [String] = []
        _ = Self.person.transformed { node in
            switch node {
            case .object: visited.append("object")
            case .array: visited.append("array")
            case .string: visited.append("string")
            default: break
            }
            return node
        }
        // The innermost string is seen before the array that contains it, which is seen before
        // the root object.
        #expect(visited.last == "object")
        #expect(visited.firstIndex(of: "string")! < visited.lastIndex(of: "array")!)
    }

    @Test("Transform rewrites nested nodes")
    func transformRewritesNestedNodes() {
        let stripped = Self.person.transformed { node in
            guard case .string(var constraints) = node else { return node }
            constraints.minLength = nil
            return .string(constraints)
        }
        let json = stripped.jsonValue()
        #expect(json["properties"]?["name"]?["minLength"] == nil)
    }

    @Test("Describing replaces the description without altering structure")
    func describingReplacesDescription() {
        let described = JSONSchema.integer(minimum: 1).describing("A count.")
        #expect(described.schemaDescription == "A count.")
        #expect(described.jsonValue()["minimum"]?.numberValue == 1)
    }

    @Test("Extensions override generated keywords")
    func extensionsOverrideGeneratedKeywords() {
        var schema = JSONSchema.string()
        schema.metadata.extensions = ["type": .string("integer"), "x-custom": .bool(true)]

        let json = schema.jsonValue()
        #expect(json["type"]?.stringValue == "integer")
        #expect(json["x-custom"]?.boolValue == true)
    }
}

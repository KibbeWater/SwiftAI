import Foundation
import Testing

@testable import SwiftAI

// MARK: - Fixtures

@Structured("A single ingredient in a recipe.")
struct Ingredient {
    @Guidance("The ingredient name, singular and lowercase.")
    var name: String

    @Guidance("How much of it, in the unit given.", .range(0.1...1000))
    var quantity: Double

    @Guidance("The unit of measurement.", .anyOf(["g", "ml", "piece"]))
    var unit: String
}

@Structured("How difficult the recipe is.")
enum Difficulty: String {
    case easy
    case moderate
    case hard
}

/// Exercises case-name-based enumerations, which have no raw type.
@Structured
enum Course {
    case starter
    case main
    case dessert
}

@Structured("A recipe with its ingredients.")
struct Recipe {
    @Guidance("The name of the dish.")
    var name: String

    @Guidance("How many people it serves.", .range(1...12))
    var servings: Int

    var ingredients: [Ingredient]
    var difficulty: Difficulty

    @Guidance("Anything worth knowing before starting.")
    var notes: String?

    /// A computed property must not appear in the schema.
    var summary: String { "\(name) for \(servings)" }

    /// A constant with a value is already determined, so it must not appear either.
    let schemaVersion: Int = 1

    /// Static members describe the type, not a value.
    static var defaultServings: Int { 4 }
}

/// Exercises deep nesting, dictionaries, and the less common scalar types.
@Structured
struct Inventory {
    var identifier: UUID
    var updatedAt: Date
    var source: URL
    var countsByAisle: [String: Int]
    var shelves: [[Ingredient]]
    var extra: JSONValue
}

// MARK: - Tests

@Suite("Structured macro")
struct StructuredMacroBehaviorTests {
    // MARK: Schema generation

    @Test("Generates an object schema from stored properties")
    func generatesObjectSchema() {
        let json = Recipe.jsonSchema.jsonValue()

        #expect(json["type"]?.stringValue == "object")
        #expect(json["description"]?.stringValue == "A recipe with its ingredients.")
        #expect(json["properties"]?["name"]?["type"]?.stringValue == "string")
        #expect(json["properties"]?["name"]?["description"]?.stringValue == "The name of the dish.")
        #expect(json["properties"]?["servings"]?["type"]?.stringValue == "integer")
    }

    @Test("Excludes computed, constant, and static members")
    func excludesNonStoredMembers() {
        let properties = try! #require(Recipe.jsonSchema.jsonValue()["properties"]?.objectValue)
        #expect(properties.keys.sorted() == ["difficulty", "ingredients", "name", "notes", "servings"])
    }

    @Test("Only non-optional properties are required")
    func optionalPropertiesAreNotRequired() {
        let json = Recipe.jsonSchema.jsonValue()
        #expect(json["required"] == .array(["difficulty", "ingredients", "name", "servings"]))
        // The optional property is still described, just nullable.
        #expect(json["properties"]?["notes"]?["type"] == .array(["string", "null"]))
        #expect(json["properties"]?["notes"]?["description"]?.stringValue == "Anything worth knowing before starting.")
    }

    @Test("Applies numeric constraints from guidance")
    func appliesNumericConstraints() {
        let servings = try! #require(Recipe.jsonSchema.jsonValue()["properties"]?["servings"])
        #expect(servings["minimum"]?.numberValue == 1)
        #expect(servings["maximum"]?.numberValue == 12)
    }

    @Test("A set of allowed values turns a string into an enumeration")
    func allowedValuesBecomeEnumeration() {
        let unit = try! #require(Ingredient.jsonSchema.jsonValue()["properties"]?["unit"])
        #expect(unit["enum"] == .array(["g", "ml", "piece"]))
        // The description survives the rewrite.
        #expect(unit["description"]?.stringValue == "The unit of measurement.")
    }

    @Test("Nests schemas of other structured types")
    func nestsSchemas() {
        let ingredients = try! #require(Recipe.jsonSchema.jsonValue()["properties"]?["ingredients"])
        #expect(ingredients["type"]?.stringValue == "array")
        #expect(ingredients["items"]?["properties"]?["name"]?["type"]?.stringValue == "string")
        #expect(ingredients["items"]?["description"]?.stringValue == "A single ingredient in a recipe.")
    }

    @Test("String-backed enumerations become value lists")
    func stringEnumerationSchema() {
        let json = Difficulty.jsonSchema.jsonValue()
        #expect(json["enum"] == .array(["easy", "moderate", "hard"]))
        #expect(json["description"]?.stringValue == "How difficult the recipe is.")
    }

    @Test("Enumerations without raw values use their case names")
    func caseNameEnumerationSchema() {
        #expect(Course.jsonSchema.jsonValue()["enum"] == .array(["starter", "main", "dessert"]))
    }

    @Test("Maps the scalar types to their JSON forms")
    func mapsScalarTypes() {
        let properties = try! #require(Inventory.jsonSchema.jsonValue()["properties"])
        #expect(properties["identifier"]?["format"]?.stringValue == "uuid")
        #expect(properties["updatedAt"]?["format"]?.stringValue == "date-time")
        #expect(properties["source"]?["format"]?.stringValue == "uri")
        // A dictionary becomes an open object constrained by its value type.
        #expect(properties["countsByAisle"]?["additionalProperties"]?["type"]?.stringValue == "integer")
        // Nested arrays nest their item schemas.
        #expect(properties["shelves"]?["items"]?["items"]?["type"]?.stringValue == "object")
        // A raw JSON value accepts anything.
        #expect(properties["extra"]?.objectValue?.isEmpty == true)
    }

    // MARK: Decoding

    @Test("Decodes a complete value")
    func decodesCompleteValue() throws {
        let json: JSONValue = [
            "name": "Pancakes",
            "servings": 4,
            "difficulty": "easy",
            "ingredients": [
                ["name": "flour", "quantity": 200, "unit": "g"],
                ["name": "milk", "quantity": 300.5, "unit": "ml"],
            ],
            "notes": "Rest the batter.",
        ]

        let recipe = try Recipe(structuredJSON: json)
        #expect(recipe.name == "Pancakes")
        #expect(recipe.servings == 4)
        #expect(recipe.difficulty == .easy)
        #expect(recipe.ingredients.count == 2)
        #expect(recipe.ingredients[1].quantity == 300.5)
        #expect(recipe.notes == "Rest the batter.")
    }

    @Test("Treats an absent optional as nil")
    func absentOptionalDecodesToNil() throws {
        let recipe = try Recipe(
            structuredJSON: [
                "name": "Toast", "servings": 1, "difficulty": "easy", "ingredients": [],
            ]
        )
        #expect(recipe.notes == nil)
    }

    @Test("Treats an explicit null optional as nil")
    func explicitNullDecodesToNil() throws {
        let recipe = try Recipe(
            structuredJSON: [
                "name": "Toast", "servings": 1, "difficulty": "easy", "ingredients": [], "notes": nil,
            ]
        )
        #expect(recipe.notes == nil)
    }

    @Test("Reports a missing required property by name")
    func reportsMissingRequiredProperty() throws {
        var caught: TypeValidationError?
        do {
            _ = try Recipe(structuredJSON: ["name": "Toast", "difficulty": "easy", "ingredients": []])
        } catch let error as TypeValidationError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.message.contains("servings"))
        #expect(error.path == "servings")
    }

    @Test("Reports a nested type mismatch with a full path")
    func reportsNestedPath() throws {
        var caught: TypeValidationError?
        do {
            _ = try Recipe(
                structuredJSON: [
                    "name": "Toast",
                    "servings": 1,
                    "difficulty": "easy",
                    "ingredients": [
                        ["name": "flour", "quantity": 200, "unit": "g"],
                        ["name": "milk", "quantity": "lots", "unit": "ml"],
                    ],
                ]
            )
        } catch let error as TypeValidationError {
            caught = error
        }

        // The path pinpoints the offending element rather than blaming the whole recipe.
        #expect(try #require(caught).path == "ingredients.1.quantity")
    }

    @Test("Accepts a whole-numbered double where an integer is expected")
    func acceptsWholeDoubleAsInteger() throws {
        // Models routinely emit `4.0`; rejecting it would fail a generation over formatting.
        let recipe = try Recipe(
            structuredJSON: ["name": "Toast", "servings": 4.0, "difficulty": "easy", "ingredients": []]
        )
        #expect(recipe.servings == 4)
    }

    @Test("Rejects a fractional double where an integer is expected")
    func rejectsFractionalDoubleAsInteger() {
        #expect(throws: TypeValidationError.self) {
            try Recipe(
                structuredJSON: ["name": "Toast", "servings": 4.5, "difficulty": "easy", "ingredients": []]
            )
        }
    }

    @Test("Rejects an unrecognized enumeration value")
    func rejectsUnknownEnumerationValue() {
        #expect(throws: TypeValidationError.self) {
            try Difficulty(structuredJSON: .string("impossible"))
        }
        #expect(throws: TypeValidationError.self) {
            try Course(structuredJSON: .string("aperitif"))
        }
    }

    @Test("Decodes case-name enumerations")
    func decodesCaseNameEnumeration() throws {
        #expect(try Course(structuredJSON: .string("dessert")) == .dessert)
    }

    @Test("Round-trips the scalar types")
    func roundTripsScalarTypes() throws {
        let json: JSONValue = [
            "identifier": "6BA7B810-9DAD-11D1-80B4-00C04FD430C8",
            "updatedAt": "2026-08-14T09:41:00Z",
            "source": "https://example.com/inventory",
            "countsByAisle": ["produce": 12, "dairy": 3],
            "shelves": [[["name": "flour", "quantity": 1, "unit": "g"]]],
            "extra": ["anything": [1, 2, 3]],
        ]

        let inventory = try Inventory(structuredJSON: json)
        #expect(inventory.identifier.uuidString == "6BA7B810-9DAD-11D1-80B4-00C04FD430C8")
        #expect(inventory.source.absoluteString == "https://example.com/inventory")
        #expect(inventory.countsByAisle["produce"] == 12)
        #expect(inventory.shelves[0][0].name == "flour")
        #expect(inventory.extra["anything"]?[2]?.intValue == 3)
    }

    // MARK: Partial snapshots

    @Test("Builds an empty snapshot from an empty object")
    func emptySnapshot() {
        let partial = Recipe.partial(from: [:])
        #expect(partial.name == nil)
        #expect(partial.servings == nil)
        #expect(partial.ingredients == nil)
        #expect(partial.notes == nil)
    }

    @Test("Fills a snapshot as fields arrive")
    func snapshotFillsIncrementally() {
        let partial = Recipe.partial(from: ["name": "Pancakes", "servings": 4])
        #expect(partial.name == "Pancakes")
        #expect(partial.servings == 4)
        #expect(partial.ingredients == nil)
    }

    @Test("Nested values appear as their own snapshots")
    func nestedSnapshots() {
        let partial = Recipe.partial(
            from: [
                "name": "Pancakes",
                "ingredients": [
                    ["name": "flour", "quantity": 200, "unit": "g"],
                    ["name": "mi"],
                ],
            ]
        )
        let ingredients = try! #require(partial.ingredients)
        #expect(ingredients.count == 2)
        #expect(ingredients[0].quantity == 200)
        // The second ingredient is half-received: its name is there, the rest is not.
        #expect(ingredients[1].name == "mi")
        #expect(ingredients[1].quantity == nil)
    }

    @Test("A snapshot tolerates values of the wrong type without failing")
    func snapshotToleratesWrongTypes() {
        // A number arriving where a string belongs is dropped, not thrown: partial JSON is
        // routinely mid-token, and a snapshot must never fail.
        let partial = Recipe.partial(from: ["name": 42, "servings": 4])
        #expect(partial.name == nil)
        #expect(partial.servings == 4)
    }

    @Test("Every prefix of a serialized value produces a usable snapshot")
    func snapshotsFromEveryPrefix() throws {
        let recipe: JSONValue = [
            "name": "Pancakes",
            "servings": 4,
            "difficulty": "easy",
            "ingredients": [["name": "flour", "quantity": 200, "unit": "g"]],
            "notes": "Rest the batter.",
        ]
        let text = recipe.serialized(sortedKeys: true)

        var lastNonNilName: String?
        for length in 1...text.count {
            let prefix = String(Array(text)[0..<length])
            let json = try JSONValue.parse(prefix, mode: .partial)
            let partial = Recipe.partial(from: json)

            // Once a field has arrived it never disappears again, so a view bound to a snapshot
            // never has to handle content vanishing.
            if let name = partial.name { lastNonNilName = name }
            if lastNonNilName != nil {
                #expect(partial.name != nil, "the name regressed to nil at length \(length)")
            }
        }

        // The complete document yields every field.
        let complete = Recipe.partial(from: try JSONValue.parse(text))
        #expect(complete.name == "Pancakes")
        #expect(complete.servings == 4)
        #expect(complete.notes == "Rest the batter.")
        #expect(complete.ingredients?.count == 1)
    }

    // MARK: Conformances

    @Test("Structures conform to StructuredOutput and enumerations to StructuredValue")
    func conformances() {
        #expect(Recipe.self is any StructuredOutput.Type)
        #expect(Ingredient.self is any StructuredOutput.Type)
        #expect(Difficulty.self is any StructuredValue.Type)
        // A partial snapshot of an enumeration would be meaningless, so it is not an output type.
        #expect((Difficulty.self as Any) is any StructuredOutput.Type == false)
    }

    @Test("Strictifying a generated schema satisfies provider strict modes")
    func strictifiedSchema() {
        let json = Recipe.jsonSchema.strictified().jsonValue(options: .openAIStrict)

        // Strict mode requires every property to be listed as required.
        #expect(json["required"] == .array(["difficulty", "ingredients", "name", "notes", "servings"]))
        #expect(json["additionalProperties"]?.boolValue == false)
        #expect(json["properties"]?["ingredients"]?["items"]?["additionalProperties"]?.boolValue == false)
    }
}

/// Generates the schema, decoding, and streaming-snapshot support for a type.
///
/// Applied to a struct, the macro writes three things:
///
/// - `static var jsonSchema` — the JSON Schema sent to the model, built from the stored
///   properties and any ``Guidance(_:_:)`` annotations.
/// - `init(structuredJSON:)` — strict decoding, with errors that name the offending property.
/// - A peer `…Partial` type in which every property is optional, plus `partial(from:)` to build
///   one. This is what ``streamObject(model:of:system:prompt:settings:onFinish:)`` emits as the
///   object arrives.
///
/// ```swift
/// @Structured("A recipe with its ingredients.")
/// struct Recipe {
///     @Guidance("The name of the dish.")
///     var name: String
///
///     @Guidance("How many people it serves.", .range(1...12))
///     var servings: Int
///
///     var ingredients: [Ingredient]
///     var notes: String?          // Optional, so absent from `required`.
/// }
/// ```
///
/// Applied to an enumeration whose cases have no associated values, it generates a schema listing
/// the permitted values. String-backed enumerations use their raw values; others use their case
/// names.
///
/// ```swift
/// @Structured("How urgent the task is.")
/// enum Priority: String {
///     case low, medium, high
/// }
/// ```
///
/// ## Property types
///
/// Any type conforming to ``StructuredValue`` may be used. That covers `String`, `Bool`, the
/// integer and floating-point types, `Date`, `URL`, `UUID`, ``AIProviderSpec/JSONValue``,
/// optionals, arrays, and `[String: Value]` dictionaries, plus any other `@Structured` type.
///
/// A property is required exactly when its type is not optional.
///
/// ## Limitations
///
/// Stored properties must carry an explicit type annotation, since a macro sees only syntax and
/// cannot infer types. Generic types, classes, actors, and enumerations with associated values
/// are not supported and produce a diagnostic. A `let` property with an initial value is treated
/// as a constant and left out of the schema entirely.
///
/// - Parameter description: What the type represents. Models rely on this heavily, so it is
///   worth writing even though it is optional.
@attached(
    member,
    names: named(jsonSchema), named(init(structuredJSON:)), named(partial(from:)), named(Partial)
)
@attached(extension, conformances: StructuredOutput, StructuredValue)
@attached(peer, names: suffixed(Partial))
public macro Structured(_ description: String? = nil) =
    #externalMacro(module: "SwiftAIMacrosImpl", type: "StructuredMacro")

/// Describes and constrains a property of a ``Structured(_:)`` type.
///
/// The description is the single most effective way to improve what a model puts in a field —
/// considerably more so than any structural constraint. Write it as an instruction to the model
/// rather than as documentation for a reader.
///
/// ```swift
/// @Guidance("The city to look up, as the user wrote it.")
/// var city: String
///
/// @Guidance("Forecast length in days.", .range(1...7))
/// var days: Int
///
/// @Guidance("An ISO 4217 currency code.", .pattern("^[A-Z]{3}$"))
/// var currency: String
/// ```
///
/// Constraints that do not apply to the property's type are ignored rather than rejected.
///
/// - Parameters:
///   - description: What the property means, addressed to the model.
///   - constraints: Restrictions to include in the schema.
@attached(peer)
public macro Guidance(_ description: String? = nil, _ constraints: SchemaConstraint...) =
    #externalMacro(module: "SwiftAIMacrosImpl", type: "GuidanceMacro")

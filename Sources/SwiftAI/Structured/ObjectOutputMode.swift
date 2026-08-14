import AIProviderSpec

/// How a structured generation asks for its result, and how it reads the answer back.
///
/// Only object mode maps directly onto what providers accept. A bare array or a bare string is
/// not a valid top-level response for most structured output implementations, so those modes wrap
/// the value in a single-property object on the way out and unwrap it on the way in. The wrapper
/// never reaches the caller.
enum ObjectOutputMode: Sendable {
    /// A single object matching a schema.
    case object(JSONSchema)

    /// An array of values matching an element schema.
    case array(element: JSONSchema)

    /// One of a fixed set of strings.
    case enumeration([String])

    /// Any JSON, with no schema.
    case free

    /// The property name used to wrap an array. Chosen to be unlikely to collide with anything
    /// meaningful if it ever leaks into a prompt.
    private static let arrayKey = "elements"

    /// The property name used to wrap an enumeration result.
    private static let enumerationKey = "result"

    /// The schema sent to the provider.
    var requestSchema: JSONSchema? {
        switch self {
        case .object(let schema):
            return schema

        case .array(let element):
            return .object(
                properties: [Self.arrayKey: .array(of: element)],
                required: [Self.arrayKey],
                description: "A container for the generated list."
            )

        case .enumeration(let values):
            return .object(
                properties: [Self.enumerationKey: .enumeration(values)],
                required: [Self.enumerationKey],
                description: "A container for the chosen value."
            )

        case .free:
            return nil
        }
    }

    /// A name for the schema, which some providers require.
    var schemaName: String? {
        switch self {
        case .object: return "response"
        case .array: return "list"
        case .enumeration: return "choice"
        case .free: return nil
        }
    }

    /// Extracts the caller's value from a decoded response.
    ///
    /// - Parameter json: The full response, wrapper included.
    /// - Returns: The unwrapped value, or `nil` when the wrapper is not present yet — which is
    ///   normal while a response is still streaming.
    func unwrap(_ json: JSONValue) -> JSONValue? {
        switch self {
        case .object, .free:
            return json
        case .array:
            return json[Self.arrayKey]
        case .enumeration:
            return json[Self.enumerationKey]
        }
    }
}

import SwiftDiagnostics
import SwiftSyntax

/// A diagnostic emitted by the SwiftAI macros.
///
/// Macro diagnostics are the only feedback a user gets when a declaration cannot be handled, so
/// each one states what is wrong, why, and what to write instead.
struct MacroDiagnostic: DiagnosticMessage {
    var message: String
    var diagnosticID: MessageID
    var severity: DiagnosticSeverity

    init(_ message: String, id: String, severity: DiagnosticSeverity = .error) {
        self.message = message
        self.diagnosticID = MessageID(domain: "SwiftAIMacros", id: id)
        self.severity = severity
    }

    static let notAStructOrEnum = MacroDiagnostic(
        """
        '@Structured' can only be applied to a struct or an enumeration. \
        Classes and actors are reference types, and a model-generated value has no identity to \
        share; declare a struct instead.
        """,
        id: "notAStructOrEnum"
    )

    static let genericTypeUnsupported = MacroDiagnostic(
        """
        '@Structured' cannot be applied to a generic type, because a schema has to be a single \
        concrete shape. Declare a concrete type for each specialization you need.
        """,
        id: "genericTypeUnsupported"
    )

    static func missingTypeAnnotation(_ name: String) -> MacroDiagnostic {
        MacroDiagnostic(
            """
            The property '\(name)' needs an explicit type annotation. A macro sees only syntax, \
            so it cannot infer the type the way the compiler does. Write 'var \(name): T'.
            """,
            id: "missingTypeAnnotation"
        )
    }

    static let enumerationWithAssociatedValues = MacroDiagnostic(
        """
        '@Structured' supports only enumerations whose cases have no associated values, since \
        each case has to map to a single JSON string. Model a case that carries data as a struct \
        with an optional payload.
        """,
        id: "enumerationWithAssociatedValues"
    )

    static func unsupportedRawValueType(_ type: String) -> MacroDiagnostic {
        MacroDiagnostic(
            """
            '@Structured' supports 'String'-backed enumerations only; '\(type)' cannot be \
            represented as a JSON enumeration. Remove the raw type to use the case names, or \
            change it to 'String'.
            """,
            id: "unsupportedRawValueType"
        )
    }

    static let emptyStructure = MacroDiagnostic(
        """
        '@Structured' needs at least one stored property. An object schema with no properties \
        gives the model nothing to fill in.
        """,
        id: "emptyStructure"
    )

    static let emptyEnumeration = MacroDiagnostic(
        "'@Structured' needs at least one case to build an enumeration schema.",
        id: "emptyEnumeration"
    )

    static let guidanceOnNonProperty = MacroDiagnostic(
        """
        '@Guidance' applies to a stored property of a '@Structured' type. It has no effect here.
        """,
        id: "guidanceOnNonProperty",
        severity: .warning
    )
}

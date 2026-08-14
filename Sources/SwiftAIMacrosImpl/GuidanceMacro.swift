import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

/// Implements `@Guidance`.
///
/// The macro generates nothing. It exists so that a description and constraints can be written
/// next to the property they describe, where they stay accurate as the type changes; `@Structured`
/// reads the attribute off the property and folds it into the schema.
///
/// Attaching it to anything other than a stored property is a warning rather than an error,
/// because the annotation is inert either way and failing the build would be disproportionate.
public struct GuidanceMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        guard declaration.is(VariableDeclSyntax.self) else {
            context.diagnose(Diagnostic(node: node, message: MacroDiagnostic.guidanceOnNonProperty))
            return []
        }
        return []
    }
}

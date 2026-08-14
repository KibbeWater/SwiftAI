import SwiftCompilerPlugin
import SwiftSyntaxMacros

/// The compiler plugin exposing SwiftAI's macros.
@main
struct SwiftAIMacrosPlugin: CompilerPlugin {
    let providingMacros: [any Macro.Type] = [
        StructuredMacro.self,
        GuidanceMacro.self,
    ]
}

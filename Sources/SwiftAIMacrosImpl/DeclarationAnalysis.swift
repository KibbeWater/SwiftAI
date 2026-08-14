import SwiftParser
import SwiftSyntax

/// A stored property that will become a schema property.
struct StoredProperty {
    /// The Swift property name, used verbatim as the JSON key.
    var name: String

    /// The declared type, copied into generated code rather than interpreted.
    ///
    /// Deferring type interpretation to the compiler is what keeps this macro small: it never has
    /// to know that `[Ingredient]` means an array, only that `PartialOf<[Ingredient]>` is the
    /// right thing to write.
    var type: TypeSyntax

    /// The description expression from ``Guidance``, if annotated.
    var description: ExprSyntax?

    /// The constraint expressions from ``Guidance``.
    var constraints: [ExprSyntax]

    /// The syntax node to attach diagnostics to.
    var anchor: Syntax
}

/// An enumeration case that will become a permitted schema value.
struct EnumerationCase {
    /// The Swift case name.
    var name: String

    /// The string the model must produce: the raw value when one is declared, otherwise the case
    /// name.
    var jsonValue: String
}

enum DeclarationAnalysis {
    // MARK: - Access level

    /// The access modifier to copy onto generated declarations.
    ///
    /// Generated members must not be less visible than the type itself, or a public type's
    /// conformance would be unusable from another module.
    static func accessLevel(of declaration: some DeclGroupSyntax) -> String {
        for modifier in declaration.modifiers {
            switch modifier.name.tokenKind {
            case .keyword(.public), .keyword(.open):
                return "public "
            case .keyword(.package):
                return "package "
            case .keyword(.fileprivate):
                return "fileprivate "
            case .keyword(.private):
                // A private type's peers live at file scope, where `private` means `fileprivate`
                // anyway; spelling it that way avoids a confusing redeclaration.
                return "fileprivate "
            default:
                continue
            }
        }
        return ""
    }

    // MARK: - Properties

    /// Collects the stored properties that should appear in the schema.
    ///
    /// Skipped, deliberately:
    /// - `static` and `class` members, which describe the type rather than a value;
    /// - computed properties, which have nothing to decode into;
    /// - `let` properties with an initial value, which are constants and cannot be assigned in a
    ///   generated initializer.
    static func storedProperties(
        of declaration: some DeclGroupSyntax,
        onMissingType: (StoredPropertyProblem) -> Void
    ) -> [StoredProperty] {
        var properties: [StoredProperty] = []

        for member in declaration.memberBlock.members {
            guard let variable = member.decl.as(VariableDeclSyntax.self) else { continue }

            let isStatic = variable.modifiers.contains { modifier in
                modifier.name.tokenKind == .keyword(.static) || modifier.name.tokenKind == .keyword(.class)
            }
            guard !isStatic else { continue }

            let isConstant = variable.bindingSpecifier.tokenKind == .keyword(.let)
            let guidance = guidanceArguments(from: variable.attributes)

            for binding in variable.bindings {
                guard let pattern = binding.pattern.as(IdentifierPatternSyntax.self) else { continue }
                guard !isComputed(binding) else { continue }
                // A constant with an initial value is already determined; the model has no say.
                if isConstant, binding.initializer != nil { continue }

                let name = pattern.identifier.text.trimmingBackticks
                guard let type = binding.typeAnnotation?.type else {
                    onMissingType(StoredPropertyProblem(name: name, anchor: Syntax(binding)))
                    continue
                }

                properties.append(
                    StoredProperty(
                        name: name,
                        type: type.trimmed,
                        description: guidance.description,
                        constraints: guidance.constraints,
                        anchor: Syntax(binding)
                    )
                )
            }
        }
        return properties
    }

    /// A property that cannot be handled, reported back to the macro for diagnostics.
    struct StoredPropertyProblem {
        var name: String
        var anchor: Syntax
    }

    /// Whether a binding is computed rather than stored.
    ///
    /// `willSet` and `didSet` observers do not make a property computed, so they are allowed
    /// through.
    private static func isComputed(_ binding: PatternBindingSyntax) -> Bool {
        guard let accessorBlock = binding.accessorBlock else { return false }
        switch accessorBlock.accessors {
        case .getter:
            return true
        case .accessors(let accessors):
            return accessors.contains { accessor in
                switch accessor.accessorSpecifier.tokenKind {
                case .keyword(.get), .keyword(._read), .keyword(.unsafeAddress): return true
                default: return false
                }
            }
        }
    }

    // MARK: - Guidance

    /// Reads the arguments of a `@Guidance` attribute, if one is present.
    ///
    /// The expressions are copied verbatim into generated code rather than being interpreted, so
    /// any expression the compiler accepts works — including references to constants and computed
    /// values.
    static func guidanceArguments(
        from attributes: AttributeListSyntax
    ) -> (description: ExprSyntax?, constraints: [ExprSyntax]) {
        for attribute in attributes {
            guard case .attribute(let attribute) = attribute,
                  attribute.attributeName.trimmedDescription == "Guidance"
            else { continue }
            guard case .argumentList(let arguments)? = attribute.arguments else { return (nil, []) }

            var expressions = Array(arguments.map(\.expression))
            var description: ExprSyntax?
            // The description is optional and comes first. It is present when the leading
            // argument is a string literal, or an explicit `nil` placeholder.
            if let first = expressions.first {
                if first.is(StringLiteralExprSyntax.self) {
                    description = first
                    expressions.removeFirst()
                } else if first.is(NilLiteralExprSyntax.self) {
                    expressions.removeFirst()
                }
            }
            return (description, expressions)
        }
        return (nil, [])
    }

    // MARK: - Enumerations

    /// Whether an enumeration declares `String` as its raw type.
    ///
    /// Syntax alone cannot distinguish a raw type from a protocol, so the leading inherited type
    /// is used, which is where Swift requires the raw type to appear.
    static func rawValueType(of declaration: EnumDeclSyntax) -> String? {
        guard let first = declaration.inheritanceClause?.inheritedTypes.first else { return nil }
        let name = first.type.trimmedDescription
        let knownRawValueTypes: Set<String> = [
            "String", "Int", "Int8", "Int16", "Int32", "Int64",
            "UInt", "UInt8", "UInt16", "UInt32", "UInt64", "Double", "Float", "Character",
        ]
        return knownRawValueTypes.contains(name) ? name : nil
    }

    /// Collects the cases of an enumeration.
    ///
    /// - Returns: The cases, or `nil` if any case carries associated values.
    static func enumerationCases(of declaration: EnumDeclSyntax) -> [EnumerationCase]? {
        var cases: [EnumerationCase] = []

        for member in declaration.memberBlock.members {
            guard let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) else { continue }
            for element in caseDecl.elements {
                guard element.parameterClause == nil else { return nil }
                let name = element.name.text.trimmingBackticks

                // An explicit raw value wins; otherwise the case name is what the model produces.
                if let rawValue = element.rawValue?.value.as(StringLiteralExprSyntax.self),
                   let text = rawValue.representedLiteralValue {
                    cases.append(EnumerationCase(name: name, jsonValue: text))
                } else {
                    cases.append(EnumerationCase(name: name, jsonValue: name))
                }
            }
        }
        return cases
    }
}

extension String {
    /// Strips the backticks from an escaped identifier such as `` `default` ``.
    ///
    /// The JSON key should be `default`, while the Swift reference still needs its backticks.
    var trimmingBackticks: String {
        hasPrefix("`") && hasSuffix("`") ? String(dropFirst().dropLast()) : self
    }

    /// The form usable as a Swift identifier, re-adding backticks for reserved words.
    var escapedIdentifier: String {
        let reserved: Set<String> = [
            "associatedtype", "class", "deinit", "enum", "extension", "fileprivate", "func",
            "import", "init", "inout", "internal", "let", "operator", "private", "protocol",
            "public", "rethrows", "static", "struct", "subscript", "typealias", "var", "where",
            "break", "case", "catch", "continue", "default", "defer", "do", "else", "fallthrough",
            "for", "guard", "if", "in", "repeat", "return", "switch", "throw", "try", "while",
            "as", "false", "is", "nil", "self", "Self", "super", "throws", "true", "Any",
        ]
        return reserved.contains(self) ? "`\(self)`" : self
    }
}

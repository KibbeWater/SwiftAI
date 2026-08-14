import Foundation

extension JSONValue {
    /// How the parser should react to input that ends before the JSON document is complete.
    public enum ParsingMode: Sendable, Hashable {
        /// Truncated input is an error. Use this for complete documents such as HTTP response
        /// bodies.
        case strict

        /// Truncated input yields the largest well-formed value that can be recovered.
        ///
        /// This is what makes streaming structured output possible: model output arrives a few
        /// characters at a time, and every prefix of the eventual JSON document can be turned
        /// into a usable snapshot. Recovery rules:
        ///
        /// - An unterminated string yields the characters decoded so far.
        /// - An object drops any trailing key whose value has not started, and any pair whose
        ///   value could not be recovered.
        /// - An array drops a trailing element that could not be recovered.
        /// - An incomplete literal (`tru`, `nul`) is dropped.
        /// - An incomplete number (`-`, `1.`, `1e`) is dropped, while a syntactically valid
        ///   number at the end of input (`12`) is kept, since the next chunk may only extend it.
        case partial
    }

    /// Parses a JSON document from UTF-8 encoded data.
    ///
    /// - Parameters:
    ///   - data: The UTF-8 bytes to parse.
    ///   - mode: Whether truncated input is an error or should be recovered. Defaults to ``ParsingMode/strict``.
    ///   - maximumDepth: The deepest nesting the parser will accept. Guards against stack
    ///     exhaustion from hostile or malformed input.
    /// - Returns: The parsed value.
    /// - Throws: ``JSONParseError`` if the input is not valid JSON.
    public static func parse(
        _ data: Data,
        mode: ParsingMode = .strict,
        maximumDepth: Int = 512
    ) throws -> JSONValue {
        var parser = JSONParser(bytes: Array(data), mode: mode, maximumDepth: maximumDepth)
        return try parser.parseDocument()
    }

    /// Parses a JSON document from a string.
    ///
    /// - Parameters:
    ///   - string: The text to parse.
    ///   - mode: Whether truncated input is an error or should be recovered. Defaults to ``ParsingMode/strict``.
    ///   - maximumDepth: The deepest nesting the parser will accept.
    /// - Returns: The parsed value.
    /// - Throws: ``JSONParseError`` if the input is not valid JSON.
    public static func parse(
        _ string: String,
        mode: ParsingMode = .strict,
        maximumDepth: Int = 512
    ) throws -> JSONValue {
        var parser = JSONParser(bytes: Array(string.utf8), mode: mode, maximumDepth: maximumDepth)
        return try parser.parseDocument()
    }

    /// Serializes the value to a JSON string.
    ///
    /// - Parameters:
    ///   - sortedKeys: Whether object keys are emitted in sorted order. Sorting makes output
    ///     stable, which matters for tests and for request bodies that get cached upstream.
    ///     Defaults to `false`.
    ///   - prettyPrinted: Whether to emit indented, multi-line output. Defaults to `false`.
    /// - Returns: A JSON string.
    ///
    /// - Note: Non-finite doubles (`infinity`, `nan`) have no JSON representation and are emitted
    ///   as `null`, matching the behaviour of JavaScript's `JSON.stringify`.
    public func serialized(sortedKeys: Bool = false, prettyPrinted: Bool = false) -> String {
        var output = ""
        JSONSerializer.write(self, into: &output, sortedKeys: sortedKeys, prettyPrinted: prettyPrinted, depth: 0)
        return output
    }

    /// Serializes the value to UTF-8 encoded JSON data, ready to use as an HTTP request body.
    ///
    /// - Parameters:
    ///   - sortedKeys: Whether object keys are emitted in sorted order.
    ///   - prettyPrinted: Whether to emit indented, multi-line output.
    /// - Returns: UTF-8 encoded JSON.
    public func serializedData(sortedKeys: Bool = false, prettyPrinted: Bool = false) -> Data {
        Data(serialized(sortedKeys: sortedKeys, prettyPrinted: prettyPrinted).utf8)
    }
}

// MARK: - Serializer

private enum JSONSerializer {
    static func write(
        _ value: JSONValue,
        into output: inout String,
        sortedKeys: Bool,
        prettyPrinted: Bool,
        depth: Int
    ) {
        switch value {
        case .null:
            output += "null"

        case .bool(let bool):
            output += bool ? "true" : "false"

        case .int(let int):
            output += String(int)

        case .double(let double):
            // JSON has no representation for infinity or NaN.
            output += double.isFinite ? shortestRepresentation(of: double) : "null"

        case .string(let string):
            writeString(string, into: &output)

        case .array(let array):
            guard !array.isEmpty else {
                output += "[]"
                return
            }
            output += "["
            for (offset, element) in array.enumerated() {
                if offset > 0 { output += "," }
                writeNewlineAndIndent(into: &output, prettyPrinted: prettyPrinted, depth: depth + 1)
                write(element, into: &output, sortedKeys: sortedKeys, prettyPrinted: prettyPrinted, depth: depth + 1)
            }
            writeNewlineAndIndent(into: &output, prettyPrinted: prettyPrinted, depth: depth)
            output += "]"

        case .object(let object):
            guard !object.isEmpty else {
                output += "{}"
                return
            }
            output += "{"
            let pairs = sortedKeys ? object.sorted { $0.key < $1.key } : Array(object)
            for (offset, pair) in pairs.enumerated() {
                if offset > 0 { output += "," }
                writeNewlineAndIndent(into: &output, prettyPrinted: prettyPrinted, depth: depth + 1)
                writeString(pair.key, into: &output)
                output += prettyPrinted ? ": " : ":"
                write(pair.value, into: &output, sortedKeys: sortedKeys, prettyPrinted: prettyPrinted, depth: depth + 1)
            }
            writeNewlineAndIndent(into: &output, prettyPrinted: prettyPrinted, depth: depth)
            output += "}"
        }
    }

    private static func writeNewlineAndIndent(into output: inout String, prettyPrinted: Bool, depth: Int) {
        guard prettyPrinted else { return }
        output += "\n"
        output += String(repeating: "  ", count: depth)
    }

    /// Renders a double using the shortest form that round-trips, but without Swift's habit of
    /// writing whole values as `1.0` when the JSON ecosystem expects `1.0` anyway — this keeps
    /// `Double(text) == value` true for every emitted value.
    private static func shortestRepresentation(of value: Double) -> String {
        let description = String(value)
        // Swift renders large or small magnitudes in exponential form (`1e+100`), which is valid
        // JSON, so the description can be used directly.
        return description
    }

    private static func writeString(_ string: String, into output: inout String) {
        output += "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case UnicodeScalar(0x08): output += "\\b"
            case UnicodeScalar(0x0C): output += "\\f"
            case let scalar where scalar.value < 0x20:
                output += "\\u" + String(format: "%04x", scalar.value)
            default:
                output.unicodeScalars.append(scalar)
            }
        }
        output += "\""
    }
}

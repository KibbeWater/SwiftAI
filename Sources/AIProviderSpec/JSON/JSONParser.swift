/// A recursive-descent JSON parser operating on UTF-8 bytes.
///
/// A hand-written parser is used in preference to `JSONSerialization` for three reasons: it
/// behaves identically on Darwin and Linux, it preserves the integer/double distinction exactly,
/// and it can recover values from truncated input, which the streaming object APIs depend on.
///
/// ## Truncation handling
///
/// In ``JSONValue/ParsingMode/partial`` mode, discovering that the input ends mid-value sets
/// ``isTruncated`` and advances the cursor to the end of the input. Both matter: the flag lets
/// every enclosing container unwind immediately and return what it has, and moving the cursor
/// stops a parent from resuming at bytes its child deliberately declined to consume. Without the
/// unwind, a parent array would resume after a rewound child and report a spurious syntax error.
struct JSONParser {
    let bytes: [UInt8]
    let mode: JSONValue.ParsingMode
    let maximumDepth: Int
    var index: Int = 0

    /// Whether partial-mode recovery has been triggered. Once set, every container returns as
    /// soon as its current element is complete.
    private(set) var isTruncated = false

    init(bytes: [UInt8], mode: JSONValue.ParsingMode, maximumDepth: Int) {
        self.bytes = bytes
        self.mode = mode
        self.maximumDepth = maximumDepth
    }

    // MARK: Entry point

    mutating func parseDocument() throws -> JSONValue {
        skipWhitespace()
        guard index < bytes.count else {
            throw JSONParseError(message: "Unexpected end of input: the document is empty.", offset: index)
        }
        guard let value = try parseValue(depth: 0) else {
            // Only reachable in partial mode, where the very first value was unrecoverable.
            throw JSONParseError(message: "No value could be recovered from the input.", offset: index)
        }
        if mode == .strict {
            skipWhitespace()
            guard index == bytes.count else {
                throw JSONParseError(
                    message: "Unexpected trailing content after the top-level JSON value.",
                    offset: index
                )
            }
        }
        return value
    }

    // MARK: Scanning primitives

    private mutating func skipWhitespace() {
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x09, 0x0A, 0x0D: index += 1
            default: return
            }
        }
    }

    private var isAtEnd: Bool { index >= bytes.count }

    private func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

    /// Records that the input ended mid-value and consumes the remainder.
    ///
    /// Only ever called in partial mode.
    private mutating func markTruncated() {
        isTruncated = true
        index = bytes.count
    }

    /// Signals end-of-input: throws in strict mode, marks truncation in partial mode.
    private mutating func handleEndOfInput(_ what: String) throws {
        guard mode == .partial else {
            throw JSONParseError(message: "Unexpected end of input while parsing \(what).", offset: index)
        }
        markTruncated()
    }

    // MARK: Values

    /// Parses a single value.
    ///
    /// Returns `nil` only in partial mode, to signal that the value at this position could not be
    /// recovered at all and the enclosing container should drop it. A value that was *partially*
    /// recovered is returned normally, with ``isTruncated`` set.
    private mutating func parseValue(depth: Int) throws -> JSONValue? {
        guard depth <= maximumDepth else {
            throw JSONParseError(
                message: "Nesting exceeds the maximum supported depth of \(maximumDepth).",
                offset: index
            )
        }
        skipWhitespace()
        guard let byte = peek() else {
            try handleEndOfInput("a value")
            return nil
        }
        switch byte {
        case UInt8(ascii: "{"):
            return try parseObject(depth: depth)
        case UInt8(ascii: "["):
            return try parseArray(depth: depth)
        case UInt8(ascii: "\""):
            let scanned = try parseString()
            // An unterminated string still carries usable text, which is what makes a streaming
            // text field readable before it finishes.
            return .string(scanned.text)
        case UInt8(ascii: "t"):
            return try parseLiteral("true", value: .bool(true))
        case UInt8(ascii: "f"):
            return try parseLiteral("false", value: .bool(false))
        case UInt8(ascii: "n"):
            return try parseLiteral("null", value: .null)
        case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
            return try parseNumber()
        default:
            throw JSONParseError(
                message: "Unexpected character '\(Character(UnicodeScalar(byte)))' where a value was expected.",
                offset: index
            )
        }
    }

    private mutating func parseLiteral(_ literal: String, value: JSONValue) throws -> JSONValue? {
        let expected = Array(literal.utf8)
        guard index + expected.count <= bytes.count else {
            let remaining = Array(bytes[index...])
            // A prefix of the literal that runs off the end of the input: `tru`, `nul`.
            if mode == .partial, expected.starts(with: remaining) {
                markTruncated()
                return nil
            }
            throw JSONParseError(message: "Unexpected end of input while parsing '\(literal)'.", offset: index)
        }
        guard Array(bytes[index..<(index + expected.count)]) == expected else {
            throw JSONParseError(message: "Invalid literal; expected '\(literal)'.", offset: index)
        }
        index += expected.count
        return value
    }

    // MARK: Strings

    /// The outcome of scanning a quoted string.
    private struct ScannedString {
        /// The characters decoded so far.
        var text: String
        /// Whether a closing quote was found. `false` means the input ran out first.
        var isTerminated: Bool
    }

    private mutating func parseString() throws -> ScannedString {
        precondition(peek() == UInt8(ascii: "\""))
        index += 1  // Consume the opening quote.

        var scalars = String.UnicodeScalarView()
        var utf8Buffer: [UInt8] = []

        /// Flushes raw UTF-8 bytes accumulated since the last escape sequence.
        func flush() {
            guard !utf8Buffer.isEmpty else { return }
            scalars.append(contentsOf: String(decoding: utf8Buffer, as: UTF8.self).unicodeScalars)
            utf8Buffer.removeAll(keepingCapacity: true)
        }

        while index < bytes.count {
            let byte = bytes[index]
            switch byte {
            case UInt8(ascii: "\""):
                index += 1
                flush()
                return ScannedString(text: String(scalars), isTerminated: true)

            case UInt8(ascii: "\\"):
                flush()
                let escapeStart = index
                index += 1
                guard let escape = peek() else {
                    // Input ends immediately after a backslash.
                    guard mode == .partial else {
                        throw JSONParseError(message: "Unexpected end of input in a string escape.", offset: index)
                    }
                    markTruncated()
                    return ScannedString(text: String(scalars), isTerminated: false)
                }
                index += 1
                switch escape {
                case UInt8(ascii: "\""): scalars.append("\"")
                case UInt8(ascii: "\\"): scalars.append("\\")
                case UInt8(ascii: "/"): scalars.append("/")
                case UInt8(ascii: "b"): scalars.append(UnicodeScalar(0x08))
                case UInt8(ascii: "f"): scalars.append(UnicodeScalar(0x0C))
                case UInt8(ascii: "n"): scalars.append("\n")
                case UInt8(ascii: "r"): scalars.append("\r")
                case UInt8(ascii: "t"): scalars.append("\t")
                case UInt8(ascii: "u"):
                    guard let scalar = try parseUnicodeEscape(escapeStart: escapeStart) else {
                        // Partial mode: the escape ran off the end. Emitting a half-decoded
                        // escape would be worse than omitting the character entirely.
                        return ScannedString(text: String(scalars), isTerminated: false)
                    }
                    scalars.append(scalar)
                default:
                    throw JSONParseError(
                        message: "Invalid escape sequence '\\\(Character(UnicodeScalar(escape)))'.",
                        offset: escapeStart
                    )
                }

            case 0x00...0x1F:
                throw JSONParseError(
                    message: "Unescaped control character U+\(hexCodePoint(UInt32(byte))) in a string.",
                    offset: index
                )

            default:
                utf8Buffer.append(byte)
                index += 1
            }
        }

        // Input ended before the closing quote.
        guard mode == .partial else {
            throw JSONParseError(message: "Unexpected end of input; the string was never closed.", offset: index)
        }
        flush()
        markTruncated()
        return ScannedString(text: String(scalars), isTerminated: false)
    }

    /// Parses the four hex digits of a `\u` escape, combining surrogate pairs.
    ///
    /// - Returns: `nil` in partial mode when the escape is truncated, having marked truncation.
    private mutating func parseUnicodeEscape(escapeStart: Int) throws -> UnicodeScalar? {
        guard let high = try parseHexQuad(escapeStart: escapeStart) else { return nil }

        // A high surrogate must be followed by `\uDC00`–`\uDFFF` to form a scalar.
        if (0xD800...0xDBFF).contains(high) {
            let afterHigh = index
            guard index + 1 < bytes.count,
                  bytes[index] == UInt8(ascii: "\\"),
                  bytes[index + 1] == UInt8(ascii: "u")
            else {
                // Running out of input here is truncation; anything else is malformed.
                if mode == .partial, index >= bytes.count - 1 {
                    markTruncated()
                    return nil
                }
                throw JSONParseError(
                    message: "A high surrogate must be followed by a low surrogate escape.",
                    offset: afterHigh
                )
            }
            index += 2
            guard let low = try parseHexQuad(escapeStart: escapeStart) else { return nil }
            guard (0xDC00...0xDFFF).contains(low) else {
                throw JSONParseError(
                    message: "Invalid low surrogate U+\(hexCodePoint(low)).",
                    offset: afterHigh
                )
            }
            let combined = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
            guard let scalar = UnicodeScalar(combined) else {
                throw JSONParseError(message: "Invalid surrogate pair.", offset: escapeStart)
            }
            return scalar
        }

        guard !(0xDC00...0xDFFF).contains(high), let scalar = UnicodeScalar(high) else {
            throw JSONParseError(
                message: "Invalid unpaired low surrogate U+\(hexCodePoint(high)).",
                offset: escapeStart
            )
        }
        return scalar
    }

    private mutating func parseHexQuad(escapeStart: Int) throws -> UInt32? {
        guard index + 4 <= bytes.count else {
            guard mode == .partial else {
                throw JSONParseError(message: "Unexpected end of input in a '\\u' escape.", offset: index)
            }
            markTruncated()
            return nil
        }
        var value: UInt32 = 0
        for offset in 0..<4 {
            let byte = bytes[index + offset]
            let digit: UInt32
            switch byte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = UInt32(byte - UInt8(ascii: "0"))
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = UInt32(byte - UInt8(ascii: "a")) + 10
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = UInt32(byte - UInt8(ascii: "A")) + 10
            default:
                throw JSONParseError(
                    message: "Invalid hexadecimal digit in a '\\u' escape.",
                    offset: index + offset
                )
            }
            value = value << 4 | digit
        }
        index += 4
        return value
    }

    // MARK: Numbers

    private mutating func parseNumber() throws -> JSONValue? {
        let start = index
        var isInteger = true

        if peek() == UInt8(ascii: "-") { index += 1 }

        // Integer part.
        let integerStart = index
        while let byte = peek(), (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) { index += 1 }
        guard index > integerStart else {
            // A lone `-` at the end of the input: the digits are still in flight.
            if mode == .partial, isAtEnd {
                markTruncated()
                return nil
            }
            throw JSONParseError(message: "A number must have at least one digit.", offset: index)
        }
        // Leading zeros are not permitted by RFC 8259.
        if bytes[integerStart] == UInt8(ascii: "0"), index - integerStart > 1 {
            throw JSONParseError(message: "Numbers may not have leading zeros.", offset: integerStart)
        }

        // Fractional part.
        if peek() == UInt8(ascii: ".") {
            isInteger = false
            index += 1
            let fractionStart = index
            while let byte = peek(), (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) { index += 1 }
            guard index > fractionStart else {
                // Trailing `1.` — wait for the fractional digits rather than guessing.
                if mode == .partial, isAtEnd {
                    markTruncated()
                    return nil
                }
                throw JSONParseError(message: "A fractional part must have at least one digit.", offset: index)
            }
        }

        // Exponent part.
        if let byte = peek(), byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") {
            isInteger = false
            index += 1
            if let sign = peek(), sign == UInt8(ascii: "+") || sign == UInt8(ascii: "-") { index += 1 }
            let exponentStart = index
            while let byte = peek(), (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) { index += 1 }
            guard index > exponentStart else {
                // Trailing `1e` or `1e-`.
                if mode == .partial, isAtEnd {
                    markTruncated()
                    return nil
                }
                throw JSONParseError(message: "An exponent must have at least one digit.", offset: index)
            }
        }

        let text = String(decoding: bytes[start..<index], as: UTF8.self)
        if isInteger, let value = Int(text) {
            return .int(value)
        }
        guard let value = Double(text) else {
            throw JSONParseError(message: "'\(text)' is not a representable number.", offset: start)
        }
        return .double(value)
    }

    // MARK: Containers

    private mutating func parseObject(depth: Int) throws -> JSONValue? {
        precondition(peek() == UInt8(ascii: "{"))
        index += 1

        var object: [String: JSONValue] = [:]
        skipWhitespace()

        if peek() == UInt8(ascii: "}") {
            index += 1
            return .object(object)
        }

        while true {
            skipWhitespace()
            guard let byte = peek() else {
                try handleEndOfInput("an object")
                return .object(object)
            }
            guard byte == UInt8(ascii: "\"") else {
                throw JSONParseError(message: "Object keys must be strings.", offset: index)
            }

            // A truncated key contributes nothing: its value has not started arriving yet.
            let key = try parseString()
            guard key.isTerminated else { return .object(object) }

            skipWhitespace()
            guard peek() == UInt8(ascii: ":") else {
                if mode == .partial, isAtEnd {
                    markTruncated()
                    return .object(object)
                }
                throw JSONParseError(message: "Expected ':' after an object key.", offset: index)
            }
            index += 1

            guard let value = try parseValue(depth: depth + 1) else {
                // Partial mode: the value could not be recovered, so the pair is dropped.
                return .object(object)
            }
            object[key.text] = value
            // The value was recovered but the input ended inside it; stop before the parent
            // resumes reading bytes that are not there.
            if isTruncated { return .object(object) }

            skipWhitespace()
            switch peek() {
            case UInt8(ascii: ","):
                index += 1
            case UInt8(ascii: "}"):
                index += 1
                return .object(object)
            case nil:
                try handleEndOfInput("an object")
                return .object(object)
            default:
                throw JSONParseError(message: "Expected ',' or '}' in an object.", offset: index)
            }
        }
    }

    private mutating func parseArray(depth: Int) throws -> JSONValue? {
        precondition(peek() == UInt8(ascii: "["))
        index += 1

        var array: [JSONValue] = []
        skipWhitespace()

        if peek() == UInt8(ascii: "]") {
            index += 1
            return .array(array)
        }

        while true {
            guard let value = try parseValue(depth: depth + 1) else {
                return .array(array)
            }
            array.append(value)
            if isTruncated { return .array(array) }

            skipWhitespace()
            switch peek() {
            case UInt8(ascii: ","):
                index += 1
            case UInt8(ascii: "]"):
                index += 1
                return .array(array)
            case nil:
                try handleEndOfInput("an array")
                return .array(array)
            default:
                throw JSONParseError(message: "Expected ',' or ']' in an array.", offset: index)
            }
        }
    }
}

/// Formats a Unicode scalar value as four uppercase hexadecimal digits.
///
/// Written by hand rather than with `String(format:)` so this file needs no `Foundation` import:
/// the parser is otherwise pure Swift, and keeping it that way avoids a platform dependency in
/// the one place performance and portability both matter.
func hexCodePoint(_ value: UInt32) -> String {
    let digits = "0123456789ABCDEF"
    var result = ""
    var shift = 12
    while shift >= 0 {
        let index = Int((value >> UInt32(shift)) & 0xF)
        result.append(Array(digits)[index])
        shift -= 4
    }
    return result
}

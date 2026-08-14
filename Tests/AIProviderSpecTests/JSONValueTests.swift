import Foundation
import Testing

@testable import AIProviderSpec

@Suite("JSONValue")
struct JSONValueTests {
    // MARK: - Literals and accessors

    @Test("Literal syntax builds the expected structure")
    func literalSyntax() {
        let value: JSONValue = [
            "name": "Ada",
            "age": 36,
            "score": 9.5,
            "active": true,
            "tags": ["mathematician", "programmer"],
            "manager": nil,
        ]

        #expect(value["name"]?.stringValue == "Ada")
        #expect(value["age"]?.intValue == 36)
        #expect(value["score"]?.doubleValue == 9.5)
        #expect(value["active"]?.boolValue == true)
        #expect(value["tags"]?[1]?.stringValue == "programmer")
        #expect(value["manager"]?.isNull == true)
    }

    @Test("Accessors do not coerce between cases")
    func accessorsDoNotCoerce() {
        #expect(JSONValue.int(1).doubleValue == nil)
        #expect(JSONValue.double(1.0).intValue == nil)
        #expect(JSONValue.string("1").intValue == nil)
        // `numberValue` is the lenient accessor and does bridge the two numeric cases.
        #expect(JSONValue.int(1).numberValue == 1.0)
        #expect(JSONValue.double(1.0).numberValue == 1.0)
    }

    @Test("Out-of-bounds and wrong-type subscripts return nil rather than trapping")
    func subscriptSafety() {
        let array: JSONValue = [1, 2, 3]
        #expect(array[5] == nil)
        #expect(array[-1] == nil)
        #expect(array["key"] == nil)
        #expect(JSONValue.string("x")[0] == nil)
    }

    @Test("Integers and doubles are distinct values")
    func integerDoubleDistinction() {
        #expect(JSONValue.int(1) != JSONValue.double(1.0))
    }

    // MARK: - Parsing

    @Test("Parses each JSON type", arguments: [
        ("null", JSONValue.null),
        ("true", .bool(true)),
        ("false", .bool(false)),
        ("0", .int(0)),
        ("-17", .int(-17)),
        ("3.5", .double(3.5)),
        ("1e3", .double(1000)),
        ("-2.5E-2", .double(-0.025)),
        ("\"text\"", .string("text")),
        ("[]", .array([])),
        ("{}", .object([:])),
    ])
    func parsesScalars(input: String, expected: JSONValue) throws {
        #expect(try JSONValue.parse(input) == expected)
    }

    @Test("Preserves the integer and double distinction through a parse")
    func parsePreservesNumericRepresentation() throws {
        #expect(try JSONValue.parse("1") == .int(1))
        #expect(try JSONValue.parse("1.0") == .double(1.0))
        // An exponent always yields a double, even when the value is integral.
        #expect(try JSONValue.parse("1e2") == .double(100))
    }

    @Test("Parses nested structures")
    func parsesNested() throws {
        let value = try JSONValue.parse(#"{"a":[1,{"b":null}],"c":{"d":true}}"#)
        #expect(value["a"]?[1]?["b"]?.isNull == true)
        #expect(value["c"]?["d"]?.boolValue == true)
    }

    @Test("Decodes string escapes")
    func parsesEscapes() throws {
        let value = try JSONValue.parse(#""a\"b\\c\/d\be\ff\ng\rh\ti""#)
        #expect(value.stringValue == "a\"b\\c/d\u{08}e\u{0C}f\ng\rh\ti")
    }

    @Test("Decodes unicode escapes including surrogate pairs")
    func parsesUnicodeEscapes() throws {
        #expect(try JSONValue.parse(#""é""#).stringValue == "é")
        // U+1F600 GRINNING FACE, expressed as a surrogate pair.
        #expect(try JSONValue.parse(#""😀""#).stringValue == "😀")
    }

    @Test("Rejects malformed input", arguments: [
        "",
        "{",
        "[1,",
        "\"unterminated",
        "{\"a\"}",
        "{\"a\":}",
        "{a:1}",
        "01",
        "-",
        "1.",
        "1e",
        "tru",
        "[1] extra",
        "'single quoted'",
        #""\ud83d""#,      // Unpaired high surrogate.
        #""\udc00""#,      // Unpaired low surrogate.
        #""\x""#,          // Unknown escape.
        "\"raw\ncontrol\"",  // Unescaped control character.
    ])
    func rejectsMalformedInput(input: String) {
        #expect(throws: JSONParseError.self) {
            try JSONValue.parse(input)
        }
    }

    @Test("Enforces a nesting depth limit")
    func enforcesDepthLimit() {
        let deep = String(repeating: "[", count: 200) + String(repeating: "]", count: 200)
        #expect(throws: JSONParseError.self) {
            try JSONValue.parse(deep, maximumDepth: 64)
        }
        #expect(throws: Never.self) {
            try JSONValue.parse(deep, maximumDepth: 512)
        }
    }

    // MARK: - Partial parsing

    @Test("Recovers values from truncated input", arguments: [
        (#"{"a": 1, "b": "hel"#, JSONValue(dictionaryLiteral: ("a", 1), ("b", "hel"))),
        (#"{"a": 1, "b""#, JSONValue(dictionaryLiteral: ("a", 1))),
        (#"{"a": 1, "b":"#, JSONValue(dictionaryLiteral: ("a", 1))),
        (#"{"a": 1, "b": tr"#, JSONValue(dictionaryLiteral: ("a", 1))),
        (#"{"a": 1, "b": -"#, JSONValue(dictionaryLiteral: ("a", 1))),
        (#"{"a": 1, "b": 1."#, JSONValue(dictionaryLiteral: ("a", 1))),
        ("[1, 2, 3", JSONValue.array([1, 2, 3])),
        ("[1, 2, ", JSONValue.array([1, 2])),
        (#"{"nested": {"deep": ["#, JSONValue(dictionaryLiteral: ("nested", ["deep": []]))),
    ])
    func recoversTruncatedInput(input: String, expected: JSONValue) throws {
        #expect(try JSONValue.parse(input, mode: .partial) == expected)
    }

    @Test("Keeps a syntactically complete number at the end of partial input")
    func keepsTrailingNumber() throws {
        // The next chunk may extend `12` to `123`, but `12` is the best snapshot available now.
        #expect(try JSONValue.parse(#"{"a": 12"#, mode: .partial) == ["a": 12])
    }

    @Test("Drops a truncated escape rather than emitting a partial one")
    func dropsTruncatedEscape() throws {
        #expect(try JSONValue.parse(#"{"a": "x\u00"#, mode: .partial) == ["a": "x"])
        #expect(try JSONValue.parse(#"{"a": "x\"#, mode: .partial) == ["a": "x"])
    }

    @Test("Every prefix of a valid document parses in partial mode")
    func everyPrefixParses() throws {
        let document = #"""
        {"id":"msg_1","choices":[{"index":0,"text":"Hello, world","done":true,"score":1.25}],\
        "meta":{"tags":["a","b"],"count":2,"ratio":0.5,"missing":null}}
        """#
            .replacingOccurrences(of: "\\\n", with: "")

        let characters = Array(document)
        for length in 1...characters.count {
            let prefix = String(characters[0..<length])
            #expect(throws: Never.self, "prefix of length \(length) failed: \(prefix)") {
                _ = try JSONValue.parse(prefix, mode: .partial)
            }
        }

        // The complete document must parse identically in both modes.
        #expect(try JSONValue.parse(document, mode: .partial) == JSONValue.parse(document))
    }

    @Test("Partial parsing grows monotonically as input arrives")
    func partialParsingGrowsMonotonically() throws {
        let document = #"{"a":"hello","b":[1,2,3]}"#
        var previousKeyCount = 0
        for length in 1...document.count {
            let prefix = String(Array(document)[0..<length])
            let value = try JSONValue.parse(prefix, mode: .partial)
            let keyCount = value.objectValue?.count ?? 0
            #expect(keyCount >= previousKeyCount, "key count regressed at length \(length)")
            previousKeyCount = keyCount
        }
    }

    // MARK: - Serialization

    @Test("Round-trips through serialization")
    func serializationRoundTrip() throws {
        let original: JSONValue = [
            "text": "quotes \" backslash \\ newline \n tab \t",
            "unicode": "héllo 😀",
            "int": 42,
            "double": 3.25,
            "negative": -7,
            "bool": false,
            "null": nil,
            "array": [1, "two", [3.0], ["four": 4]],
            "empty_object": [:],
            "empty_array": [],
        ]
        let reparsed = try JSONValue.parse(original.serialized())
        #expect(reparsed == original)
    }

    @Test("Escapes control characters")
    func escapesControlCharacters() {
        let value = JSONValue.string("\u{01}\u{1F}")
        #expect(value.serialized() == "\"\\u0001\\u001f\"")
    }

    @Test("Sorted keys produce stable output")
    func sortedKeysAreStable() {
        let value: JSONValue = ["z": 1, "a": 2, "m": 3]
        #expect(value.serialized(sortedKeys: true) == #"{"a":2,"m":3,"z":1}"#)
    }

    @Test("Non-finite doubles serialize as null")
    func nonFiniteDoublesSerializeAsNull() {
        #expect(JSONValue.double(.infinity).serialized() == "null")
        #expect(JSONValue.double(.nan).serialized() == "null")
    }

    @Test("Pretty printing is re-parseable")
    func prettyPrintingIsReparseable() throws {
        let value: JSONValue = ["a": [1, 2], "b": ["c": true]]
        let pretty = value.serialized(sortedKeys: true, prettyPrinted: true)
        #expect(pretty.contains("\n"))
        #expect(try JSONValue.parse(pretty) == value)
    }

    // MARK: - Codable interoperability

    @Test("Round-trips through Codable")
    func codableRoundTrip() throws {
        let original: JSONValue = [
            "string": "value",
            "int": 7,
            "double": 1.5,
            "bool": true,
            "null": nil,
            "nested": ["array": [1, 2, 3]],
        ]
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(JSONValue.self, from: encoded)
        #expect(decoded == original)
    }

    @Test("Codable keeps booleans distinct from numbers")
    func codableDistinguishesBooleansFromNumbers() throws {
        let decoded = try JSONDecoder().decode(JSONValue.self, from: Data(#"{"a":true,"b":1}"#.utf8))
        #expect(decoded["a"] == .bool(true))
        #expect(decoded["b"] == .int(1))
    }

    @Test("Nests inside other Codable types")
    func nestsInsideCodableTypes() throws {
        struct Envelope: Codable, Equatable {
            var name: String
            var payload: JSONValue
        }
        let original = Envelope(name: "test", payload: ["k": [1, 2]])
        let data = try JSONEncoder().encode(original)
        #expect(try JSONDecoder().decode(Envelope.self, from: data) == original)
    }

    @Test("Description renders sorted JSON")
    func descriptionIsSortedJSON() {
        let value: JSONValue = ["b": 2, "a": 1]
        #expect(value.description == #"{"a":1,"b":2}"#)
    }
}

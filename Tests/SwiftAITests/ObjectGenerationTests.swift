import AITestSupport
import Foundation
import Testing

@testable import SwiftAI

// MARK: - Fixtures

@Structured("A person.")
private struct Person {
    @Guidance("Their full name.")
    var name: String

    @Guidance("Their age in years.", .range(0...130))
    var age: Int

    var nickname: String?
}

private let personJSON = #"{"age":36,"name":"Ada","nickname":null}"#

// MARK: - Tests

@Suite("generateObject")
struct GenerateObjectTests {
    @Test("Decodes an object and reports the schema it asked for")
    func decodesObject() async throws {
        let model = MockLanguageModel(responses: [.text(personJSON)])
        let result = try await generateObject(model: model, of: Person.self, prompt: "Describe Ada.")

        #expect(result.object.name == "Ada")
        #expect(result.object.age == 36)
        #expect(result.object.nickname == nil)
        #expect(result.rawText == personJSON)

        guard case .json(let schema, let name, _)? = model.recordedOptions()?.responseFormat else {
            Issue.record("Expected a JSON response format.")
            return
        }
        #expect(name == "response")
        #expect(schema?.jsonValue()["properties"]?["name"]?["type"]?.stringValue == "string")
    }

    @Test("Wraps and unwraps a list")
    func decodesArray() async throws {
        let model = MockLanguageModel(
            responses: [.text(#"{"elements":[{"age":36,"name":"Ada"},{"age":41,"name":"Grace"}]}"#)]
        )
        let result = try await generateObject(model: model, arrayOf: Person.self, prompt: "Two people.")

        #expect(result.object.map(\.name) == ["Ada", "Grace"])

        // Most providers reject a bare array as a top-level response, so it travels in a wrapper.
        guard case .json(let schema, _, _)? = model.recordedOptions()?.responseFormat else {
            Issue.record("Expected a JSON response format.")
            return
        }
        #expect(schema?.jsonValue()["properties"]?["elements"]?["type"]?.stringValue == "array")
    }

    @Test("Chooses from a fixed set of values")
    func decodesEnumeration() async throws {
        let model = MockLanguageModel(responses: [.text(#"{"result":"positive"}"#)])
        let result = try await generateObject(
            model: model,
            enumOf: ["positive", "neutral", "negative"],
            prompt: "Classify: lovely."
        )
        #expect(result.object == "positive")
    }

    @Test("Rejects a value outside the permitted set")
    func rejectsValueOutsideEnumeration() async {
        let model = MockLanguageModel(responses: [.text(#"{"result":"ecstatic"}"#)])
        await #expect(throws: NoObjectGeneratedError.self) {
            try await generateObject(model: model, enumOf: ["positive", "negative"], prompt: "Classify.")
        }
    }

    @Test("Rejects an empty set of values")
    func rejectsEmptyEnumeration() async {
        let model = MockLanguageModel(text: "{}")
        await #expect(throws: InvalidArgumentError.self) {
            try await generateObject(model: model, enumOf: [], prompt: "Classify.")
        }
    }

    @Test("Generates against a runtime schema")
    func generatesAgainstRuntimeSchema() async throws {
        let model = MockLanguageModel(responses: [.text(#"{"city":"Malmö","population":360000}"#)])
        let schema = JSONSchema.object(
            properties: ["city": .string(), "population": .integer()],
            required: ["city", "population"]
        )

        let result = try await generateObject(model: model, schema: schema, prompt: "A Swedish city.")
        #expect(result.object["city"]?.stringValue == "Malmö")
    }

    @Test("Generates JSON with no schema")
    func generatesWithoutSchema() async throws {
        let model = MockLanguageModel(responses: [.text(#"{"anything":[1,2,3]}"#)])
        let result = try await generateObject(model: model, prompt: "Any JSON.")

        #expect(result.object["anything"]?[2]?.intValue == 3)
        guard case .json(let schema, _, _)? = model.recordedOptions()?.responseFormat else {
            Issue.record("Expected a JSON response format.")
            return
        }
        #expect(schema == nil)
    }

    // MARK: Failures

    @Test("Attaches the model's output when the JSON does not parse")
    func attachesOutputOnParseFailure() async throws {
        let model = MockLanguageModel(responses: [.text("Here you go: {name: Ada}")])

        var caught: NoObjectGeneratedError?
        do {
            _ = try await generateObject(model: model, of: Person.self, prompt: "Describe Ada.")
        } catch let error as NoObjectGeneratedError {
            caught = error
        }

        let error = try #require(caught)
        // Seeing what the model actually said is the whole point of this error type.
        #expect(error.text == "Here you go: {name: Ada}")
        #expect(error.description.contains("Ada"))
    }

    @Test("Attaches the model's output when the value does not match the schema")
    func attachesOutputOnValidationFailure() async throws {
        let model = MockLanguageModel(responses: [.text(#"{"name":"Ada"}"#)])

        var caught: NoObjectGeneratedError?
        do {
            _ = try await generateObject(model: model, of: Person.self, prompt: "Describe Ada.")
        } catch let error as NoObjectGeneratedError {
            caught = error
        }

        let error = try #require(caught)
        #expect(error.message.contains("age"))
    }

    @Test("Explains a truncated response as a token limit")
    func explainsTruncatedResponse() async throws {
        let model = MockLanguageModel(
            responses: [.text(#"{"name":"Ada","ag"#, finishReason: .length)]
        )

        var caught: NoObjectGeneratedError?
        do {
            _ = try await generateObject(model: model, of: Person.self, prompt: "Describe Ada.")
        } catch let error as NoObjectGeneratedError {
            caught = error
        }

        // The actionable advice is "raise maxOutputTokens", not "the JSON was malformed".
        let error = try #require(caught)
        #expect(error.finishReason == .length)
        #expect(error.message.contains("maxOutputTokens"))
    }

    @Test("Reports an empty response clearly")
    func reportsEmptyResponse() async {
        let model = MockLanguageModel(responses: [.text("")])
        await #expect(throws: NoObjectGeneratedError.self) {
            try await generateObject(model: model, of: Person.self, prompt: "Describe Ada.")
        }
    }
}

@Suite("streamObject")
struct StreamObjectTests {
    /// Splits a JSON document into small chunks so snapshots are exercised mid-token.
    private func chunkedModel(_ json: String, chunkSize: Int = 3) -> MockLanguageModel {
        MockLanguageModel(responses: [.text(json, textChunkSize: chunkSize)])
    }

    @Test("Publishes snapshots as fields arrive")
    func publishesSnapshots() async throws {
        let stream = streamObject(
            model: chunkedModel(personJSON),
            of: Person.self,
            prompt: "Describe Ada."
        )

        let snapshots = try await stream.partialStream.collect()
        #expect(!snapshots.isEmpty)

        // Every snapshot is valid on its own, and the last one is complete.
        let final = try #require(snapshots.last)
        #expect(final.name == "Ada")
        #expect(final.age == 36)
    }

    @Test("Snapshots only ever gain information")
    func snapshotsAreMonotonic() async throws {
        let stream = streamObject(
            model: chunkedModel(personJSON, chunkSize: 1),
            of: Person.self,
            prompt: "Describe Ada."
        )

        var sawName = false
        var sawAge = false
        for try await snapshot in stream.partialStream {
            if snapshot.name != nil { sawName = true }
            if snapshot.age != nil { sawAge = true }
            // Once a field has appeared it must never revert, or a bound view would flicker.
            if sawName { #expect(snapshot.name != nil) }
            if sawAge { #expect(snapshot.age != nil) }
        }
        #expect(sawName)
        #expect(sawAge)
    }

    @Test("Collapses snapshots that did not change")
    func collapsesUnchangedSnapshots() async throws {
        let stream = streamObject(
            model: chunkedModel(personJSON, chunkSize: 1),
            of: Person.self,
            prompt: "Describe Ada."
        )

        let snapshots = try await stream.partialStream.collect()
        // One snapshot per character would be wasteful; only meaningful changes are published.
        #expect(snapshots.count < personJSON.count)
    }

    @Test("Resolves the finished value")
    func resolvesFinishedValue() async throws {
        let stream = streamObject(
            model: chunkedModel(personJSON),
            of: Person.self,
            prompt: "Describe Ada."
        )

        let person = try await stream.object
        #expect(person.name == "Ada")
        #expect(try await stream.finishReason == .stop)
        #expect(try await stream.rawText == personJSON)
    }

    @Test("Exposes the raw JSON text")
    func exposesRawText() async throws {
        let stream = streamObject(
            model: chunkedModel(personJSON, chunkSize: 5),
            of: Person.self,
            prompt: "Describe Ada."
        )
        let text = try await stream.textStream.collect().joined()
        #expect(text == personJSON)
    }

    @Test("A malformed response fails when the value is read")
    func malformedResponseFails() async {
        let stream = streamObject(
            model: MockLanguageModel(responses: [.text("not json at all")]),
            of: Person.self,
            prompt: "Describe Ada."
        )
        await #expect(throws: NoObjectGeneratedError.self) { _ = try await stream.object }
    }

    // MARK: Arrays

    @Test("Publishes list elements as each one completes")
    func publishesElementsAsTheyComplete() async throws {
        let json = #"{"elements":[{"age":36,"name":"Ada"},{"age":41,"name":"Grace"},{"age":29,"name":"Karen"}]}"#
        let stream = streamObject(
            model: MockLanguageModel(responses: [.text(json, textChunkSize: 4)]),
            arrayOf: Person.self,
            prompt: "Three people."
        )

        let people = try await stream.elementStream.collect()
        #expect(people.map(\.name) == ["Ada", "Grace", "Karen"])
    }

    @Test("Publishes list snapshots with partially generated elements")
    func publishesListSnapshots() async throws {
        let json = #"{"elements":[{"age":36,"name":"Ada"},{"age":41,"name":"Grace"}]}"#
        let stream = streamObject(
            model: MockLanguageModel(responses: [.text(json, textChunkSize: 2)]),
            arrayOf: Person.self,
            prompt: "Two people."
        )

        let snapshots = try await stream.partialStream.collect()
        let final = try #require(snapshots.last)
        #expect(final.count == 2)
        #expect(final[0].name == "Ada")
        #expect(final[1].name == "Grace")

        // Somewhere in the middle there was a list with one complete entry and one in progress.
        #expect(snapshots.contains { $0.count == 1 })
    }

    @Test("Resolves the finished list")
    func resolvesFinishedList() async throws {
        let json = #"{"elements":[{"age":36,"name":"Ada"}]}"#
        let stream = streamObject(
            model: MockLanguageModel(responses: [.text(json, textChunkSize: 6)]),
            arrayOf: Person.self,
            prompt: "One person."
        )
        #expect(try await stream.object.map(\.name) == ["Ada"])
    }

    @Test("An empty list streams no elements and resolves empty")
    func emptyList() async throws {
        let stream = streamObject(
            model: MockLanguageModel(responses: [.text(#"{"elements":[]}"#, textChunkSize: 3)]),
            arrayOf: Person.self,
            prompt: "No people."
        )
        #expect(try await stream.elementStream.collect().isEmpty)
        #expect(try await stream.object.isEmpty)
    }

    // MARK: Parity

    @Test("Produces the same value as the buffered path")
    func parityWithGenerateObject() async throws {
        let buffered = try await generateObject(
            model: MockLanguageModel(responses: [.text(personJSON)]),
            of: Person.self,
            prompt: "Describe Ada."
        )
        let streamed = try await streamObject(
            model: MockLanguageModel(responses: [.text(personJSON, textChunkSize: 2)]),
            of: Person.self,
            prompt: "Describe Ada."
        ).object

        #expect(buffered.object.name == streamed.name)
        #expect(buffered.object.age == streamed.age)
    }
}

import Foundation
import Testing

@testable import AIProviderSpec

@Suite("Usage")
struct UsageTests {
    @Test("Falls back to the sum of input and output when no total is reported")
    func resolvesTotalFromComponents() {
        #expect(Usage(inputTokens: 10, outputTokens: 5).resolvedTotalTokens == 15)
        #expect(Usage(inputTokens: 10, outputTokens: 5, totalTokens: 100).resolvedTotalTokens == 100)
        #expect(Usage(inputTokens: 10).resolvedTotalTokens == 10)
        #expect(Usage.none.resolvedTotalTokens == nil)
    }

    @Test("Adding sums reported fields")
    func addingSumsFields() {
        let first = Usage(inputTokens: 10, outputTokens: 5, reasoningTokens: 2)
        let second = Usage(inputTokens: 3, outputTokens: 7, cachedInputTokens: 1)
        let total = first.adding(second)

        #expect(total.inputTokens == 13)
        #expect(total.outputTokens == 12)
        #expect(total.reasoningTokens == 2)
        #expect(total.cachedInputTokens == 1)
    }

    @Test("Adding leaves a field unreported only when neither side reports it")
    func addingPreservesUnreportedFields() {
        let total = Usage(inputTokens: 1).adding(Usage(outputTokens: 2))
        #expect(total.inputTokens == 1)
        #expect(total.outputTokens == 2)
        #expect(total.reasoningTokens == nil)
    }

    @Test("Adding is associative across a sequence of steps")
    func addingIsAssociative() {
        let steps = [
            Usage(inputTokens: 1, outputTokens: 1),
            Usage(inputTokens: 2, outputTokens: 2),
            Usage(inputTokens: 3, outputTokens: 3),
        ]
        let leftFold = steps.reduce(Usage.none) { $0.adding($1) }
        let rightFold = steps.reversed().reduce(Usage.none) { $0.adding($1) }
        #expect(leftFold == rightFold)
        #expect(leftFold.inputTokens == 6)
    }
}

@Suite("ProviderOptions")
struct ProviderOptionsTests {
    @Test("Reads values by provider namespace")
    func readsByNamespace() {
        let options: ProviderOptions = [
            "openai": ["reasoningEffort": "high"],
            "anthropic": ["thinking": ["type": "enabled"]],
        ]
        #expect(options.value("reasoningEffort", for: "openai")?.stringValue == "high")
        #expect(options.value("reasoningEffort", for: "anthropic") == nil)
        #expect(options["anthropic"]?["thinking"]?["type"]?.stringValue == "enabled")
    }

    @Test("Merging combines keys within a namespace rather than replacing it")
    func mergingIsPerKey() {
        let defaults: ProviderOptions = ["openai": ["reasoningEffort": "low", "store": true]]
        let overrides: ProviderOptions = ["openai": ["reasoningEffort": "high"]]
        let merged = defaults.merging(overrides)

        #expect(merged.value("reasoningEffort", for: "openai")?.stringValue == "high")
        // The key the override did not mention survives.
        #expect(merged.value("store", for: "openai")?.boolValue == true)
    }

    @Test("Merging adds namespaces that were absent")
    func mergingAddsNamespaces() {
        let merged = ProviderOptions(["openai": ["a": 1]])
            .merging(ProviderOptions(["anthropic": ["b": 2]]))
        #expect(merged.namespaces.keys.sorted() == ["anthropic", "openai"])
    }

    @Test("Metadata merges two optionals, yielding nil only when both are absent")
    func metadataOptionalMerging() {
        let left = ProviderMetadata(["openai": ["a": 1]])
        let right = ProviderMetadata(["openai": ["b": 2]])

        #expect(ProviderMetadata.merging(nil, nil) == nil)
        #expect(ProviderMetadata.merging(left, nil) == left)
        #expect(ProviderMetadata.merging(nil, right) == right)
        #expect(ProviderMetadata.merging(left, right)?["openai"]?.count == 2)
    }
}

@Suite("Content")
struct ContentTests {
    @Test("Extracts text while ignoring other content")
    func extractsText() {
        let content: [ModelContent] = [
            .reasoning(ReasoningPart("thinking...")),
            .text("Hello, "),
            .toolCall(ToolCallPart(toolCallID: "1", toolName: "lookup", input: ["q": "x"])),
            .text("world."),
        ]
        #expect(content.text == "Hello, world.")
        #expect(content.reasoningText == "thinking...")
        #expect(content.toolCalls.count == 1)
        #expect(content.toolCalls.first?.toolName == "lookup")
    }

    @Test("Reasoning text is nil when there is no reasoning")
    func reasoningTextIsNilWhenAbsent() {
        #expect([ModelContent.text("plain")].reasoningText == nil)
    }

    @Test("File parts expose their bytes and data URI")
    func filePartAccessors() {
        let bytes = Data([0xDE, 0xAD, 0xBE, 0xEF])
        let part = FilePart.data(bytes, mediaType: "image/png")

        #expect(part.isImage)
        #expect(part.data == bytes)
        #expect(part.url == nil)
        #expect(part.base64EncodedString == bytes.base64EncodedString())
        #expect(part.dataURI == "data:image/png;base64,\(bytes.base64EncodedString())")
    }

    @Test("URL file parts carry no inline bytes")
    func urlFilePartAccessors() {
        let part = FilePart.url(URL(string: "https://example.com/a.pdf")!, mediaType: "application/pdf")
        #expect(part.data == nil)
        #expect(part.dataURI == nil)
        #expect(part.isImage == false)
    }

    @Test("Tool result outputs report whether they are failures")
    func toolResultOutputErrorFlag() {
        #expect(ToolResultOutput.text("ok").isError == false)
        #expect(ToolResultOutput.json(["a": 1]).isError == false)
        #expect(ToolResultOutput.errorText("boom").isError)
        #expect(ToolResultOutput.errorJSON(["code": 500]).isError)
    }

    @Test("String literals build user content and messages")
    func stringLiteralConveniences() {
        let content: UserContent = "hello"
        #expect(content == .text(TextPart("hello")))

        let message: ModelMessage = "hello"
        #expect(message.role == .user)
        #expect(message.text == "hello")
    }

    @Test("Messages expose role and text uniformly")
    func messageInspection() {
        let messages: [ModelMessage] = [
            .system("Be brief."),
            .user("Hi"),
            .assistant("Hello"),
            .tool([ToolResultPart(toolCallID: "1", toolName: "t", output: .text("done"))]),
        ]
        #expect(messages.map(\.role) == [.system, .user, .assistant, .tool])
        #expect(messages.map(\.text) == ["Be brief.", "Hi", "Hello", ""])
    }
}

@Suite("Errors")
struct ErrorTests {
    @Test("API call errors classify retryable status codes", arguments: [
        (408, true), (409, true), (429, true), (500, true), (503, true),
        (400, false), (401, false), (403, false), (404, false), (422, false),
    ])
    func classifiesRetryableStatusCodes(statusCode: Int, isRetryable: Bool) {
        #expect(APICallError.defaultIsRetryable(statusCode: statusCode) == isRetryable)
    }

    @Test("A request that never completed is treated as retryable")
    func missingStatusCodeIsRetryable() {
        #expect(APICallError.defaultIsRetryable(statusCode: nil))
    }

    @Test("An explicit retryability flag overrides the default classification")
    func explicitRetryabilityWins() {
        let error = APICallError(
            message: "Quota exhausted for the month.",
            url: URL(string: "https://example.com")!,
            statusCode: 429,
            isRetryable: false
        )
        #expect(error.isRetryable == false)
    }

    @Test("Validation errors build a path from the inside out")
    func validationErrorPathBuilding() {
        let error = TypeValidationError(message: "Expected a number.")
            .prependingPath("quantity")
            .prependingPath("2")
            .prependingPath("ingredients")

        #expect(error.path == "ingredients.2.quantity")
        #expect(error.description.contains("ingredients.2.quantity"))
    }

    @Test("Errors carry a stable machine-readable name")
    func errorsHaveStableNames() {
        let errors: [any AISDKError] = [
            APICallError(message: "x", url: URL(string: "https://example.com")!),
            JSONParseError(message: "x"),
            TypeValidationError(message: "x"),
            UnsupportedFunctionalityError(functionality: "x"),
            NoSuchModelError(modelID: "x", modelKind: .language),
            InvalidArgumentError(argument: "x", message: "y"),
        ]
        #expect(errors.allSatisfy { $0.name.hasPrefix("AI_") })
        #expect(errors.allSatisfy { $0.description.contains($0.name) })
    }

    @Test("Missing credentials name the environment variable to set")
    func missingAPIKeyIsActionable() {
        let error = MissingAPIKeyError(provider: "openai", environmentVariable: "OPENAI_API_KEY")
        #expect(error.message.contains("OPENAI_API_KEY"))
    }
}

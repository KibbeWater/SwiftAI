# SwiftAI

One API for every language model provider, designed for Swift rather than translated into it.

SwiftAI takes the architecture of the [Vercel AI SDK](https://ai-sdk.dev) — a versioned provider
specification with a thin orchestration layer above it — and gives it a surface that belongs in
Swift: macro-generated schemas, snapshot streaming, typed tools, and strict concurrency throughout.

```swift
import AIAnthropic
import SwiftAI

let anthropic = AnthropicProvider(apiKey: key)
let result = try await generateText(
    model: anthropic.languageModel("claude-sonnet-4-5"),
    prompt: "Explain kinetic energy in one sentence."
)
print(result.text)
```

Swapping providers is a one-line change. Everything else — tools, streaming, structured output,
retries, middleware — works identically across all of them.

---

## Contents

- [Installation](#installation)
- [Generating text](#generating-text)
- [Streaming](#streaming)
- [Structured output](#structured-output)
- [Tools](#tools)
- [Agents](#agents)
- [Middleware](#middleware)
- [Embeddings](#embeddings)
- [Evaluation](#evaluation)
- [Images, speech, and transcription](#images-speech-and-transcription)
- [Providers](#providers)
- [Testing](#testing)
- [Architecture](#architecture)

---

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/KibbeWater/SwiftAI.git", from: "0.1.0")
]
```

Then depend on the core plus whichever providers you use:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "SwiftAI", package: "SwiftAI"),
        .product(name: "AIOpenAI", package: "SwiftAI"),
    ]
)
```

Requires Swift 6.1 and runs on macOS 13+, iOS 16+, tvOS 16+, watchOS 9+, visionOS 1+, and Linux.

---

## Generating text

```swift
let result = try await generateText(
    model: model,
    system: "You answer in one short paragraph.",
    prompt: "Why is the sky blue?"
)

result.text        // the answer
result.usage       // token counts
result.finishReason
```

Continue a conversation by appending what the last call produced. Do it this way rather than
rebuilding messages by hand: `responseMessages` preserves reasoning signatures and tool call
identifiers that providers require on the next turn.

```swift
var history: [ModelMessage] = [.user("Why is the sky blue?")]
let first = try await generateText(model: model, prompt: Prompt(history))

history += first.responseMessages
history.append(.user("And at sunset?"))
let second = try await generateText(model: model, prompt: Prompt(history))
```

Attach images or documents as content parts:

```swift
let result = try await generateText(
    model: model,
    prompt: Prompt([
        .text("What is in this picture?"),
        .file(.data(imageBytes, mediaType: "image/png")),
    ])
)
```

If the provider cannot fetch a URL itself, SwiftAI downloads it and substitutes the bytes, so the
same prompt works everywhere.

---

## Streaming

`streamText` returns immediately and publishes output as it arrives. It does not throw: a failure
surfaces when you read the result.

```swift
let stream = streamText(model: model, prompt: "Write a haiku about Malmö.")

for try await delta in stream.textStream {
    print(delta, terminator: "")
}

print("\nUsed \(try await stream.totalUsage)")
```

Every view of the result is independent, replayed from the start, and consumable at any point —
including after the generation has finished. Use `fullStream` when you want to show more than the
answer:

```swift
for try await part in stream.fullStream {
    switch part {
    case .reasoningDelta(_, let delta): thinking += delta
    case .textDelta(_, let delta): answer += delta
    case .toolWillRun(let call): status = "Running \(call.toolName)…"
    case .toolResult: status = "Thinking…"
    case .finish(let reason, let usage): log(reason, usage)
    default: break
    }
}
```

Cancelling the consuming task, calling `cancel()`, or simply discarding the result aborts the
upstream request.

---

## Structured output

Describe the shape you want as a Swift type. The `@Structured` macro generates the JSON Schema,
the decoder, and a partial type used for streaming.

```swift
@Structured("A recipe with its ingredients.")
struct Recipe {
    @Guidance("The name of the dish.")
    var name: String

    @Guidance("How many people it serves.", .range(1...12))
    var servings: Int

    var ingredients: [Ingredient]
    var notes: String?          // Optional, so not required.
}

@Structured("A single ingredient.")
struct Ingredient {
    var name: String
    var quantity: Double
    @Guidance("The unit of measurement.", .anyOf(["g", "ml", "piece"]))
    var unit: String
}

let result = try await generateObject(model: model, of: Recipe.self, prompt: "A quick pasta dish.")
result.object.ingredients.first?.name
```

On providers with constrained decoding the output is guaranteed to parse; on the rest, the schema
is supplied as an instruction and a warning tells you conformance is not guaranteed.

There are variants for lists, classification, runtime schemas, and schema-less JSON:

```swift
try await generateObject(model: model, arrayOf: Recipe.self, prompt: "Three dinners.")
try await generateObject(model: model, enumOf: ["positive", "neutral", "negative"], prompt: review)
try await generateObject(model: model, schema: runtimeSchema, prompt: "…")
try await generateObject(model: model, prompt: "Any JSON.")
```

### Streaming objects

`streamObject` publishes snapshots. Every snapshot is a complete, valid value in which properties
that have not arrived yet are `nil` — never a half-parsed string, never an invalid intermediate.
That makes it directly bindable:

```swift
let stream = streamObject(model: model, of: Recipe.self, prompt: "A quick pasta dish.")

for try await recipe in stream.partialStream {
    await MainActor.run {
        title = recipe.name ?? "…"
        ingredients = recipe.ingredients ?? []
    }
}
let complete = try await stream.object
```

Snapshots only ever gain information, so a bound view never has to handle a field disappearing.
For lists, `elementStream` publishes each element as soon as it is complete.

---

## Tools

A tool is a name, a description of when to use it, and a typed argument structure.

```swift
struct WeatherTool: Tool {
    @Structured("Where and when to look up the weather.")
    struct Arguments {
        @Guidance("The city, as the user wrote it.")
        var city: String
        @Guidance("How many days ahead to forecast.", .range(1...7))
        var days: Int
    }

    let client: WeatherClient
    var description: String { "Look up the weather forecast for a city." }

    func call(_ arguments: Arguments, context: ToolContext) async throws -> ToolOutput {
        .json(try await client.forecast(city: arguments.city, days: arguments.days))
    }
}
```

Or, for something that is just a closure:

```swift
let weather = tool("weather", description: "Look up the weather in a city.") {
    (query: CityQuery, _) in
    .text(try await forecast(for: query.city))
}
```

By default a call runs a single step: the model may ask for a tool, but the result is not sent
back. Supply a step budget to make it a loop.

```swift
let result = try await generateText(
    model: model,
    prompt: "Should I take an umbrella in Malmö tomorrow?",
    tools: [weather],
    stopWhen: [.stepCount(5)]
)
```

Each pass runs the model, executes the tools it asked for concurrently, and feeds the results
back. A tool that throws does not fail the generation: the error is reported to the model as a
tool result so it can recover.

`clientTool` declares a tool your application resolves — a confirmation, a file picker, a payment.
The loop stops and hands you the call.

---

## Agents

An agent bundles a model, its instructions, and its tools so the call site carries only the
prompt.

```swift
let researcher = Agent(
    model: model,
    system: """
        You answer questions about the company's documentation. Search before answering, \
        and cite the page you used.
        """,
    tools: [searchTool, fetchPageTool],
    stopWhen: [.stepCount(8)]
)

let answer = try await researcher.generate(prompt: "What is our refund policy?")
let live = researcher.stream(prompt: "And for digital goods?")
```

Agents are values: `researcher.with(settings: .deterministic)` makes a variant without disturbing
the original.

---

## Middleware

Middleware wraps a model at the provider boundary, so the same logic works against every provider.

```swift
let model = wrapLanguageModel(
    model: openai.languageModel("gpt-5"),
    middleware: [
        RequestLogger(),                            // outermost: sees everything
        DefaultSettingsMiddleware(temperature: 0.2),
        ExtractReasoningMiddleware(),               // closest to the model
    ]
)
```

Three are built in:

| Middleware | What it does |
| --- | --- |
| `DefaultSettingsMiddleware` | Fills in settings a call did not specify. The call site always wins. |
| `ExtractReasoningMiddleware` | Lifts `<think>…</think>` out of the text channel into proper reasoning parts. |
| `SimulateStreamingMiddleware` | Produces a well-formed stream from a model that only supports buffered calls. |

Writing your own means implementing one method. Returning without calling `next` short-circuits
the call, which is how a cache is built.

---

## Embeddings

```swift
let result = try await embedMany(model: openai.embeddingModel("text-embedding-3-small"), values: documents)
let similarity = try cosineSimilarity(result.embeddings[0], result.embeddings[1])
```

Inputs larger than a provider accepts are split into batches, sent with bounded concurrency, and
reassembled in the original order.

---

## Evaluation

> **Experimental.** The evaluation API may change in a minor release.

`evaluate` answers named questions about one piece of state: pick an option, place it on a
rubric, or estimate how likely something is to be true.

```swift
enum Department: String, EvaluationChoice {
    case billing, support
}

let result = try await evaluate(
    model: openai.evaluationModel("gpt-5-mini"),
    state: ["message": "I was charged twice. Please refund the extra charge."],
    questions: [
        "department": .choice("Which team should handle this?", options: Department.self),
        "severity": .score("How severe is it?", levels: ["Cosmetic", "Workaround exists", "Blocking"]),
        "refund": .boolean("Is the customer asking for money back?"),
    ]
)

result.choice("department", as: Department.self)  // .billing
result["severity"]?.score                          // 0.4, a position on the zero-based rubric
result["refund"]?.probability                      // 0.97, P(true) — not confidence
```

A boolean answer is always P(true), so `0.02` is a confident no. Some providers also return a
probability distribution for choices and scores. Answers are checked before they are returned:
every question gets one answer of its own kind, distributions are complete and sum to one, and
scores agree with their distribution. Nothing is renormalized.

OpenAI, Anthropic, and Google answer through structured output with reasoning turned down. Every
question goes in a single prompt, and no distributions come back. Native evaluation endpoints judge
each question on its own and do return distributions. Either way, check a model's judgments
against your own labeled examples before choosing thresholds.

---

## Images, speech, and transcription

```swift
let image = try await generateImage(model: openai.imageModel("gpt-image-1"), prompt: "A watercolour of Malmö.")
let audio = try await generateSpeech(model: openai.speechModel("gpt-4o-mini-tts"), text: "Good morning.")
let text  = try await transcribe(model: openai.transcriptionModel("whisper-1"), audio: bytes, mediaType: "audio/mpeg")
```

---

## Providers

| Product | Covers |
| --- | --- |
| `AIOpenAI` | Responses API, Chat Completions, embeddings, images, speech, transcription, evaluation |
| `AIAnthropic` | Messages API, extended thinking, prompt caching, evaluation |
| `AIGoogle` | Gemini `generateContent`, thinking, embeddings, evaluation |
| `AIOpenRouter` | Hundreds of models through one key: chat with reasoning replay, web search, embeddings, images, decisions |
| `AIOpenAICompatible` | Any OpenAI-compatible server: Ollama, vLLM, Groq, Together, LM Studio |

### OpenRouter

```swift
let openrouter = OpenRouterProvider(appName: "My App")  // Reads OPENROUTER_API_KEY.

var settings = GenerationSettings()
settings.providerOptions = ["openrouter": [
    "models": ["anthropic/claude-haiku-4.5", "openai/gpt-5-mini"],  // Fallbacks, in order.
    "provider": ["sort": "throughput"],
    "reasoning": ["max_tokens": 1024],
]]

let result = try await generateText(
    model: openrouter.languageModel("anthropic/claude-haiku-4.5"),
    prompt: "What changed in the latest Swift release?",
    tools: [OpenRouterTools.webSearch(maxResults: 3)],
    settings: settings
)
result.providerMetadata?["openrouter"]?["cost"]  // What the call cost, in credits.
```

Options under `"openrouter"` go into the request body under their documented wire names, so
OpenRouter features that arrive after this package was written already work. Reasoning is
replayed across turns through `responseMessages`, including Anthropic's and Gemini's signatures and
OpenAI's encrypted reasoning. `decisionModel(_:)` serves OpenRouter's Decisions API for
[evaluation](#evaluation).

Provider-specific features are reached through namespaced options rather than a union of every
provider's settings:

```swift
var settings = GenerationSettings(maxOutputTokens: 16_000)
settings.providerOptions = [
    "anthropic": ["thinking": ["type": "enabled", "budget_tokens": 8_000]],
    "openai": ["reasoning": ["effort": "high"]],
]
```

A provider ignores namespaces that are not its own, so one settings value can travel across
models.

Resolve models by string when they come from configuration:

```swift
let registry = ProviderRegistry([
    "openai": OpenAIProvider(apiKey: openAIKey),
    "anthropic": AnthropicProvider(apiKey: anthropicKey),
])
let model = try registry.languageModel("anthropic:claude-sonnet-4-5")
```

---

## Testing

Everything reaches the network through one protocol, so the whole SDK is testable without it.
`AITestSupport` provides the doubles.

```swift
import AITestSupport

// Replace the model, to test logic built on top of it.
let model = MockLanguageModel(responses: [
    .toolCall(name: "weather", input: ["city": "Malmö"]),
    .text("Yes, take an umbrella."),
])
let result = try await generateText(model: model, prompt: "Umbrella?", tools: [weather], stopWhen: [.stepCount(3)])

// Or replace the transport, to test a real provider against recorded bytes.
let transport = MockHTTPTransport(exchange: .serverSentEvents(recordedFixture, chunkSize: 7))
let real = OpenAIProvider(apiKey: "test", transport: transport).languageModel("gpt-5")
```

The `chunkSize` is the interesting part: splitting a fixture at awkward boundaries is how stream
parsing gets exercised, and it is the condition that breaks naive implementations under load.

---

## Architecture

The package is layered so that providers and core can evolve independently:

```
AIProviderSpec     pure types: protocols, messages, JSON, errors. No dependencies.
AIProviderUtils    HTTP transport, SSE parsing, retries, JSON helpers.
SwiftAI            generation functions, tool loop, agents, middleware, macros.
AIOpenAI, AIAnthropic, AIGoogle, AIOpenAICompatible
                   depend on the spec and utils only — never on SwiftAI.
```

A provider implements two methods:

```swift
public protocol LanguageModelV2: Sendable {
    var provider: String { get }
    var modelID: String { get }
    func generate(_ options: LanguageModelCallOptions) async throws -> LanguageModelResponse
    func stream(_ options: LanguageModelCallOptions) async throws -> LanguageModelStreamResponse
}
```

Cross-cutting concerns are deliberately not a provider's problem. Retries, the tool loop, prompt
normalization, and file downloading all happen above this line, once. A provider translates to and
from its wire format and nothing else — which is why adding one is a few hundred lines rather than
a project.

Settings a provider cannot honor produce a `CallWarning` rather than an error, so a call that uses
`topK` still succeeds against a provider that has no such setting, and tells you what was dropped.

---

## Design notes

A few decisions worth knowing about, because they differ from what a direct port would have done:

- **Untyped `throws`.** A single call aggregates failures from networking, decoding, schema
  validation, and user-supplied tool bodies. Errors conform to `AISDKError` and carry structured
  detail; catch the specific type you care about.
- **Existentials, not parameter packs.** Tools are `[any Tool]`. Dispatch is inherently dynamic —
  the model picks a tool by name at runtime — so tracking the set in the type system buys nothing.
- **Snapshot streaming for objects, deltas for text.** A half-received object is a valid value with
  `nil` fields, not a parse error.
- **A hand-written JSON parser.** It behaves identically on Darwin and Linux, preserves the
  integer/double distinction, and can recover values from truncated input — which is what makes
  streaming structured output possible.

---

## License

MIT. See [LICENSE](LICENSE).

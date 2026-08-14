# ``SwiftAI``

One API for every language model provider, designed for Swift rather than translated into it.

## Overview

SwiftAI gives you a single set of generation functions that work identically across OpenAI,
Anthropic, Google, and any OpenAI-compatible server. Swapping providers is a one-line change;
tools, streaming, structured output, retries, and middleware behave the same underneath all of
them.

```swift
import AIAnthropic
import SwiftAI

let anthropic = AnthropicProvider(apiKey: key)
let result = try await generateText(
    model: anthropic.languageModel("claude-sonnet-4-5"),
    prompt: "Explain kinetic energy in one sentence."
)
```

### Structured output

Describe the shape you want as a Swift type. ``Structured(_:)`` generates the JSON Schema, the
decoder, and the partial type that streaming publishes.

```swift
@Structured("A recipe with its ingredients.")
struct Recipe {
    @Guidance("The name of the dish.") var name: String
    @Guidance("How many it serves.", .range(1...12)) var servings: Int
    var ingredients: [String]
}

let recipe = try await generateObject(model: model, of: Recipe.self, prompt: "A pasta dish.").object
```

### Tools and agents

Give a model tools and a step budget, and a single call becomes an agentic loop: the model asks
for tools, they run concurrently, the results go back, and it answers.

```swift
let result = try await generateText(
    model: model,
    prompt: "Should I take an umbrella tomorrow?",
    tools: [weatherTool],
    stopWhen: [.stepCount(5)]
)
```

## Topics

### Generating text

- ``generateText(model:system:prompt:tools:toolChoice:settings:stopWhen:prepareStep:onStepFinish:)``
- ``streamText(model:system:prompt:tools:toolChoice:settings:stopWhen:prepareStep:onStepFinish:onFinish:onError:)``
- ``GenerateTextResult``
- ``StreamTextResult``
- ``TextStreamPart``

### Building prompts

- ``Prompt``
- ``GenerationSettings``

### Structured output

- ``Structured(_:)``
- ``Guidance(_:_:)``
- ``StructuredValue``
- ``StructuredOutput``
- ``SchemaConstraint``
- ``generateObject(model:of:system:prompt:settings:)``
- ``streamObject(model:of:system:prompt:settings:)``
- ``GenerateObjectResult``
- ``StreamObjectResult``
- ``StreamArrayResult``

### Tools

- ``Tool``
- ``tool(_:description:arguments:providerOptions:execute:)``
- ``dynamicTool(_:description:inputSchema:providerOptions:execute:)``
- ``clientTool(_:description:inputSchema:providerOptions:)``
- ``ToolContext``
- ``ToolExecution``

### Multi-step generation

- ``Agent``
- ``StopCondition``
- ``StepResult``
- ``PrepareStepContext``
- ``PrepareStepAdjustments``

### Middleware

- ``LanguageModelMiddleware``
- ``wrapLanguageModel(model:middleware:provider:modelID:)``
- ``DefaultSettingsMiddleware``
- ``ExtractReasoningMiddleware``
- ``SimulateStreamingMiddleware``

### Choosing models

- ``ProviderRegistry``
- ``CustomProvider``

### Embeddings

- ``embed(model:value:retryPolicy:headers:providerOptions:)``
- ``embedMany(model:values:maximumParallelCalls:retryPolicy:headers:providerOptions:)``
- ``cosineSimilarity(_:_:)``

### Images, speech, and transcription

- ``generateImage(model:prompt:count:size:aspectRatio:seed:retryPolicy:headers:providerOptions:)``
- ``generateSpeech(model:text:voice:outputFormat:instructions:speed:language:retryPolicy:headers:providerOptions:)``
- ``transcribe(model:audio:mediaType:filename:retryPolicy:headers:providerOptions:)``

### Errors

- ``NoObjectGeneratedError``
- ``NoSuchToolError``
- ``InvalidToolInputError``

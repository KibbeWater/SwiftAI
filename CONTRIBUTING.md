# Contributing

## Getting set up

```bash
swift build
swift test
```

Swift 6.1 or later. Everything must build and pass on both macOS and Linux — Linux is a supported
target, not a best-effort one.

## The layering rule

The dependency graph is the most important invariant in the package:

```
AIProviderSpec  ←  AIProviderUtils  ←  SwiftAI
       ↑                  ↑
       └────── providers ─┘
```

**Providers must never import `SwiftAI`, and `SwiftAI` must never import a provider.** That
inversion is what lets a provider ship on its own schedule and what keeps the core from growing
provider-shaped special cases. If you find yourself wanting to break it, the thing you need
probably belongs in `AIProviderSpec`.

## Adding a provider

A provider implements `LanguageModelV2`, which is two methods. It should not implement anything
else, because everything else already exists above it:

- **Retries** happen in the core, driven by `APICallError.isRetryable`. Classify failures
  accurately and throw; do not retry.
- **The tool loop** happens in the core. A provider handles exactly one round trip.
- **Prompt normalization** happens in the core. You receive a flat message list and a tool list
  already reduced to JSON Schema.
- **File downloading** happens in the core. Report what you can fetch natively through
  `supportsNativeURL(_:mediaType:)` and you will receive bytes for everything else.

Two conventions matter for how a provider behaves when it cannot do something:

- A setting with no equivalent produces a `CallWarning` and the call still succeeds. Throwing
  `UnsupportedFunctionalityError` is for requests that cannot be served at all.
- A stream must always be well formed: a `streamStart` first, matching start/end parts around every
  block, and a `finish` last. If the wire format has no block boundaries, synthesize them — that
  guarantee is what lets consumers write provider-agnostic code.

### Fixtures

Provider tests run against payloads recorded from the real API and checked into
`Tests/<Provider>Tests/Fixtures`. Record them verbatim, including fields the SDK ignores: a
provider changing its output shape shows up there first.

Streaming tests should run the same fixture at several chunk sizes:

```swift
@Test("Streams text", arguments: [1, 7, 64, 4096])
func streamsText(chunkSize: Int) async throws {
    let transport = MockHTTPTransport(exchange: .serverSentEvents(fixture, chunkSize: chunkSize))
    …
}
```

A chunk size of one splits every event across many deliveries. This is not paranoia: events
routinely arrive split across TCP segments in production, and a parser that assumes otherwise
fails intermittently and only under load.

## Testing

- `swift-testing` for new tests. The macro expansion tests use XCTest because
  `assertMacroExpansion` reports through `XCTFail`.
- No network access in the default suite. Substitute `MockHTTPTransport` to test a provider, or
  `MockLanguageModel` to test anything above it.
- Test names read as sentences describing the behaviour, not the method under test:
  `"Counts cached tokens toward the input total"`, not `"testUsage"`.
- Where a test encodes a non-obvious decision, say why in a comment. The next person to read a
  failing assertion needs to know whether the expectation or the code is wrong.

## Style

- Document every public declaration. Say what it is for and when to reach for it; the signature
  already says what it takes.
- Comment the *why*, never the *what*. A comment explaining that a line assigns a variable is
  noise; one explaining that Anthropic rejects an empty assistant turn is the reason the code
  looks the way it does.
- Prefer a warning to an error wherever a call can still produce a useful result.
- Errors should name the fix. `MissingAPIKeyError` names the environment variable to set;
  `NoSuchToolError` lists the tools that were available.

## Before opening a pull request

```bash
swift build --build-tests   # Also catches missing imports, which only fail on Linux otherwise.
swift test
```

`MemberImportVisibility` is enabled package-wide, so a file that relies on `Foundation` leaking in
from a sibling file fails to compile on macOS rather than only on Linux. If you see that
diagnostic, add the import.

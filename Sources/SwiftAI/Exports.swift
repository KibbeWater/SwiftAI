/// The provider specification is re-exported so that `import SwiftAI` is all an application
/// needs.
///
/// Application code works with ``ModelMessage``, ``JSONValue``, ``Usage``, and the error types
/// constantly; making callers import a second module to name them would be noise. Provider
/// authors still import ``AIProviderSpec`` directly, which is what keeps providers independent of
/// this module.
@_exported import AIProviderSpec

/// The provider utilities are re-exported because parts of them are in this module's own public
/// API: ``GenerationSettings/retryPolicy`` is a `RetryPolicy`, and
/// ``GenerationSettings/fileTransport`` is an `HTTPTransport`. Substituting a transport is also
/// the supported way to test code built on this package without a network.
@_exported import AIProviderUtils

// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

/// Settings applied to every first-party target.
///
/// `ExistentialAny` is enabled package-wide so that existential types are always spelled
/// `any Protocol`. The SDK leans on existentials heavily (`any LanguageModel`, `any Tool`),
/// and making them explicit keeps the cost of dynamic dispatch visible at every use site.
/// `MemberImportVisibility` requires every file to import the modules whose members it uses,
/// rather than inheriting them from a sibling file. That matters here because the package targets
/// Linux, where a file that relies on `Foundation` leaking in from elsewhere in the module fails
/// to build — a class of error that would otherwise only surface in CI.
let sharedSwiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

let package = Package(
    name: "SwiftAI",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        // The umbrella product: core generation functions, tools, agents, middleware.
        // Re-exports `AIProviderSpec`, so importing `SwiftAI` is enough for application code.
        .library(name: "SwiftAI", targets: ["SwiftAI"]),

        // Provider packages. Each is a separate product so applications link only what they use.
        .library(name: "AIOpenAI", targets: ["AIOpenAI"]),
        .library(name: "AIAnthropic", targets: ["AIAnthropic"]),
        .library(name: "AIGoogle", targets: ["AIGoogle"]),
        .library(name: "AIOpenAICompatible", targets: ["AIOpenAICompatible"]),
        .library(name: "AIOpenRouter", targets: ["AIOpenRouter"]),

        // Test doubles for consumers writing tests against their own SwiftAI integrations.
        .library(name: "AITestSupport", targets: ["AITestSupport"]),

        // The provider authoring surface, for third parties writing their own providers.
        .library(name: "AIProviderSpec", targets: ["AIProviderSpec"]),
        .library(name: "AIProviderUtils", targets: ["AIProviderUtils"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "600.0.0"..<"603.0.0"),
    ],
    targets: [
        // MARK: - Specification layer

        // Pure types with no dependencies. Providers and core both build on this, which is what
        // keeps them decoupled from one another. Mirrors `@ai-sdk/provider`.
        .target(
            name: "AIProviderSpec",
            swiftSettings: sharedSwiftSettings
        ),

        // Shared runtime for provider authors: HTTP transport, SSE parsing, JSON helpers,
        // error mapping and retries. Mirrors `@ai-sdk/provider-utils`.
        .target(
            name: "AIProviderUtils",
            dependencies: ["AIProviderSpec"],
            swiftSettings: sharedSwiftSettings
        ),

        // MARK: - Macros

        .macro(
            name: "SwiftAIMacrosImpl",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            swiftSettings: sharedSwiftSettings
        ),

        // MARK: - Core

        .target(
            name: "SwiftAI",
            dependencies: ["AIProviderSpec", "AIProviderUtils", "SwiftAIMacrosImpl"],
            swiftSettings: sharedSwiftSettings
        ),

        // MARK: - Providers
        //
        // Providers depend on the spec and utils only. They never import `SwiftAI`, and
        // `SwiftAI` never imports a provider. This inversion is what lets providers and core
        // evolve independently, and it is enforced structurally by these dependency lists.

        .target(
            name: "AIOpenAICompatible",
            dependencies: ["AIProviderSpec", "AIProviderUtils"],
            swiftSettings: sharedSwiftSettings
        ),
        .target(
            name: "AIOpenAI",
            dependencies: ["AIProviderSpec", "AIProviderUtils", "AIOpenAICompatible"],
            swiftSettings: sharedSwiftSettings
        ),
        .target(
            name: "AIAnthropic",
            dependencies: ["AIProviderSpec", "AIProviderUtils"],
            swiftSettings: sharedSwiftSettings
        ),
        .target(
            name: "AIGoogle",
            dependencies: ["AIProviderSpec", "AIProviderUtils"],
            swiftSettings: sharedSwiftSettings
        ),
        .target(
            name: "AIOpenRouter",
            dependencies: ["AIProviderSpec", "AIProviderUtils"],
            swiftSettings: sharedSwiftSettings
        ),

        // MARK: - Test support

        .target(
            name: "AITestSupport",
            dependencies: ["AIProviderSpec", "AIProviderUtils"],
            swiftSettings: sharedSwiftSettings
        ),

        // MARK: - Tests

        .testTarget(
            name: "AIProviderSpecTests",
            dependencies: ["AIProviderSpec"],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "AIProviderUtilsTests",
            dependencies: ["AIProviderUtils", "AITestSupport"],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "SwiftAITests",
            dependencies: ["SwiftAI", "AITestSupport"],
            swiftSettings: sharedSwiftSettings
        ),
        // Macro expansion tests use XCTest because `assertMacroExpansion` from
        // SwiftSyntaxMacrosTestSupport reports failures through XCTest.
        .testTarget(
            name: "SwiftAIMacrosTests",
            dependencies: [
                "SwiftAIMacrosImpl",
                .product(name: "SwiftSyntaxMacrosTestSupport", package: "swift-syntax"),
            ],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "AIOpenAICompatibleTests",
            dependencies: ["AIOpenAICompatible", "AITestSupport"],
            resources: [.copy("Fixtures")],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "AIOpenAITests",
            dependencies: ["AIOpenAI", "AITestSupport"],
            resources: [.copy("Fixtures")],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "AIAnthropicTests",
            dependencies: ["AIAnthropic", "AITestSupport"],
            resources: [.copy("Fixtures")],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "AIOpenRouterTests",
            dependencies: ["AIOpenRouter", "AITestSupport", "SwiftAI"],
            resources: [.copy("Fixtures")],
            swiftSettings: sharedSwiftSettings
        ),
        .testTarget(
            name: "AIGoogleTests",
            dependencies: ["AIGoogle", "AITestSupport"],
            resources: [.copy("Fixtures")],
            swiftSettings: sharedSwiftSettings
        ),
    ]
)

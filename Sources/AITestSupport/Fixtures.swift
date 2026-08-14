import Foundation

/// Loads recorded provider payloads from a test bundle.
///
/// Provider tests are driven by fixtures captured from the real APIs and checked in, which is
/// what makes them both realistic and reproducible: the parsing code meets exactly the bytes a
/// provider sends, every time, with no network and no API key.
public enum Fixtures {
    /// Loads a fixture as text.
    ///
    /// - Parameters:
    ///   - name: The filename, including its extension, such as `"chat-stream.sse"`.
    ///   - bundle: The bundle to search. Pass `Bundle.module` from the test target that owns the
    ///     fixture.
    ///   - subdirectory: The folder within the bundle. Defaults to `"Fixtures"`.
    /// - Returns: The file's contents, decoded as UTF-8.
    /// - Throws: ``FixtureError`` when the file is missing or not valid UTF-8.
    public static func text(
        _ name: String,
        bundle: Bundle,
        subdirectory: String = "Fixtures"
    ) throws -> String {
        let data = try data(name, bundle: bundle, subdirectory: subdirectory)
        guard let text = String(data: data, encoding: .utf8) else {
            throw FixtureError.notUTF8(name: name)
        }
        return text
    }

    /// Loads a fixture as raw bytes.
    public static func data(
        _ name: String,
        bundle: Bundle,
        subdirectory: String = "Fixtures"
    ) throws -> Data {
        let components = name.split(separator: ".")
        let fileExtension = components.count > 1 ? String(components.last!) : nil
        let baseName = fileExtension.map { String(name.dropLast($0.count + 1)) } ?? name

        let url = bundle.url(forResource: baseName, withExtension: fileExtension, subdirectory: subdirectory)
            ?? bundle.url(forResource: baseName, withExtension: fileExtension)
        guard let url else {
            throw FixtureError.notFound(name: name, bundle: bundle.bundlePath)
        }
        return try Data(contentsOf: url)
    }
}

/// Failures raised while loading a fixture.
public enum FixtureError: Error, CustomStringConvertible {
    case notFound(name: String, bundle: String)
    case notUTF8(name: String)

    public var description: String {
        switch self {
        case .notFound(let name, let bundle):
            return """
                Fixture '\(name)' was not found in \(bundle). Check that it is listed under \
                `resources:` for the test target.
                """
        case .notUTF8(let name):
            return "Fixture '\(name)' is not valid UTF-8."
        }
    }
}

// MARK: - Stream collection

extension AsyncSequence {
    /// Collects every element of an asynchronous sequence into an array.
    ///
    /// Tests routinely need the whole stream before asserting on it; writing the loop by hand
    /// each time adds noise without adding clarity.
    ///
    /// - Note: Declared without a `Failure` constraint because that associated type is only
    ///   available on macOS 15 and later, while this package supports macOS 13.
    public func collect() async throws -> [Element] {
        var elements: [Element] = []
        for try await element in self {
            elements.append(element)
        }
        return elements
    }
}

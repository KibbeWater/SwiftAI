import Foundation

/// Builds a `multipart/form-data` request body.
///
/// Needed by the audio endpoints, which take file uploads rather than JSON. Kept minimal and
/// dependency-free: fields are appended in order and the body is produced once.
///
/// ```swift
/// var form = MultipartFormData()
/// form.addField(name: "model", value: "whisper-1")
/// form.addFile(name: "file", filename: "audio.mp3", mediaType: "audio/mpeg", data: audio)
/// let request = HTTPRequest(url: url, headers: ["Content-Type": form.contentType], body: form.encoded())
/// ```
public struct MultipartFormData: Sendable {
    /// The boundary separating parts.
    public let boundary: String

    private var body = Data()

    /// Creates a builder.
    ///
    /// - Parameter boundary: The part separator. Injectable so tests can produce byte-identical
    ///   bodies; defaults to a random value as the specification requires.
    public init(boundary: String = "SwiftAIFormBoundary\(UUID().uuidString)") {
        self.boundary = boundary
    }

    /// The value for the `Content-Type` header.
    public var contentType: String { "multipart/form-data; boundary=\(boundary)" }

    /// Appends a plain text field.
    public mutating func addField(name: String, value: String) {
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        body.append(value)
        body.append("\r\n")
    }

    /// Appends a file field.
    ///
    /// - Parameters:
    ///   - name: The form field name.
    ///   - filename: The filename reported to the server. Some providers infer the audio format
    ///     from its extension, so it should carry a plausible one.
    ///   - mediaType: The part's content type.
    ///   - data: The file bytes.
    public mutating func addFile(name: String, filename: String, mediaType: String, data: Data) {
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
        body.append("Content-Type: \(mediaType)\r\n\r\n")
        body.append(data)
        body.append("\r\n")
    }

    /// Produces the complete body, including the closing boundary.
    public func encoded() -> Data {
        var complete = body
        complete.append("--\(boundary)--\r\n")
        return complete
    }
}

extension Data {
    /// Appends the UTF-8 bytes of a string.
    fileprivate mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}

// MARK: - Identifiers

/// Generates identifiers for stream blocks and tool calls.
///
/// Providers that do not supply their own identifiers use these so that the block structure of a
/// stream is always well formed.
public enum IdentifierGenerator {
    /// Returns a short, URL-safe random identifier.
    ///
    /// - Parameter prefix: A prefix that makes the identifier's origin legible in logs.
    public static func generate(prefix: String = "id") -> String {
        // 12 base-36 characters carry roughly 62 bits, which is ample for uniqueness within a
        // single response while staying short enough to read in a log.
        let alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        let suffix = String((0..<12).map { _ in alphabet.randomElement()! })
        return "\(prefix)_\(suffix)"
    }
}

/// Suggests a filename for an audio payload based on its media type.
///
/// Several transcription APIs infer the container format from the uploaded filename rather than
/// the part's content type, so a plausible extension is load-bearing rather than cosmetic.
public func suggestedAudioFilename(for mediaType: String) -> String {
    let normalized = mediaType.lowercased().split(separator: ";").first.map(String.init) ?? mediaType
    let knownExtensions: [String: String] = [
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "mp4",
        "audio/m4a": "m4a",
        "audio/x-m4a": "m4a",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/wave": "wav",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        "audio/aac": "aac",
        "video/mp4": "mp4",
        "video/webm": "webm",
    ]
    let fileExtension = knownExtensions[normalized]
        ?? normalized.split(separator: "/").last.map(String.init)
        ?? "bin"
    return "audio.\(fileExtension)"
}

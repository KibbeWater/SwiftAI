import Foundation

/// One event from a `text/event-stream` response.
public struct ServerSentEvent: Sendable, Hashable {
    /// The `event:` field, naming the event type. Providers such as Anthropic and OpenAI's
    /// Responses API use this to discriminate; others send only `data:`.
    public var event: String?

    /// The accumulated `data:` fields, joined with newlines.
    public var data: String

    /// The `id:` field, used by servers that support resuming a stream.
    public var id: String?

    /// The `retry:` field, in milliseconds.
    public var retry: Int?

    public init(event: String? = nil, data: String, id: String? = nil, retry: Int? = nil) {
        self.event = event
        self.data = data
        self.id = id
        self.retry = retry
    }

    /// Whether the data payload is the `[DONE]` sentinel that OpenAI-style APIs use to close a
    /// stream.
    public var isDoneSentinel: Bool { data == "[DONE]" }
}

/// An incremental parser for `text/event-stream` bodies.
///
/// Feed it whatever chunks the network produces; it buffers across chunk boundaries and emits
/// events only once they are complete. This matters more than it might seem: a single SSE event
/// routinely arrives split across several TCP segments, and a parser that assumes chunk
/// boundaries align with event boundaries fails intermittently and only under load.
///
/// ```swift
/// var parser = EventSourceParser()
/// for try await chunk in response.body {
///     for event in parser.consume(chunk) {
///         handle(event)
///     }
/// }
/// ```
///
/// The implementation follows the WHATWG event stream specification: fields are `name: value`
/// with one optional leading space stripped from the value, `data:` fields accumulate across
/// lines joined by newlines, lines beginning with `:` are comments, a blank line dispatches the
/// buffered event, and `\n`, `\r\n`, and a lone `\r` are all accepted as line terminators.
public struct EventSourceParser: Sendable {
    /// Bytes received but not yet forming a complete line.
    private var buffer: [UInt8] = []

    /// Fields accumulated for the event currently being built.
    private var eventType: String?
    private var dataLines: [String] = []
    private var lastEventID: String?
    private var retry: Int?

    /// Whether the last chunk ended with a carriage return, so that a leading newline in the next
    /// chunk is recognized as the second half of a `\r\n` pair rather than a second line break.
    private var pendingCarriageReturn = false

    public init() {}

    /// Consumes a chunk of the response body and returns any events it completed.
    ///
    /// - Parameter chunk: Raw bytes, at arbitrary boundaries.
    /// - Returns: The events completed by this chunk, in order. Often empty.
    public mutating func consume(_ chunk: Data) -> [ServerSentEvent] {
        var events: [ServerSentEvent] = []
        for byte in chunk {
            if pendingCarriageReturn {
                pendingCarriageReturn = false
                // Swallow the `\n` of a `\r\n` pair; the line was already dispatched by the `\r`.
                if byte == 0x0A { continue }
            }
            switch byte {
            case 0x0A:  // Line feed.
                if let event = processLine() { events.append(event) }
            case 0x0D:  // Carriage return.
                pendingCarriageReturn = true
                if let event = processLine() { events.append(event) }
            default:
                buffer.append(byte)
            }
        }
        return events
    }

    /// Flushes any event still buffered when the stream ends.
    ///
    /// Well-behaved servers terminate the final event with a blank line, but not all do. Calling
    /// this at end of stream avoids silently dropping the last event.
    ///
    /// - Returns: The final event, if one was buffered.
    public mutating func finish() -> ServerSentEvent? {
        if !buffer.isEmpty {
            _ = processLine()
        }
        return dispatchEvent()
    }

    /// Interprets the buffered line and clears the buffer.
    ///
    /// - Returns: An event, when the line was blank and terminated a non-empty event.
    private mutating func processLine() -> ServerSentEvent? {
        let line = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: true)

        // A blank line dispatches whatever has accumulated.
        guard !line.isEmpty else { return dispatchEvent() }

        // A line beginning with a colon is a comment; servers send these as keep-alives.
        guard !line.hasPrefix(":") else { return nil }

        let name: String
        var value: String
        if let colonIndex = line.firstIndex(of: ":") {
            name = String(line[line.startIndex..<colonIndex])
            value = String(line[line.index(after: colonIndex)...])
            // A single leading space after the colon is part of the syntax, not the value.
            if value.hasPrefix(" ") { value.removeFirst() }
        } else {
            // A line with no colon is a field name with an empty value.
            name = line
            value = ""
        }

        switch name {
        case "event": eventType = value
        case "data": dataLines.append(value)
        case "id":
            // The specification requires ignoring an id containing a NULL character.
            if !value.contains("\0") { lastEventID = value }
        case "retry": retry = Int(value)
        default: break  // Unknown fields are ignored.
        }
        return nil
    }

    /// Emits the accumulated event and resets per-event state.
    private mutating func dispatchEvent() -> ServerSentEvent? {
        defer {
            eventType = nil
            dataLines.removeAll(keepingCapacity: true)
        }
        // The specification says an event with no data is not dispatched.
        guard !dataLines.isEmpty else { return nil }
        return ServerSentEvent(
            event: eventType,
            data: dataLines.joined(separator: "\n"),
            id: lastEventID,
            retry: retry
        )
    }
}

// MARK: - Stream adapters

extension AsyncThrowingStream where Element == Data, Failure == any Error {
    /// Re-expresses a stream of raw body chunks as a stream of server-sent events.
    ///
    /// - Parameter stopAtDoneSentinel: Whether to end the stream when a `[DONE]` payload arrives,
    ///   without emitting it. OpenAI-style APIs use that sentinel to mark the end; Anthropic and
    ///   Google do not. Defaults to `true`.
    /// - Returns: A stream of decoded events.
    public func serverSentEvents(stopAtDoneSentinel: Bool = true) -> AsyncThrowingStream<ServerSentEvent, any Error> {
        AsyncThrowingStream<ServerSentEvent, any Error> { continuation in
            let task = Task {
                var parser = EventSourceParser()
                do {
                    for try await chunk in self {
                        for event in parser.consume(chunk) {
                            if stopAtDoneSentinel, event.isDoneSentinel {
                                continuation.finish()
                                return
                            }
                            continuation.yield(event)
                        }
                    }
                    if let final = parser.finish(), !(stopAtDoneSentinel && final.isDoneSentinel) {
                        continuation.yield(final)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

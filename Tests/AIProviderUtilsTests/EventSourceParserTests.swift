import AIProviderSpec
import AITestSupport
import Foundation
import Testing

@testable import AIProviderUtils

@Suite("EventSourceParser")
struct EventSourceParserTests {
    /// Feeds a body to the parser in fixed-size chunks and returns every event produced.
    private func parse(_ text: String, chunkSize: Int) -> [ServerSentEvent] {
        var parser = EventSourceParser()
        var events: [ServerSentEvent] = []
        let data = Data(text.utf8)
        for start in stride(from: 0, to: data.count, by: chunkSize) {
            let chunk = data[start..<min(start + chunkSize, data.count)]
            events.append(contentsOf: parser.consume(chunk))
        }
        if let final = parser.finish() { events.append(final) }
        return events
    }

    @Test("Parses a simple data-only stream")
    func parsesDataOnlyStream() {
        let events = parse("data: one\n\ndata: two\n\n", chunkSize: 1024)
        #expect(events.map(\.data) == ["one", "two"])
    }

    @Test("Parses named events")
    func parsesNamedEvents() {
        let body = """
            event: message_start
            data: {"type":"message_start"}

            event: content_block_delta
            data: {"type":"content_block_delta"}


            """
        let events = parse(body, chunkSize: 1024)
        #expect(events.map(\.event) == ["message_start", "content_block_delta"])
        #expect(events[0].data == #"{"type":"message_start"}"#)
    }

    @Test("Joins multi-line data fields with newlines")
    func joinsMultiLineData() {
        let events = parse("data: first\ndata: second\ndata: third\n\n", chunkSize: 1024)
        #expect(events.count == 1)
        #expect(events[0].data == "first\nsecond\nthird")
    }

    @Test("Strips exactly one leading space after the colon")
    func stripsOneLeadingSpace() {
        let events = parse("data:  two spaces\n\n", chunkSize: 1024)
        #expect(events[0].data == " two spaces")
    }

    @Test("Handles a field with no value")
    func handlesFieldWithoutValue() {
        let events = parse("data\n\n", chunkSize: 1024)
        #expect(events.count == 1)
        #expect(events[0].data == "")
    }

    @Test("Ignores comment lines used as keep-alives")
    func ignoresComments() {
        let events = parse(": keep-alive\n\ndata: real\n\n", chunkSize: 1024)
        #expect(events.map(\.data) == ["real"])
    }

    @Test("Does not dispatch an event with no data")
    func doesNotDispatchEmptyEvents() {
        // A lone `event:` field with no data must not produce an event.
        let events = parse("event: ping\n\ndata: real\n\n", chunkSize: 1024)
        #expect(events.count == 1)
        #expect(events[0].data == "real")
    }

    @Test("Reads id and retry fields")
    func readsIDAndRetry() {
        let events = parse("id: 42\nretry: 3000\ndata: hello\n\n", chunkSize: 1024)
        #expect(events[0].id == "42")
        #expect(events[0].retry == 3000)
    }

    @Test("An id persists across events until replaced, per the specification")
    func idPersistsAcrossEvents() {
        let events = parse("id: 1\ndata: a\n\ndata: b\n\nid: 2\ndata: c\n\n", chunkSize: 1024)
        #expect(events.map(\.id) == ["1", "1", "2"])
    }

    @Test("Ignores an id containing a null character")
    func ignoresIDWithNullCharacter() {
        let events = parse("id: bad\u{0}id\ndata: hello\n\n", chunkSize: 1024)
        #expect(events[0].id == nil)
    }

    @Test("Accepts every line terminator", arguments: ["\n", "\r\n", "\r"])
    func acceptsAllLineTerminators(terminator: String) {
        let body = "data: one\(terminator)\(terminator)data: two\(terminator)\(terminator)"
        let events = parse(body, chunkSize: 1024)
        #expect(events.map(\.data) == ["one", "two"])
    }

    /// The condition that breaks naive parsers: events straddling chunk boundaries.
    @Test("Produces identical events at every chunk size", arguments: [1, 2, 3, 5, 7, 13, 64, 4096])
    func chunkSizeDoesNotAffectResult(chunkSize: Int) {
        let body = """
            event: start
            data: {"id":"abc","value":1}

            : keep-alive

            event: delta
            data: {"text":"héllo 😀"}
            data: {"continued":true}

            id: 7
            data: [DONE]


            """
        let events = parse(body, chunkSize: chunkSize)
        #expect(events.count == 3)
        #expect(events[0].event == "start")
        #expect(events[1].data == #"{"text":"héllo 😀"}"# + "\n" + #"{"continued":true}"#)
        #expect(events[2].isDoneSentinel)
    }

    @Test("A multi-byte character split across chunks survives")
    func splitMultiByteCharacterSurvives() {
        // The emoji is four UTF-8 bytes; a chunk size of one splits it in the middle.
        let events = parse("data: 😀\n\n", chunkSize: 1)
        #expect(events[0].data == "😀")
    }

    @Test("Flushes a trailing event that lacks a blank line")
    func flushesUnterminatedFinalEvent() {
        // Not all servers terminate the last event properly; dropping it silently would be worse.
        let events = parse("data: complete\n\ndata: dangling", chunkSize: 1024)
        #expect(events.map(\.data) == ["complete", "dangling"])
    }

    @Test("Recognizes the done sentinel")
    func recognizesDoneSentinel() {
        #expect(ServerSentEvent(data: "[DONE]").isDoneSentinel)
        #expect(ServerSentEvent(data: "[done]").isDoneSentinel == false)
    }

    // MARK: - Stream adapter

    @Test("The stream adapter stops at the done sentinel without emitting it")
    func streamAdapterStopsAtSentinel() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents("data: a\n\ndata: [DONE]\n\ndata: never\n\n", chunkSize: 5)
        )
        let response = try await transport.stream(HTTPRequest(url: URL(string: "https://example.com")!))
        let events = try await response.body.serverSentEvents().collect()
        #expect(events.map(\.data) == ["a"])
    }

    @Test("The stream adapter can pass the done sentinel through")
    func streamAdapterCanKeepSentinel() async throws {
        let transport = MockHTTPTransport(
            exchange: .serverSentEvents("data: a\n\ndata: [DONE]\n\n", chunkSize: 5)
        )
        let response = try await transport.stream(HTTPRequest(url: URL(string: "https://example.com")!))
        let events = try await response.body.serverSentEvents(stopAtDoneSentinel: false).collect()
        #expect(events.map(\.data) == ["a", "[DONE]"])
    }

    @Test("The stream adapter propagates a mid-stream failure")
    func streamAdapterPropagatesFailure() async throws {
        struct Boom: Error {}
        let transport = MockHTTPTransport(
            exchanges: [
                MockHTTPTransport.Exchange(
                    head: HTTPResponseHead(statusCode: 200),
                    chunks: [Data("data: a\n\n".utf8)],
                    error: Boom()
                )
            ]
        )
        let response = try await transport.stream(HTTPRequest(url: URL(string: "https://example.com")!))

        var received: [ServerSentEvent] = []
        await #expect(throws: Boom.self) {
            for try await event in response.body.serverSentEvents() {
                received.append(event)
            }
        }
        // Events delivered before the failure are still surfaced.
        #expect(received.map(\.data) == ["a"])
    }
}

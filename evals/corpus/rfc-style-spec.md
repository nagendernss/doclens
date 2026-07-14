# Meridian Sync Protocol (MSP), Version 4

## Status of This Memo

This document specifies the Meridian Sync Protocol (MSP), an application-layer protocol for synchronizing small binary objects between a client and a coordinating server over a reliable transport. This is a fictional specification written for demonstration and evaluation purposes; it does not describe a deployed or standardized protocol, and it carries no IETF status.

## Abstract

Meridian Sync Protocol lets a client push and pull versioned objects to and from a remote store while detecting conflicts through a monotonic revision counter carried in every message. MSP Version 4 replaces the fixed-length header used in Version 3 with a 16-byte variable-flag header, adds a mandatory checksum field, and introduces a formal retry and backoff schedule for lossy links. Implementations conforming to this document listen on TCP port 7847 by default and negotiate protocol capabilities during the handshake described in Section 5.

## 1. Introduction

Devices that synchronize state across an unreliable network need an agreed way to detect when two copies of an object have diverged. Earlier point-to-point synchronization schemes solved this informally, with each vendor inventing its own conflict resolution rules. Meridian Sync Protocol standardizes the wire format, the handshake, and the retry behavior so that independent implementations can interoperate without sharing source code. The design favors small messages and low round-trip counts because MSP targets constrained links such as satellite backhaul and congested cellular networks, not high-bandwidth data-center transfers.

## 2. Terminology

The following terms recur throughout this specification. A "session" is a single authenticated connection between a client and a server, bounded by a handshake and a close message. An "object" is a named, versioned blob of at most 64 KiB (65536 bytes) after decompression. A "revision" is a 64-bit monotonically increasing integer attached to every object write; the server rejects any write whose revision is not strictly greater than the revision it currently holds. A "peer" refers to either endpoint of a session, since MSP messages use the same envelope in both directions.

## 3. Protocol Overview

Every MSP exchange begins with a three-message handshake, proceeds through zero or more object operations, and ends with an explicit close or an idle timeout. Clients initiate sessions; servers never initiate a session on their own, though they may push unsolicited change notifications once a session is established. All multi-byte integers in this specification are encoded big-endian, matching common network byte order conventions. A session carries at most 32 concurrent in-flight requests per client; the 33rd request on a saturated session receives error code 0x07 (TOO_MANY_INFLIGHT) until an earlier request completes.

## 4. Message Format

Each MSP message begins with a fixed 16-byte header: a 2-byte magic value (0x4D53, the ASCII characters "MS"), a 1-byte version field (4 for this revision), a 1-byte opcode, a 4-byte payload length, a 4-byte CRC32C checksum of the payload, and a 4-byte reserved field that must be zero on send and ignored on receive. The header is immediately followed by the payload, whose length matches the header's length field exactly; a mismatch causes the receiver to close the session with error code 0x02 (FRAMING_ERROR). Payloads larger than 65536 bytes are rejected outright with error code 0x03 (OBJECT_TOO_LARGE) before any bytes are read from the socket, protecting the receiver from unbounded buffering.

## 5. Handshake Sequence

The client opens a TCP connection to port 7847 and sends a HELLO message containing its supported protocol versions and a random 8-byte nonce. The server replies with an ACCEPT message naming the negotiated version (the highest version both sides support) and echoing the nonce combined with its own 8-byte nonce. The client completes the handshake with a CONFIRM message carrying an HMAC computed over both nonces using the pre-shared session key. If the HMAC does not validate, the server closes the connection immediately without sending an error message, to avoid leaking timing information to an attacker. A successful handshake must complete within 8 seconds of the initial TCP connect; exceeding this handshake timeout causes the server to drop the connection silently.

## 6. Error Codes

MSP defines nine standard error codes carried in the ERROR message opcode. Code 0x01 is UNKNOWN_OPCODE, sent when a peer receives an opcode it does not recognize. Code 0x02 is FRAMING_ERROR, described above in Section 4. Code 0x03 is OBJECT_TOO_LARGE, also described in Section 4. Code 0x04 is STALE_REVISION, returned when a write's revision number is not greater than the stored revision. Code 0x05 is AUTH_FAILED, returned only after the handshake succeeds, for later per-object authorization failures. Code 0x06 is NOT_FOUND, returned when a client requests an object name the server has never seen. Code 0x07 is TOO_MANY_INFLIGHT, described in Section 3. Code 0x08 is SERVER_BUSY, a hint that the client should back off and retry later. Code 0x09 is PROTOCOL_MISMATCH, sent when no common version exists during handshake negotiation.

## 7. Retry and Backoff

Clients that receive SERVER_BUSY or that experience a connection reset during an object operation must not reconnect immediately. The mandated backoff schedule starts at 250 milliseconds and doubles after each failed attempt, capped at a maximum delay of 8 seconds, with a maximum of 5 retries before the client surfaces a hard failure to its caller. Each retry delay should be jittered by up to 20 percent to avoid synchronized reconnection storms when many clients lose connectivity to the same server simultaneously. Servers may also send a RETRY_AFTER hint, expressed in milliseconds, inside a SERVER_BUSY error payload, which takes precedence over the client's own backoff calculation whenever it is present.

## 8. Security Considerations

MSP relies entirely on a pre-shared session key established out of band; this specification does not define a key exchange mechanism, and implementers should not treat the handshake nonce exchange as a substitute for one. All object payloads are transmitted without encryption at the MSP layer, so deployments carrying sensitive data must tunnel MSP inside TLS or an equivalent transport-layer encryption scheme. The CRC32C checksum in the message header protects against accidental corruption only; it is not a cryptographic integrity check and must never be relied on to detect deliberate tampering by an active attacker on the network path.

## 9. Port and Registry Notes

This specification assigns TCP port 7847 for MSP traffic in test and demonstration deployments described by this document; no IANA registration has been requested since MSP is a fictional protocol created for the doclens evaluation corpus. Implementations that need to run multiple MSP servers on one host should offset the port by session group, a convention documented in local deployment notes rather than in this specification. Keepalive messages, sent using opcode 0x0A, should go out every 45 seconds of idle time on an open session to prevent intermediate stateful firewalls from silently dropping the mapping.

## 10. Appendix: Worked Example

Picture a thermostat controller that has just booted and holds a locally cached copy of object "config/thermostat-7" at revision 12. It connects to a server on port 7847, completes the handshake in roughly 400 milliseconds, and issues a GET for that object name. The server responds with revision 14 and a fresh payload, indicating that two writes happened elsewhere since the client last synchronized. The client applies both changes locally, updates its cached revision to 14, and closes the session cleanly with a BYE message rather than waiting for the 45-second keepalive window to expire.

## 11. Conformance Levels

A fully conforming MSP implementation MUST support every opcode listed in Sections 5 through 7, spanning the range 0x00 through 0x0A inclusive. Compression is optional: implementations MAY advertise support for opcode 0x0B (COMPRESSED_PUT) during the handshake, and peers that do not advertise it must never receive a compressed payload. This specification also defines a constrained profile named MSP-Lite for battery-powered sensors: MSP-Lite peers cap object size at 4 KiB rather than the standard 64 KiB ceiling, and they disable both the PROGRAM extension and the 45-second keepalive, relying instead on the transport's own idle timeout. A server advertises MSP-Lite support in its ACCEPT message by setting the reserved header field to 0x01 instead of 0x00.

## 12. Change Log

Version 4 of this specification, described throughout the document above, differs from Version 3 in three ways. First, the fixed header grew from 12 bytes to 16 bytes to accommodate the new CRC32C checksum field. Second, error code 0x09 (PROTOCOL_MISMATCH) was introduced; Version 3 servers instead closed the connection silently on a version mismatch, which made debugging difficult. Third, the retry and backoff schedule in Section 7 became mandatory rather than a recommendation. Version 2 of MSP was formally deprecated in January 2025, and conforming servers built against this document must refuse any HELLO message that proposes Version 2 or lower by responding with error code 0x09.

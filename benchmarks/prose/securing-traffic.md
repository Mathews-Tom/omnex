# Securing traffic with certificates

A server proves its identity by presenting a signed certificate during the opening handshake. The client validates the certificate chain against a trusted authority, then both sides negotiate a session key and switch to encrypted records for the rest of the connection.

## Cipher selection

Operators pin a modern cipher suite and disable legacy protocol versions, so the negotiated channel resists downgrade attempts. Forward secrecy ensures a leaked long-term key cannot decrypt past sessions.

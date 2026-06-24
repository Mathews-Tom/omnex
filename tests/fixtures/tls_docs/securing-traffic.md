# Securing traffic with certificates

Obtain a certificate from a trusted authority and rotate it before expiry. The ingress controller presents this certificate to clients during the handshake.

## Renewal

Automate renewal with a controller that watches expiry dates so certificates never lapse.

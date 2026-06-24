# Ingress

The ingress controller routes external traffic to backend services. To configure TLS termination, see [Securing traffic with certificates](securing-traffic.md).

## TLS secrets

Store the TLS certificate and private key in a Kubernetes secret, then reference the secret from the ingress resource.[^tls]

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ingress-tls
type: kubernetes.io/tls
data:
  tls.crt: BASE64_ENCODED_CERT
  tls.key: BASE64_ENCODED_KEY
```

## Verification

Confirm the ingress serves the certificate stored by the [TLS secrets](#tls-secrets) step.

[^tls]: A TLS secret uses the kubernetes.io/tls type.

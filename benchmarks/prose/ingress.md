# Ingress controller

The ingress controller routes external HTTP traffic to backend services running inside the cluster. It is the single entry point that maps each incoming request to a service.

## TLS secrets

To configure TLS for the ingress controller, store the certificate and private key in a Kubernetes secret and reference that secret from the ingress resource. The controller terminates TLS at the edge and forwards plaintext to the backend service.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ingress-tls-cert
type: kubernetes.io/tls
data:
  tls.crt: BASE64_ENCODED_CERTIFICATE
  tls.key: BASE64_ENCODED_PRIVATE_KEY
```

## Routing rules

Define host and path rules on the ingress resource so the controller forwards each request to the correct backend service. Rules are evaluated most-specific first.

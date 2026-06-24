# Service discovery

Workloads find one another through the cluster DNS service rather than hard-coded addresses. Each service receives a stable virtual name, and the resolver returns the current set of healthy backends behind it.

## Headless services

A headless service skips the virtual address and returns the pod addresses directly. Clients that need to talk to a specific replica, such as a stateful database member, rely on this mode.

## Endpoint slices

The control plane tracks reachable backends in endpoint slices and updates them as pods come and go, so the resolver never points a caller at a backend that has already been removed.

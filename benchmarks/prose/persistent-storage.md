# Persistent storage

Stateful workloads request durable disks through persistent volume claims. The scheduler binds each claim to a volume that survives pod restarts and rescheduling, so data outlives the process that wrote it.

## Storage classes

A storage class names a provisioner and its parameters, letting a claim ask for fast solid-state capacity or cheaper spinning disk without naming a concrete volume.

## Snapshots

Volume snapshots capture a point-in-time copy that operators restore into a fresh claim, which is the usual path for cloning an environment or recovering from a bad migration.

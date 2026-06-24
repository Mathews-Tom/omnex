# Service discovery

Service discovery lets workloads find one another without hard-coded network
addresses. A registry tracks healthy endpoints and hands them to callers on
demand, so a client always reaches a live instance even as the fleet scales up
and down throughout the day.

## DNS records

The registry publishes address records that resolvers cache for a short window.
A resolver asks the registry for a name, receives a set of addresses, and rotates
through them on subsequent lookups. Short cache windows keep the view fresh while
long windows reduce lookup chatter, so operators tune the window to match how
quickly the fleet changes.

## Round-robin pools

A pool groups equivalent endpoints behind one name. Round-robin selection spreads
calls evenly across the pool, while weighted selection sends more calls to larger
endpoints. Health probes remove a failing endpoint from the pool until it
recovers, so callers never wait on an endpoint that has already gone dark.

## Failover regions

When a whole region drops out, the registry promotes a standby region and
redirects new lookups to it. Operators rehearse this promotion regularly so the
switch is dull and predictable rather than a surprise during an outage.

## Observability

Every lookup, promotion, and health transition is logged with a timestamp and the
identity of the endpoint involved. Dashboards summarize lookup latency and pool
occupancy so operators can spot a slow drift long before it becomes an incident.

## Registration lifecycle

A workload registers itself on startup and sends a steady heartbeat while it runs.
If the heartbeat stops, the registry marks the entry stale and stops handing it to
callers, then removes it once a grace period passes. Graceful shutdown deregisters
the entry immediately so callers stop reaching for an endpoint that has chosen to
leave, which keeps the steady-state lookup path quiet and avoids a burst of
retries whenever a single instance rolls during a routine deployment window.

## Caching strategy

Callers cache the address set they receive and reuse it until the entry expires.
A small jitter on each expiry spreads renewal across callers so the registry never
sees a synchronized stampede of lookups at the top of every minute. Negative
results are cached briefly too, so a momentary miss does not turn into a tight loop
of repeated lookups against a registry that has nothing new to say yet.

## Capacity planning

Operators size the registry for the peak lookup rate plus headroom for a failover
event, when a surge of callers refresh their caches at once. They track the ratio
of reads to writes, the size of the largest pool, and the tail latency of a single
lookup, then add replicas before any of those measures drift toward a threshold.
Rehearsed load tests replay a recorded peak so the headroom estimate rests on
measured behavior rather than a hopeful guess about how the fleet will behave.

## Migration notes

Moving from a static address list to dynamic discovery happens in stages. First the
old list and the registry run side by side and their answers are compared offline.
Once the answers agree for long enough, callers switch to the registry as the
source of truth while the static list lingers as a read-only fallback. Only after a
full quiet period does the static list retire, so any surprise in the new path is
caught while the old path is still a safe option to fall back upon.

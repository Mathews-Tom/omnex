# Observability

Operators understand a running cluster through three signals: metrics, logs, and traces. Each answers a different question, and a healthy pipeline keeps them correlated by a shared request identifier.

## Metrics

Counters and gauges scraped on an interval drive dashboards and alerts. They are cheap to store and fast to query, but they sample aggregate behavior rather than individual requests.

## Traces

A trace follows one request across services, recording the time spent in each hop. Traces are the tool of choice when a request is slow but every individual service reports healthy.

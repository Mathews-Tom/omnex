# Local Metrics

This document explains what omnex can record locally, how the savings numbers are derived, and where the hard privacy and surface boundaries are.

See also: [system-design.md](system-design.md).

## The boundary

omnex ships with local usage metrics off by default.

That boundary is strict:

- No telemetry.
- No network calls.
- Metrics controls are CLI-only.
- The MCP server exposes no metrics tools, so an agent can route retrieval through omnex but cannot read, change, or delete the operator's metrics state.
- When metrics are disabled, no usage ledger file is created and nothing is recorded.

The recorder is a non-essential side channel. Retrieval does not depend on the metrics ledger.

## Enablement

There are two separate opt-ins:

1. Usage metrics
2. Detailed traces

### Usage metrics

Turn usage metrics on or off persistently with the CLI:

```bash
omnex metrics enable
omnex metrics enable --off
```

The persisted setting lives in `~/.omnex/settings.json` unless `OMNEX_HOME` relocates the home directory.

You can also override the setting for the current process with `OMNEX_USAGE_METRICS`:

```bash
OMNEX_USAGE_METRICS=on omnex query docs "What changed?" --budget 1200
OMNEX_USAGE_METRICS=off omnex query docs "What changed?" --budget 1200
```

The override is case-insensitive. These values mean “on”:

| Value | Effect |
| --- | --- |
| `1` | enable |
| `on` | enable |
| `true` | enable |
| `yes` | enable |

Anything else is treated as off.

### Detailed traces

Traces are a second, separate opt-in.

Turn tracing on or off persistently with the CLI:

```bash
omnex metrics trace
omnex metrics trace --off
```

Or override it for the current process:

```bash
OMNEX_USAGE_TRACE=on omnex query docs "What changed?" --budget 1200
OMNEX_USAGE_TRACE=off omnex query docs "What changed?" --budget 1200
```

`OMNEX_USAGE_TRACE` uses the same case-insensitive truthy rules as `OMNEX_USAGE_METRICS`.

Trace does not enable metrics by itself. If usage metrics are off, nothing is recorded even if trace is set to on.

## Where it lives

omnex stores local metrics state under the omnex home directory:

| Artifact | Default path |
| --- | --- |
| Settings | `~/.omnex/settings.json` |
| SQLite ledger | `~/.omnex/usage.sqlite` |

Set `OMNEX_HOME` to relocate that home directory:

```bash
OMNEX_HOME=/tmp/omnex-home omnex metrics enable
```

The SQLite ledger schema version is `user_version = 2`.

A missing settings file means the default-off state. A missing ledger file means no events have been recorded.

## What is stored

The default event row is `UsageEvent`. It stores anonymous counters only.

| Field | Meaning |
| --- | --- |
| `occurred_at` | ISO-8601 UTC timestamp for the recorded run |
| `tool` | `query` or `index` |
| `surface` | `cli` or `mcp` |
| `category` | Coarse content-free label: a render style for `query`, or `index` for `index` |
| `returned_tokens` | Token count omnex returned, copied verbatim from the receipt |
| `baseline_tokens` | Full-dump baseline from the receipt, copied verbatim |
| `file_count` | Number of files in the run |
| `repo_id` | Stable repo-local random ID, never a path |

A few important consequences follow from that schema:

- `query` events can contribute savings because they carry receipt token counts.
- `index` events do not contribute savings. They record `returned_tokens = 0`, `baseline_tokens = 0`, and `category = "index"`.
- The recorder does not recompute token counts from files and does not call a model.

### What is never stored

The ledger never stores:

- query text
- corpus paths
- symbol names
- unit text
- rendered output
- handles

The repo path also stays out of the ledger. `repo_id` is an anonymous random ID associated with the local repo through settings, not through a stored path in the usage ledger.

## Traces

When both metrics and tracing are enabled, omnex also writes `UsageTrace` rows.

| Field | Meaning |
| --- | --- |
| `occurred_at` | ISO-8601 UTC timestamp |
| `tool` | Recorded operation, currently `query` |
| `surface` | `cli` or `mcp` |
| `repo_id` | Same anonymous repo-local random ID used by events |
| `tier` | Comma-joined receipt tiers, such as `T0` |
| `determinism_class` | Receipt determinism class |
| `recall_basis` | Receipt recall basis |
| `reference_closure_complete` | Whether the receipt reported a complete emitted reference closure |

Traces still do not store the question, a path, unit text, or output.

## Savings math

All savings figures are derived from the receipt fields already stored in `UsageEvent`:

- `returned_tokens`
- `baseline_tokens`

The savings code never re-reads files and never calls a model.

Only `query` events contribute. `index` events carry no savings.

### 1. Full-file paste

This is the headline savings number.

Per query event:

```text
full_file_paste_saved = baseline_tokens - returned_tokens
```

In the aggregate implementation, each event contributes:

```text
max(0, baseline_tokens - returned_tokens)
```

That is the realized saving versus pasting the queried file or corpus in full.

### 2. Targeted read

This is the conservative headline.

omnex models a targeted read as a labeled assumption:

```text
TARGETED_READ_MULTIPLE = 3
```

Per query event, the targeted-read baseline is:

```text
targeted_read_baseline = min(baseline_tokens, returned_tokens * 3)
```

The corresponding conservative saving is:

```text
targeted_read_saved = targeted_read_baseline - returned_tokens
```

In the aggregate implementation, each event contributes:

```text
max(0, targeted_read_baseline - returned_tokens)
```

This is intentionally the hardest baseline for omnex to beat. It is modeled, labeled, and capped by the full baseline.

### 3. Whole-corpus

This is a demoted upper bound, not a headline savings number:

```text
whole_corpus_tokens = sum(baseline_tokens)
```

It answers “what would it cost to dump every queried corpus in full?” not “what did a user realistically avoid on this one read?”

### Worked example

Suppose the ledger has two `query` events:

| Event | `returned_tokens` | `baseline_tokens` |
| --- | ---: | ---: |
| A | 200 | 1000 |
| B | 150 | 400 |

Then the figures are:

```text
Full-file paste saved
= max(0, 1000 - 200) + max(0, 400 - 150)
= 800 + 250
= 1050

Targeted-read baseline
= min(1000, 200 * 3) + min(400, 150 * 3)
= 600 + 400
= 1000

Targeted-read saved
= max(0, 600 - 200) + max(0, 400 - 150)
= 400 + 250
= 650

Whole-corpus tokens
= 1000 + 400
= 1400
```

The corresponding percentages are:

```text
Full-file paste percent = 1050 / 1400 = 75.0%
Targeted-read percent   = 650 / 1000 = 65.0%
```

That matches the implementation:

- full-file paste uses summed `baseline_tokens` as the denominator
- targeted read uses the summed targeted-read baseline as the denominator
- whole-corpus is reported as context, not as the headline

## Commands

### `omnex metrics summary`

`metrics summary` reports:

- whether usage metrics are on or off
- whether trace is on or off
- total recorded events
- a labeled savings block
- a CLI-vs-MCP surface split

Examples:

```bash
omnex metrics summary
omnex metrics summary --format json
```

The text rendering labels the savings block explicitly:

- Full-file paste: headline
- Targeted read: conservative, with the `3x` modeling assumption called out
- Whole-corpus: upper bound, not a realized saving

The JSON form includes the enable state, trace state, event total, `targeted_read_multiple`, overall savings, and a `by_surface` breakdown.

### `omnex metrics export`

`metrics export` dumps every recorded event as JSON:

```bash
omnex metrics export
```

The payload is an `events` array of serialized `UsageEvent` rows. It exports anonymous counters only.

### `omnex metrics delete`

`metrics delete` removes the local ledger file:

```bash
omnex metrics delete
omnex metrics delete --yes
```

`--yes` skips the confirmation prompt.

Deleting the ledger does not disable metrics. The persisted enable setting is kept.

## Failure behavior

Metrics recording must never break a successful retrieval or index operation.

If recording fails because of something like a bad home directory, a locked or corrupt ledger, or malformed settings, omnex still returns the retrieval result and prints the failure loudly to standard error:

```text
omnex: usage metrics not recorded: ...
```

That failure is never silently swallowed, and it never causes omnex to discard an otherwise successful retrieval.
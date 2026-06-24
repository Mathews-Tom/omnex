# Spec family pre-registration

This note fixes the spec family's claims, query set, and labeling procedure ahead of the numbers, so the headline ("omnex T1 tokens are at most chunk-and-embed tokens at equal recall") cannot be read as self-graded after the fact. It is the integrity record for the benchmark; the numbers live only in the checked-in artifact at `benchmarks/results/specs.json`, regenerated from a clean run, never hand-edited.

## Corpus

One OpenAPI document, `commerce_api.json` (938 tokens in the whitespace `count_tokens` ledger). Each operation uses a distinct action verb ("Create a payment", "Dispatch a shipment", "Enroll a subscriber", and unrelated distractor operations), so a keyword query lexically seeds only its target operation. That isolates the claim under test: the structurally-required schemas are reached by the `$ref` closure, not by happening to share vocabulary with the query.

## Query set (count: 3)

| id | query | gold closure size |
| --- | --- | ---: |
| `create_payment` | `create payment` | 6 |
| `dispatch_shipment` | `dispatch shipment parcel` | 8 |
| `enroll_subscriber` | `enroll subscriber recurring plan` | 6 |

The query count is fixed at three and is pre-registered here before any number is reported. The set is small and labeled by construction (see below); it is not a sample from a larger pool, so there is no selection-after-results step.

## Query selection criterion

Queries are moat-case: each gold closure contains schemas that are structurally required but semantically distant from the query (for example `Money`, `Address`, `Dimensions`, `Weight` for a payment or shipment query). This is exactly where the design (system-design section 9) claims the structural advantage: a semantic top-k misses distant members and must raise k to recover them, dragging tokens, while omnex closes the `$ref` graph deterministically.

Coherent-closure queries -- where every closure member is semantically near the query, so chunk-and-embed retrieves them cheaply too -- are out of scope for the T1 token claim and are the honest boundary of the moat, not a counterexample hidden from the report.

## Labeling procedure

For each query the gold set is the seed operation plus the transitive closure of its request and response schemas over `$ref` edges, read by hand from the spec. Each gold unit is identified by a unique **marker**: the distinctive description string present verbatim in the corpus and in that unit's definition only. Recall is graded by marker presence in the retrieved text, applied identically to omnex and to every baseline, so no path is graded on a different rule.

Markers are pre-listed in `tasks.json`. A duplicate marker within a task fails the loader, since it would make recall un-gradeable.

## Equal-recall rule

Every reported token figure is `tokens_at_recall` at recall 1.0 (the full closure). A path that never reaches recall 1.0 reports no token figure (`null`), and no delta is drawn against it. Recall is therefore held equal in every comparison by construction.

## Baselines

- **Upper bound (demoted):** full-document dump. Reaches recall 1.0 trivially by returning the whole corpus; it bounds token waste, it is not the thing beaten.
- **Headline:** chunk-and-embed, pinned to 256-token `cl100k_base` chunks with 32-token overlap and the `BAAI/bge-small-en-v1.5` embedding via fastembed, no reranker. The pinned configuration is recorded in the artifact.
- **Deterministic cross-check:** a TF-IDF cosine embedder, byte-exact and offline, used in CI and to prove the win does not depend on a specific embedding model.

## Win criterion

On every labeled task, omnex T1 `tokens_at_recall(1.0)` is at most chunk-and-embed `tokens_at_recall(1.0)`. The artifact records this per task and as `totals.omnex_at_or_below_headline_at_equal_recall`.

## Reproduce

```text
# Headline (pinned strong embedding model; requires the bench extra and a one-time model download):
uv run --extra bench omnex-bench run --family specs --out benchmarks/results --embedder bge-small

# Deterministic, offline cross-check (byte-exact):
uv run omnex-bench run --family specs --embedder tfidf
```

omnex T1 and the TF-IDF embedder are byte-exact; the bge-small headline is `pinned_reproducible` (reproducible only with the pinned model, tokenizer, runtime, and architecture), which the artifact's `determinism` field records. Latency is environment-dependent and is excluded from the determinism guarantee.

# Receipts

Receipts are omnex's auditable record for a retrieval run. They travel with the returned `ContextBundle`, explain what omnex did, and make token savings and determinism claims inspectable after the fact. For the broader architecture and stateless execution model, see [system-overview.md](system-overview.md) and [system-design.md](system-design.md).

Unlike archex-style context receipts, omnex receipts do not include freshness or index-revision fields. omnex is stateless: each `index` or `query` call builds its in-memory structures for that call and discards them when the call ends.

## What a receipt is

In `src/omnex/kernel/receipt.py`, `Receipt` is a frozen dataclass that serves as the audit trail for one retrieval.

A receipt answers a few concrete questions:

- How many tokens did omnex emit?
- What was the full-dump upper bound for the same corpus?
- Which tiers actually ran?
- Did the run use a model or extraction lane, and if so, which version?
- What determinism guarantee may this run claim?
- Did omnex fully emit a computed reference closure?
- Did recall rest on lexical retrieval alone, or on lexical plus the vector lane?
- If the vector lane ran, which embedding stack must be matched to replay it?

The type is frozen so it cannot be mutated after construction. On the byte-exact tiers, identical inputs yield identical outputs: the same corpus, config, and query produce a byte-identical bundle and a byte-identical receipt. Today that byte-exact guarantee applies to T0 and T1. T2 is weaker by design, and T3 is a roadmap class rather than a shipped path.

## Receipt fields

The table below follows the meanings documented in `Receipt` and the related kernel configuration types.

| Field | Type | Meaning |
| --- | --- | --- |
| `returned_tokens` | `int` | The emitted token total for the returned bundle. |
| `baseline_tokens` | `int` | The full-dump upper bound used as the comparison baseline. In the modality-blind kernel it starts as the sum of indexed unit text; `query_sources` refines it to the true whole-document dump. |
| `tiers_run` | `tuple[str, ...]` | The tiers exercised by the run. T1 reports `("T0", "T1")`; T2 reports `("T0", "T2")`; the T0 floor reports `("T0",)`. |
| `model_used` | `bool` | Whether the run used an opt-in model-backed lane outside the byte-exact floor. In today's shipped kernel, that means the T2 vector lane. |
| `model_version` | `str \| None` | The model version associated with model use. It is `None` on model-free runs. |
| `extraction_used` | `bool` | Whether adapter-local model extraction ran. This records OCR or similar extraction outside the byte-exact floor. In today's shipped surfaces it is `False`; T3 extraction is roadmap-only. |
| `determinism_class` | `"byte_exact" \| "pinned_reproducible" \| "model_versioned"` | The reproducibility guarantee the run may claim. |
| `reference_closure_complete` | `bool` | `True` only when a tier computed a reference closure and every unit in that closure was emitted in full. This is an exact set-membership fact, not a score threshold. It is `False` on tiers that compute no closure and on an empty closure. |
| `recall_basis` | `"lexical" \| "lexical_plus_vector"` | What recall rested on: the lexical lane alone, or the lexical lane plus the opt-in vector lane. |
| `embedding_provenance` | `EmbeddingProvenance \| None` | Set only on a pinned-reproducible T2 run. Records the embedding `model`, `tokenizer`, `runtime`, and CPU `architecture` required for replay. It is `None` on the byte-exact tiers. |
| `recall_limitations` | `tuple[str, ...]` in Python; `list[str]` in JSON surfaces | A derived property implied by `recall_basis`. It contains two plain-language recall caveats for lexical-only runs, and is empty when the vector lane contributed semantic recall. |

JSON surfaces do not hand-roll this schema. `src/omnex/_surface.py` uses `receipt_dict(receipt)`, which calls `dataclasses.asdict(receipt)` and then appends the derived `recall_limitations` field.

## Determinism classes

`src/omnex/kernel/config.py` defines the determinism classes, and `src/omnex/kernel/kernel.py` maps the shipped tiers onto them.

| Determinism class | Tiers | What it means |
| --- | --- | --- |
| `byte_exact` | T0, T1 | Same corpus, config, and query produce a byte-identical bundle and receipt. No model is loaded. |
| `pinned_reproducible` | T2 | The run used the opt-in vector lane. Replay depends on matching the stamped embedding `model`, `tokenizer`, `runtime`, and CPU `architecture`, so this class is intentionally weaker than `byte_exact`. |
| `model_versioned` | T3 | Reserved for model extraction paths such as OCR, captioning, or transcription. T3 is roadmap-only today, not a shipped surface path. |

A few consequences fall straight out of the source:

- T0 and T1 may claim `byte_exact`.
- T2 may not claim `byte_exact`, even though it still runs on top of the T0 lexical floor.
- `embedding_provenance` exists specifically so a T2 receipt cannot be mistaken for the byte-exact floor.

## `recall_basis` and `recall_limitations`

`recall_basis` says what kind of recall the run actually depended on:

- `lexical`: recall rested on the lexical FTS/BM25F lane alone.
- `lexical_plus_vector`: recall rested on the lexical lane plus the opt-in vector lane.

`recall_limitations` is the plain-language honesty layer derived from that basis:

- For `lexical`, the receipt reports both caveats:
  - `Recall is lexical-only: a semantically distant unit can be missed unless a structural edge such as CROSS_REF reaches it.`
  - `Lexical recall trails embedding-based retrieval where query and content vocabulary diverge; this run makes no claim to beat embeddings.`
- For `lexical_plus_vector`, `recall_limitations` is empty because the vector lane contributed semantic recall.

This split matters because omnex does not let a reader confuse lexical retrieval with semantic retrieval. The receipt says which one happened, and the caveats follow mechanically from that fact.

## `reference_closure_complete`

`reference_closure_complete` is an exact completeness claim about a computed closure, not a heuristic quality score.

It is `True` only when both conditions hold:

1. A tier actually computed a reference closure.
2. Every unit in that closure was emitted in full.

It is `False` when:

- no closure tier ran,
- the computed closure was empty, or
- a closure existed but the emitted bundle did not include every unit in it.

In the shipped kernel, T1 is the closure tier. T0 returns an empty closure set, so T0 receipts report `False`. T2 adds the vector lane, not the T1 closure, so current T2 receipts also report `False`.

## Example shapes

The CLI markdown renderer and the JSON serializer both derive from the same `receipt_dict(receipt)` data. The markdown form renders `recall_limitations` as its own subsection; the JSON form includes it inline as a field.
The values below are illustrative; the shape is the important part.

### Markdown `## Receipt` block

```markdown
## Receipt

- returned_tokens: 824
- baseline_tokens: 4821
- tiers_run: T0
- model_used: False
- model_version: None
- extraction_used: False
- determinism_class: byte_exact
- reference_closure_complete: False
- recall_basis: lexical
- embedding_provenance: None

### Recall limitations

- Recall is lexical-only: a semantically distant unit can be missed unless a structural edge such as CROSS_REF reaches it.
- Lexical recall trails embedding-based retrieval where query and content vocabulary diverge; this run makes no claim to beat embeddings.
```

That shape matches `src/omnex/cli.py`: the CLI prints the rendered bundle first, then a `## Receipt` section, then one `- key: value` row per serialized field other than `recall_limitations`, and finally a `### Recall limitations` block when the tuple is non-empty.

### JSON receipt object

```json
{
  "returned_tokens": 824,
  "baseline_tokens": 4821,
  "tiers_run": ["T0"],
  "model_used": false,
  "model_version": null,
  "extraction_used": false,
  "determinism_class": "byte_exact",
  "reference_closure_complete": false,
  "recall_basis": "lexical",
  "embedding_provenance": null,
  "recall_limitations": [
    "Recall is lexical-only: a semantically distant unit can be missed unless a structural edge such as CROSS_REF reaches it.",
    "Lexical recall trails embedding-based retrieval where query and content vocabulary diverge; this run makes no claim to beat embeddings."
  ]
}
```

This is the shape produced by `omnex._surface.receipt_dict`: dataclass fields from `asdict(receipt)`, plus `recall_limitations`.

## How surfaces expose the receipt

| Surface | Where the receipt appears | Notes |
| --- | --- | --- |
| CLI `omnex query CORPUS QUESTION --budget INT --format json` | Top-level `receipt` field in the JSON result | `src/omnex/_surface.py` builds `{ "bundle": ..., "receipt": receipt_dict(receipt) }`, and `src/omnex/cli.py` serializes that shared payload as deterministic, key-sorted JSON. |
| MCP `query` tool | Top-level `receipt` field in the returned envelope | `src/omnex/mcp.py` returns the same `result_payload(bundle, receipt)` object the CLI uses, so the MCP server exposes the identical receipt schema. |
| LangChain and LlamaIndex integrations | `omnex_receipt` inside each emitted document or node's metadata | `src/omnex/integrations/_common.py` stores the JSON-serializable receipt provenance under `omnex_receipt`, so every emitted item carries the same auditable receipt alongside unit provenance. |

The practical rule is simple: omnex has one receipt shape, and every surface reuses it rather than redefining it.

## Summary

A receipt is omnex's compact trust contract for a retrieval run:

- it is frozen and auditable,
- it is byte-identical on the byte-exact tiers for identical inputs,
- it states exactly which determinism class the run may claim,
- it tells you whether recall was lexical-only or vector-assisted,
- it makes closure completeness an exact set-membership fact, and
- it serializes consistently across the CLI, MCP server, and retriever integrations.

If you are consuming omnex programmatically, treat the receipt as the canonical explanation of what the retrieval run did and what claims that run is allowed to make.
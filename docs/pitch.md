# omnex: a retrieval idea worth pressure-testing

## 1. Hook

The durable pain is not just that LLM context is expensive. It is that more context often makes answers worse. Chroma's 2025 context-rot study reports degradation across 18 frontier models as input length rises, with serious degradation by roughly 50K tokens inside 200K-token windows and 30–50% accuracy drops; the older "lost in the middle" result shows more than a 30% drop when the relevant evidence sits mid-context rather than near the edges ([Chroma](https://www.trychroma.com/research/context-rot), [Lost in the Middle](https://www.emergentmind.com/papers/2307.03172), [Morph summary](https://www.morphllm.com/context-rot)). That makes retrieval a quality problem before it is a cost problem.

At the same time, token prices have been falling fast. The brief's cited pricing surveys put mainstream input pricing around $2.50–$3 per 1M tokens and describe a roughly 80% drop from early 2025 to early 2026 ([CloudZero](https://www.cloudzero.com/blog/llm-api-pricing-comparison/), [PE Collective](https://pecollective.com/blog/llm-api-pricing-comparison/)). So "we make prompts cheaper" is not a moat by itself. If there is a real opportunity here, it has to be about answer quality, precision, reproducibility, and auditability under growing context windows.

## 2. The idea

**omnex** is a universal, structure-aware retrieval engine that turns heterogeneous corpora into token-budget-aware context for LLMs. It ingests each modality through an adapter, converts everything into one intermediate representation, then uses a modality-blind kernel to retrieve, graph-expand, and pack complete units into a budgeted `ContextBundle` plus an auditable `Receipt`.

**Tagline:** Universal, structure-aware retrieval — at a fraction of the tokens.

## 3. Why now

Longer context windows did not kill retrieval; they raised the penalty for low-precision retrieval. If context rot is architectural, then blindly stuffing more tokens into the window becomes self-defeating. The need is not "yet another RAG stack," but a retriever that can return fewer, more complete, more defensible tokens.

Also, hybrid search is no longer novel. BM25 + vector fusion is table stakes in 2026 and should be treated that way, not pitched as differentiation. The interesting gap is elsewhere: deterministic, model-free, auditable, local-first retrieval for teams that care about reproducibility, air-gapped operation, CI stability, and being able to explain exactly why a piece of context was included.

That is why the recommended first proof is narrow and honest: prove **T1 on structured specs first**, where structure is real and closure is provable. T0 still serves every modality as the deterministic floor. T2 follows when we want to be competitive on prose.

## 4. What's different

- **Structure as a dependency graph, not just text to rank.** Where the source has real edges (`$ref`, foreign keys, imports, calls), omnex can retrieve the transitive closure deterministically instead of hoping semantic similarity happens to surface every required neighbor.
- **Deterministic by default, with receipts.** T0 and T1 are byte-exact, LLM-free, offline, and auditable. The output includes a receipt that says what was included, what tier ran, and where any model-assisted extraction entered the pipeline.
- **Tiered honesty instead of one fuzzy claim.** T0 is the safe floor for any modality, T1 is the strongest deterministic proof on structured specs/code, T2 is the prose-competitive lane, and T3 is for perceptual extraction. This is not one monotonic slider.
- **Local-first and zero-infra by default.** The core path is SQLite FTS5, graph traversal, and deterministic packing. Hybrid BM25+vector is table stakes, not the pitch.

The tiers are the key honesty device, so the table matters:

| Tier | Adds | Determinism class | Where it bites | Win bar / claim | Headline baseline |
| --- | --- | --- | --- | --- | --- |
| **T0** (default floor) | FTS5/BM25F + efficiency packer | byte-exact, LLM-free, offline | any modality | "far fewer tokens than full-dump; zero model; reproducible; auditable" | full-document dump (upper bound) |
| **T1** | deterministic graph **closure** expansion | byte-exact, LLM-free | structured specs / code (real `$ref`/FK/import deps) | "complete reference-closure at budget; tokens <= chunk-and-embed at equal recall" | chunk-and-embed top-k |
| **T2** | local embedding lane (opt-in extra) | reproducible only with pinned model + tokenizer + runtime + arch | prose / NL queries | "tokens <= chunk-and-embed at equal recall on prose" | chunk-and-embed (STRONG config) |
| **T3** | model extraction (OCR / caption / transcribe) | cached-by-content-hash, model-versioned in receipt | scanned PDF / perceptual (image/audio/video) | "structured retrieval over non-text inputs at all" | coverage, not tokens |

## 5. Before / after

The strongest example is an OpenAPI spec.

**Before: chunk-and-embed.** A query like "What's the request/response shape for creating a payment?" will usually retrieve `POST /payments` and maybe `PaymentRequest`, because those are semantically close to the query. But it can still miss `Money` and `Address`, even though they are required through `$ref` edges. The returned context looks relevant but is structurally incomplete; to recover, you raise `k` and start dragging unrelated schemas into the budget.

**After: omnex T1.** omnex starts from `POST /payments`, then walks the `$ref` closure deterministically: request closure `PaymentRequest -> {Money, Customer -> Address}` plus response closure `Payment -> Money`, with shared `Money` deduped. The result is complete because completeness is the transitive closure of real edges, not a probabilistic guess. That is the core pitch: not "better similarity," but "provably complete where structure exists," usually at fewer tokens because the system does not have to over-retrieve to be safe.

A quick sanity check on prose keeps the honesty intact. In the brief's TLS example, an FTS-only floor can match the obvious `Ingress` and `TLS secrets` material yet still miss a page titled "Securing traffic with certificates" because the vocabulary does not line up. That is exactly why the product is tiered: T0 is the reproducible floor for any modality, but only T2 is meant to be prose-competitive.

## 6. Market

This is not a tiny niche if the product earns a real wedge. The brief's cited market reports put the **RAG market at roughly $2.8–3.3B in 2026**, growing toward roughly **$10–11B by 2030** at about **38–49% CAGR** ([MarketsandMarkets](https://www.marketsandmarkets.com/Market-Reports/retrieval-augmented-generation-rag-market-135976317.html), [Grand View Research](https://www.grandviewresearch.com/industry-analysis/retrieval-augmented-generation-rag-market-report), [Precedence Research](https://www.precedenceresearch.com/retrieval-augmented-generation-market), [Mordor Intelligence](https://www.mordorintelligence.com/industry-reports/retrieval-augmented-generation-market)).

Separately, the **vector database market is around $3.2B in 2026**, which is a useful proxy for how much infrastructure spend already exists around retrieval systems ([MarketsandMarkets](https://www.marketsandmarkets.com/Market-Reports/vector-database-market-112683895.html), [Global Market Insights](https://www.gminsights.com/industry-analysis/vector-database-market)).

## 7. Honest risks / open questions

- **Prose is the weakest-moat modality.** The strongest demand may be in prose, but prose mostly gives us a containment tree, not a hard dependency graph. T0/T1 will honestly trail embeddings on prose; only T2 closes that gap, and T2 weakens the determinism headline.
- **Benchmark integrity matters more than benchmark scores.** "Fewer tokens at equal recall" is only credible if the chunk-and-embed baseline is strong and the labeling procedure is pre-registered and unbiased.
- **The first beachhead may be narrower than the largest market.** Specs are the strongest proof surface, but they may be a smaller initial market than prose-heavy documentation workflows.
- **The IR is still a design bet until adapter two lands.** A universal intermediate representation validated by one adapter can still be wrong in detail.

## 8. The ask

I want critical feedback, not encouragement.

- Would you use a deterministic, structure-aware retriever if it gave you reproducible outputs and an audit receipt?
- If we prove one modality first, should it be **specs** (stronger moat, cleaner proof) or **prose** (bigger demand, weaker moat)?
- In your pipeline, is reproducibility and auditability actually valuable, or just nice-to-have?
- Who do you know that is already drowning in token spend, context-quality degradation, or retrieval results they cannot explain?

If the answer is "this is only interesting if it wins on prose," that is useful. If the answer is "the determinism story is the real wedge," that is useful too. The point of this note is to test whether the problem feels sharp enough, and whether the proposed first proof is the right one.
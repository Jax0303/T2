# H6 payoff (preliminary): does total-row injection's OSC gain convert to answer accuracy?

Paired test, HiTab dev arithmetic m>=2 (n=161). Retriever = dense top-10; treatment =
dense top-10 ∪ total-like row-chunks (all-column injection). Same solver, same queries.
Script: `scripts/answer_accuracy_injection.py`.

## OSC (LLM-free, full n=161) — completeness DOES rise

| | base (dense@10) | +total-row injection | Δ |
|---|---|---|---|
| OSC | 0.7888 | **0.9068** | **+0.1180** |

19 queries become newly complete (OSC 0→1); mean 21.6 cells injected (blind all-column).

## Answer accuracy (Groq llama-3.1-8b-instant, codegen) — does NOT rise

Two partial runs (the full run is blocked by Groq's **daily token limit, 500k TPD**,
exhausted today):

| checkpoint | acc base | acc treat | flips (wrong→right) |
|---|---|---|---|
| run A, first n=88 | 0.182 | 0.182 | **0** |
| run B, first n=48 | 0.229 | 0.229 | **0** |

Across ~136 paired evaluations, base and treatment answers **never diverge** — zero
queries flipped in either direction.

## Read

**Retrieval completeness (OSC) rises substantially, but final answer accuracy does
not move at an 8B solver.** The binding ceiling here is the solver's *computation*, not
retrieval completeness: the 8B answers only ~18–23% of these hard m>=2 aggregations and
cannot capitalize on the extra complete operand cells. Consistent with the prior
oracle≫operand gap (oracle 0.90 ≫ operand 0.70; ceiling = LLM computation).

**Caveat / next step.** This does NOT show injection is useless — it shows the 8B solver
is the bottleneck. To test whether a *capable* solver converts the OSC gain into
accuracy (the reviewer's likely objection), rerun with a stronger solver (70B / GPT-class).
That run is blocked today by Groq's per-day token limit (resets daily); resume when the
quota refreshes or with a Dev-tier/other key.

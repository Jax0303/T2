# E4 (W6) — context format effect on silent grounding errors (H3)

Retrieval held fixed at the **oracle** operand set (gold cells), so the only
variable is the context *format*. Both arms carry the same numbers and the same
header words; only the (header-path → value) binding differs:

- **flat** — naive dump: "`<leaf header> <value>`" tokens in one blob.
- **struct** — one "`<full header path> = <value>`" line per operand cell.

Answerer: codegen mode, Groq llama-3.1-8b-instant. Population: HiTab dev arithmetic
m≥2, n=158. seed=42, paired bootstrap 95% CI + McNemar.

| arm | NM accuracy | silent-wrong rate | non-number |
|---|---|---|---|
| flat | 0.335 (53/158) | 0.665 (105/158) | 0.000 |
| **struct** | **0.576 (91/158)** | 0.424 (67/158) | 0.000 |

- **ΔNM (struct − flat) = +0.241**, CI [0.158, 0.323] (excludes 0).
- McNemar: struct-only 49 vs flat-only 11 (n_discordant 60) — strongly asymmetric.

**Verdict: H3 supported.** With retrieval held perfect, making the
(header-path, value) binding explicit raises numeric-match accuracy from 0.34 to
0.58 and cuts the silent-grounding-error rate from 0.66 to 0.42. The flat dump
forces the model to guess which value goes with which header; it does so wrongly
two-thirds of the time, emitting a number with no exception (a *silent* error).

**Caveat / residual.** Even with oracle content and structured format, the
silent-wrong rate is still 0.42 — explicit binding helps a lot but does not close
the gap; the residual is the small model's arithmetic/grounding capability (8b,
codegen). non-number = 0 means codegen always returned a number (no refusals), so
the format effect is purely on *which* number, i.e. grounding — exactly the H3
mechanism (`docs/FAILURE_ANALYSIS.md` bucket ⑥).

Artifact: `results/e4_format.json` ·
reproduce: `PYTHONPATH=. python scripts/e4_format.py --split dev --llm groq:llama-3.1-8b-instant`

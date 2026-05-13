# 🧠 RAG-Hallucination-Monitor

> Detects when an AI ignores your document and answers from memory instead.

---

## The Problem

In February 2024, Air Canada's chatbot invented a bereavement fare discount that didn't exist in its own policy document. A grieving passenger was misled. Air Canada was taken to court. They lost.

The chatbot wasn't broken. It was doing something specific and detectable: it ignored the document it was given and answered from its training memory instead.

This is called **parametric override** — the model's learned beliefs about the world overriding the document in front of it. It is the most dangerous class of RAG hallucination because the answer sounds confident, fluent, and completely wrong.

**RAG-Hallucination-Monitor detects this in real time, before the wrong answer reaches the user.**

---

## How It Works

Most hallucination detectors look at the output — they compare the AI's answer to the source document after the fact.

RAG-Hallucination-Monitor looks **inside the model while it is thinking**.

It runs the model three ways on every query:
- With your document
- Without any document (pure memory baseline)
- With a random unrelated document (control)

Then it compares the internal neural activity across all three runs. If the model behaves the same with and without your document, the document is not affecting its computation. That is the hallucination signal.

On top of that, it decomposes the generated answer into atomic claims, retrieves the best supporting evidence span for each claim from the document, and scores each (claim, evidence) pair individually. A claim with no grounding in any document sentence is flagged as hallucinated — even if it doesn't contradict the document.

---

## Architecture

Four layers inspired by how the human hippocampus verifies memory:

### Layer 1 — Claim Decomposer

Splits the model's answer into atomic factual claims. Each claim is independently verifiable. "Yes, we offer a 20% bereavement discount" → ["bereavement discount exists", "discount is 20%"].

### Layer 2 — Evidence Retriever

For each claim, finds the document sentence with highest semantic alignment. Uses sentence embeddings (all-MiniLM-L6-v2). Returns both the evidence span and a retrieval similarity score.

### Layer 3 — Span-Level NLI

Scores grounding of each (claim, evidence span) pair using DeBERTa NLI. Short pairs only — avoids the distribution shift failure of whole-document NLI on unusual or legal text. Low retrieval similarity is treated as evidence absence — a hallucination signal, not a neutral.

### Layer 4 — Brain-Native Evidence Integrator

Continuously integrates per-claim grounding scores with internal activation signals from the model's neural layers. No thresholds. The belief converges to grounded or hallucinated based on the proportion of claims with strong evidence support, modulated by six internal signals:

| Signal | Biological analog | What it measures |
|---|---|---|
| CDR delta | Attention/MLP ratio | Is the model in reading or recall mode? |
| ECS (External Context Score) | CA3 binding | How aligned is attention to the document? |
| PKS (Parametric Knowledge Score) | CA3 completion | How strongly is parametric memory active? |
| Uptake (KL divergence) | Hippocampal novelty detection | Did the document shift the model's distribution? |
| MMD (Symmetric KL) | Pattern separation | How different is this document from a random one? |
| θ/γ Ratio | Theta-gamma phase coding | Reading mode vs recall mode balance |

---

## Key Findings

Trained on Gemma-2-2b-it using TransformerLens for activation access.

- **Three-phase CDR pattern** at layers 4-7 (separation), 8-9 (knowledge check, Cohen's d = +1.044, p < 0.0001), 19+ (commitment) — a three-phase internal conflict structure that predicts hallucination before the answer is formed
- **Internal activations carry more information** about parametric override than output text — the brain-native ensemble outperforms both a standard LR classifier and NLI-based text comparison alone
- **Claim-level span verification** eliminates false positives on adversarial legal and contractual text where whole-document NLI fails
- **Activation patching** confirms causal claim: mean Δ_LD = +6.36, patching context activations into parametric runs causally shifts the output toward the document answer

---

## What It Catches

✅ Model reads policy document, answers "no bereavement discount" → **LOW RISK**

✅ Model reads predatory contract (AeroCloud), correctly summarises absurd cancellation terms → **LOW RISK**

✅ Model given irrelevant document, refuses to answer → **LOW RISK**

🚨 Model ignores policy document, invents bereavement discount → **HIGH RISK**

🚨 Model given empty policy, invents return window and requirements → **HIGH RISK**

---

## Deployment Note

The full 3-pass evaluation runs at approximately 1.3x the cost of a standard RAG forward pass when Pass A (parametric baseline) and Pass C (random document) are cached. In production, caching reduces the marginal overhead to the hook extraction on Pass B plus the claim verification pipeline.

Recommended for high-stakes RAG deployments: legal, medical, financial, insurance, customer service.

---

## Stack

- **Model:** Gemma-2-2b-it (Google)
- **Mechanistic interpretability:** TransformerLens
- **NLI:** potsawee/deberta-v3-large-mnli
- **Semantic encoder:** sentence-transformers/all-MiniLM-L6-v2
- **Training:** Kaggle 2x Tesla T4 (16GB each)
- **UI:** Gradio

---

## References

- ReDeEP: ECS and PKS signals (ICLR 2025)
- LUMINA: MMD and top-overlap signals (NeurIPS 2025)
- PCIB: Context uptake KL divergence (January 2026)
- Two Pathways: CDR delta framework (January 2026)
- Ji-An et al.: Induction head monitor (NeurIPS 2024)
- Gold & Shadlen: Drift-diffusion accumulation (2007)
- Vinogradova: CA1 match-mismatch (1975)
- Marr: Hippocampal pattern completion (1971)
- Kornblith et al.: Linear CKA (NeurIPS 2019)

---

*Research project. Not a finished product.*

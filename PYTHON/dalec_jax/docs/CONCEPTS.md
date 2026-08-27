# Two diagrams: how it was built, and what the fast path does

Both render natively on GitHub — edit the text, not an image.

---

## 1 — How the port was built

The question about an LLM-written port is not "does it run" but "why would I
trust it". The answer is the shape of the process: the unmodified C is never
the thing being replaced during development. It is the referee, at every
level, against tolerances fixed before any porting started.

```mermaid
flowchart TB
    ASK["<b>The instruction</b><br/>“Convert DALEC_1100 from C to JAX — and write tests in which<br/>the <b>old C code</b> produces the reference output the JAX must reproduce.”<br/><i>the second clause is the whole design</i>"]

    subgraph LOOP [" "]
        direction LR
        C["<b>UNMODIFIED C</b><br/><i>the referee</i><br/><br/>compiled at production flags<br/>into an oracle that calls the<br/>model directly and dumps<br/>every intermediate"]
        J["<b>JAX TRANSCRIPTION</b><br/><i>the candidate</i><br/><br/>leaf modules → step body →<br/>240-step scan → EDCs →<br/>likelihood<br/><br/>operation order preserved"]
        C -- "golden files<br/><small>module I/O · trajectories · EDC slots · 31 likelihood terms</small>" --> J
        J -- "mismatch → <b>classify, don't patch</b><br/><small>transcription bug · XLA op semantics · irreducible ULP</small>" --> C
    end

    GATE["<b>A tolerance contract, fixed before porting</b><br/>leaf modules 1e-15 · sub-step checkpoints 1e-13 · trajectories 1e-10<br/>EDC gates <b>exact</b> · likelihood 1e-12 · paper analyses 1e-10<br/><i>every override must cite a measured cause, not a hunch</i>"]

    AUDIT["<b>Adversarial audit</b><br/>a fresh agent is told to <b>refute</b> each surprising claim,<br/>never to confirm it<br/><br/>3 rounds — each overturned a headline claim"]

    EFFORT["<b>Where the effort went</b><br/>1,888 lines of model<br/>2,016 lines of oracle + tests<br/><br/><i>Writing the model was the short part.<br/>Proving it was the same model was the project.</i>"]

    OUT["≤1e-10 per timestep and variable · 60,000 EDC operator slots, 0 mismatches · 59/59 paper analyses reproduced<br/>C defects reproduced deliberately and catalogued, never silently “fixed” — so the two engines stay interchangeable"]

    ASK --> LOOP --> GATE --> AUDIT --> OUT
    GATE --> EFFORT

    classDef ref fill:#fdf0ef,stroke:#b2182b,stroke-width:2px,color:#1a1a1a
    classDef cand fill:#eef4fa,stroke:#1f6fb4,stroke-width:2px,color:#1a1a1a
    classDef neutral fill:#f6f7f8,stroke:#b9bfc6,color:#1a1a1a
    classDef good fill:#eef7f0,stroke:#2a9d5c,stroke-width:2px,color:#1a1a1a
    class C,AUDIT ref
    class J,ASK,OUT cand
    class GATE neutral
    class EFFORT good
```

**Note on sources.** The model was transcribed from the **C source**, not from
papers — papers cannot specify operation order, a 7-digit π, or which defects
to preserve. The methodology follows arXiv:2606.07681; Bloom & Williams (2015),
Norton et al. (2023) and Yang et al. (2022) informed the EDCs, the model
lineage and the FluxVal protocol respectively.

---

## 2 — What "Laplace-guided MCMC" means

The optimizer does **not** replace the chain. It supplies three things the
chain would otherwise have to discover for itself, expensively.

```mermaid
flowchart LR
    P["<b>The problem</b><br/><br/>89 parameters<br/>hard EDC boundary:<br/>outside is −∞, not<br/>merely unlikely<br/><br/>a blind draw passes<br/>about once in 10⁶"]

    subgraph FAST ["What exact gradients buy"]
        direction TB
        M["<b>1 · Where to start</b><br/>multipoint L-BFGS on the<br/>exact 89-parameter gradient<br/><br/><b>15/16</b> modes ≥ the best<br/>sample the chain ever stored"]
        H["<b>2 · Which way to step</b><br/>exact Hessian → covariance,<br/>flat directions capped at the<br/>prior width<br/><br/><b>2.4–3×</b> the mixing of<br/>per-parameter tuning"]
        U["<b>3 · What to expect</b><br/>the repaired Gaussian as a<br/>provisional answer in minutes<br/><br/>widths match the MCMC to a<br/>median ratio <b>0.92</b>"]
    end

    CH["<b>The C MCMC still<br/>delivers the posterior</b><br/><br/>now started at a mode,<br/>stepping in the right shape,<br/>and checkable against a<br/>provisional answer"]

    AUD["<b>Free convergence audit</b><br/>MAP minus chain-best is a<br/>per-site alarm the chain<br/>cannot ring for itself"]

    P --> FAST
    M --> CH
    H --> CH
    U --> CH
    CH --> AUD

    classDef prob fill:#fdf0ef,stroke:#b2182b,stroke-width:2px,color:#1a1a1a
    classDef step fill:#eef7f0,stroke:#2a9d5c,stroke-width:2px,color:#1a1a1a
    classDef chain fill:#eef4fa,stroke:#1f6fb4,stroke-width:2px,color:#1a1a1a
    class P prob
    class M,H,U step
    class CH,AUD chain
```

### The honest limits

| | |
| --- | --- |
| The Gaussian is a **screening** answer | a symmetric bell cannot represent this posterior's skew; the MCMC stays the referee for published numbers |
| The optimizer can land **on the boundary** | only **39 of 64** modes across 8 sites had a finite, positive-definite Hessian — where the best mode is not among them, the covariance is unusable |
| That used to fail **silently** | a NaN covariance gives NaN proposals, every Metropolis ratio compares false, and the chains sit frozen at 0% acceptance looking like a merely hard target. Both failure modes now raise. |
| Gradient samplers do **not** rescue this | NUTS freezes on the hard EDC cliffs (step size → 2.5e-13); a soft-EDC formulation is the untested prerequisite |

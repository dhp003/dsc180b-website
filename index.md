<style>
  @media only screen and (min-width: 640px) {
    header {
      width: 100% !important;
      float: none !important;
      position: relative !important;
    }
    section {
      width: 100% !important;
      float: none !important;
      margin-left: 0 !important;
    }
    footer {
      position: relative !important;
      width: 100% !important;
      bottom: auto !important;
    }
  }
  body {
    overflow-x: hidden;
  }
  .plots-wrap {
    width: 75vw;
    max-width: 1200px;
    margin: 12px auto 0;
  }
  .plots-wrap iframe {
    width: 100%;
    height: 1100px;
    border: 0;
    display: block;
  }
  @media (max-width: 900px) {
    .plots-wrap {
      width: 100%;
    }
    .plots-wrap iframe {
      height: 1000px;
    }
  }
</style>

## Quick Navigation
- [Overview](#overview)
- [Introduction](#introduction)
- [Research Question](#research-question)
- [Methods](#methods)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Impact and Implications](#impact-and-implications)
- [Team](#team)
- [Links](#links)

---

## Overview

Large language models (LLMs) often demonstrate **in-context learning (ICL)** which is the ability to infer a task from examples provided in the prompt without updating model parameters. Recent theoretical work suggests that attention mechanisms may implicitly implement optimization procedures such as gradient descent when solving simple tasks like linear regression.

In this project, we investigate whether this behavior depends on the **specific structure of the attention mechanism**. We compare several attention variants, including standard softmax attention, linear attention, sparse attention, grouped-query attention, low-rank attention, and gated attention, under the **same experimental setup**.

Our goal is to understand the **tradeoff between efficiency and learning capability** in transformer architectures.

---

## Introduction

### What is In-Context Learning?

In-context learning is an interesting phenomenon that describes the ability of LLMs (transformers) to learn an input-output mapping from examples provided within the prompt. 

For example, suppose a prompt contains several input-output examples of a function:

x₁ → y₁  
x₂ → y₂  
x₃ → y₃  

followed by a new input: x*

The model must predict the corresponding output: y*

Despite never seeing this exact task before, large transformer models can often infer the relationship between inputs and outputs from the examples and apply it to the query. This behavior is surprising because the model is not updating its weights. Instead, it appears to adapt purely through the computation performed during the forward pass. In other words, the model is effectively learning at inference time from the prompt itself.

### What is Attention?

Attention is the core mechanism that allows transformers to determine which parts of an input sequence are relevant when computing a representation for each token.

Instead of processing tokens strictly from left to right, attention allows each token in the sequence to look at other tokens and determine how relevant they are.

In practice, attention works by computing similarity scores between tokens using three learned vectors:

- **Query (Q):** what information the token is looking for  
- **Key (K):** what information the token contains  
- **Value (V):** the information that will be shared

The similarity between queries and keys determines how strongly information from one token should influence another. The final representation of a token is therefore a weighted combination of information from other tokens in the sequence.

This mechanism is particularly important for ICL, because it allows the model to read example input–output pairs in the prompt and combine them to infer the relationship needed to solve the query task.

### Problem Statement

Previous research has shown that these models are able to perform in-context learning due to their attention mechanism. At a high level, attention allows transformers to selectively relate information across tokens, which is a key part of how they form useful contextual representations.

However, one of the main drawbacks of standard attention is that it grows quadratically as the input size increases. For example, take a prompt of 100 tokens, this would require 10,000 "attention connections". This was one of the main limitations with the first iteration of LLMs, the size of the prompt you could give it was capped. 

In order to address this, researchers came up with more efficient attention mechanisms that didn't grow quadratically with the input size. Although they were more efficient, with less "attention connections," they weren't as powerful. 

This begs the question: Do these more efficient attention mechanisms still support the same level of in-context learning as standard attention? What are the tradeoffs of using more efficient attention mechanisms?

<iframe
  src="{{ 'assets/attention-slideshow.html' | relative_url }}"
  title="Attention slideshow"
  width="100%"
  height="900"
  scrolling="no"
  style="border: 0; overflow: hidden; display: block;"
></iframe>

> **Why this matters:** Efficient attention can reduce compute and memory costs, but it may also change how well a model learns from examples in the prompt. Our goal is to understand that tradeoff.

### Target Users & Stakeholders

- **Researchers:** to study which architectural features support optimization-like in-context learning.
- **Engineers:** to evaluate the tradeoff between model efficiency and adaptation ability in real deployments.

---

## Research Question

Our central research question is:

**How does the structure of the attention mechanism affect a transformer’s ability to perform in-context learning?**

Specifically, we investigate whether different attention mechanisms:

- achieve similar prediction performance on regression tasks
- produce update directions aligned with gradient descent
- exhibit tradeoffs between computational efficiency and learning capability.

To investigate this question, we compare several alternative attention mechanisms under the same experimental setup. 

---

## Methods

### Attention Variants

We compare six attention mechanisms in the same transformer setup:

- **Standard Softmax:** full-context attention with softmax normalization; our baseline.
- **GQA:** shares key-value projections across groups of query heads to reduce cost.
- **Sparse:** restricts token-to-token interactions to improve efficiency.
- **Linear:** removes softmax and is closely tied to gradient-descent interpretations of ICL.
- **Low-Rank:** compresses attention into a lower-dimensional representation.
- **Gated:** uses learned gates to selectively incorporate information.
<div class="mechanism-grid">
  <div class="mechanism-card">
    <h4>Standard</h4>
    <div class="mech-subtitle">Softmax baseline</div>
    <div class="mech-row"><span>Softmax</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Global context</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Memory cost</span><strong>High</strong></div>
    <div class="mech-row"><span>Main idea</span><strong>Full attention</strong></div>
    <p>Baseline transformer attention with unrestricted token-to-token interaction.</p>
  </div>

  <div class="mechanism-card">
    <h4>GQA</h4>
    <div class="mech-subtitle">Shared KV heads</div>
    <div class="mech-row"><span>Softmax</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Global context</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Memory cost</span><strong>Medium</strong></div>
    <div class="mech-row"><span>Main idea</span><strong>Shared KV</strong></div>
    <p>Reduces cost by letting multiple query heads share key and value projections.</p>
  </div>

  <div class="mechanism-card">
    <h4>Sparse</h4>
    <div class="mech-subtitle">Restricted connectivity</div>
    <div class="mech-row"><span>Softmax</span><strong>Usually yes</strong></div>
    <div class="mech-row"><span>Global context</span><strong>Limited</strong></div>
    <div class="mech-row"><span>Memory cost</span><strong>Low</strong></div>
    <div class="mech-row"><span>Main idea</span><strong>Fewer links</strong></div>
    <p>Only a subset of token pairs interact, improving efficiency but reducing coverage.</p>
  </div>

  <div class="mechanism-card">
    <h4>Linear</h4>
    <div class="mech-subtitle">No softmax</div>
    <div class="mech-row"><span>Softmax</span><strong>No</strong></div>
    <div class="mech-row"><span>Global context</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Memory cost</span><strong>Low</strong></div>
    <div class="mech-row"><span>Main idea</span><strong>Linear update</strong></div>
    <p>Removes softmax and is closely connected to one-step gradient descent interpretations of ICL.</p>
  </div>

  <div class="mechanism-card">
    <h4>Low-Rank</h4>
    <div class="mech-subtitle">Compressed attention</div>
    <div class="mech-row"><span>Softmax</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Global context</span><strong>Compressed</strong></div>
    <div class="mech-row"><span>Memory cost</span><strong>Low</strong></div>
    <div class="mech-row"><span>Main idea</span><strong>Projection</strong></div>
    <p>Approximates full attention using a lower-dimensional summary of keys and values.</p>
  </div>

  <div class="mechanism-card">
    <h4>Gated</h4>
    <div class="mech-subtitle">Learned update control</div>
    <div class="mech-row"><span>Softmax</span><strong>No</strong></div>
    <div class="mech-row"><span>Global context</span><strong>Yes</strong></div>
    <div class="mech-row"><span>Memory cost</span><strong>Low</strong></div>
    <div class="mech-row"><span>Main idea</span><strong>Selective update</strong></div>
    <p>Uses a learned gate to control how strongly new information changes the representation.</p>
  </div>
</div>

More concretely, the following table identifies when and why to use each mechanism over others

| Variant | Main idea | Why include it? |
|---|---|---|
| Standard | Full softmax attention | Baseline |
| GQA | Shared KV heads | Efficiency |
| Sparse | Limited connectivity | Scalability |
| Linear | No softmax | GD connection |
| Low-Rank | Compressed attention | Approximation |
| Gated | Learned update control | Flexible dynamics |

### Experiments

Four controlled sweeps are run on synthetic in‑context linear regression to isolate how attention structure shapes in‑context learning. All attention variants are evaluated under a matched training setup and the same evaluation metrics.

**Common setup**
- **Task:** in‑context linear regression with context pairs \((x_i, y_i)\) and a query \((x_\ast, 0)\)
- **Input dimension:** \(d = 20\)
- **Evaluation:** `num_eval_tasks = 1000`
- **Metrics:** Mean Squared Error (MSE); cosine similarity between the model’s implied update and a one‑step gradient‑descent baseline
- **Models:** Softmax, Linear Self (LSA), Kernelized Linear, Grouped‑Query (GQA), Gated Linear (GLA), Sparse Causal, Low‑Rank (k = 0.5–1.0)

**1) Training‑Steps Sweep (Learning Curve)**
- **Goal:** how performance evolves with optimization
- **Parameters:** `num_layers = 8`, `n_points = 41`, `train_steps ∈ {0, 1k, 2k, 5k, 10k, 20k}`

**2) Layers Sweep (Depth Scaling)**
- **Goal:** how ICL scales with depth
- **Parameters:** `num_layers ∈ {2, 4, 8, 16, 32, 64}`, `n_points = 41`, training steps set uniformly (or step‑scheduled when shallow models use fewer steps)

**3) Context Sweep (Trained)**
- **Goal:** effect of in‑context length after training
- **Parameters:** `num_layers = 8`, `train_steps = 5k`, `n_points ∈ {5, 10, 20, 40, 80, 120, 160, 200}`

**4) Context Sweep (Zero‑Train)**
- **Goal:** in‑context behavior at random initialization
- **Parameters:** `num_layers = 8`, `train_steps = 0`, `n_points ∈ {5, 10, 20, 40, 80, 120, 160, 200}`

### Data Generation

Each task samples a fresh ground‑truth linear map `W*`, draws inputs `x` from a zero‑mean distribution, and defines outputs `y = W* x`. A prompt is constructed from several context pairs `(x_i, y_i)` plus a query input `x*`. The query token is represented as `(x*, 0)`, and the model must predict `y*`.

A new regression task is sampled for every training example, preventing memorization and forcing learning from the prompt itself. The setup remains simple enough to compare against analytic baselines (least squares and one‑step GD) while still capturing the core in‑context learning structure.

---

## Results
<div class="plots-wrap">
  <iframe
    src="{{ 'plots/interactive_plots.html' | relative_url }}"
    loading="lazy"
  ></iframe>
</div>

---

## Key Takeaways
- **Training is required for ICL.** At random initialization, performance stays near ~19–22 MSE across context lengths, while the GD baseline improves with more examples. Architecture alone does not produce ICL; optimization is essential.
- **Global connectivity is necessary.** Sparse causal remains near ~19–21 MSE in steps, layers, and context sweeps, and its cosine similarity to GD stays near 0. Limited connectivity blocks the query from aggregating all `(x_i, y_i)` pairs needed to recover regression statistics.
- **Linear / kernelized attention is strongest and most stable.** Linear Self and kernelized linear achieve the lowest MSE in trained regimes (e.g., ~3.35 at 20k steps; ~1.76–1.77 at `n=160`). Their feature‑map formulation preserves global aggregation and aligns with optimizer‑like updates driven by sums of `x_i y_i` and `x_i x_i^T`.
- **Low‑rank attention is competitive but sensitive to rank and variant.** In steps/layers sweeps, block low‑rank with `k ≈ 0.8–1.0` is near‑best (MSE ≈ 3.24–3.57). In the context sweep, non‑block low‑rank improves with more context but stays worse than linear/kernelized (best ≈ 4.17 at `n=160`). Compression preserves global information up to a point, but rank choice and implementation details matter.
- **Softmax / GQA / GLA align with GD but not full accuracy.** At 20k steps, many non‑sparse methods reach high cosine similarity (~0.88–0.94) to one‑step GD while MSE remains higher (e.g., softmax ≈ 7.8, GQA ≈ 8.6, GLA ≈ 6.2). The update direction is learned, but scaling/conditioning is imperfect; shared KV (GQA) and gating (GLA) can damp or distort value flow.
- **Depth helps up to ~8–16 layers, then saturates.** In the layers sweep, linear/kernelized improve sharply through 8 layers and plateau (~3.1–3.3 MSE). Softmax/GQA/GLA continue improving but remain above linear/kernelized, indicating diminishing returns for depth in this regime.
- **More context helps only when aggregation is global.** Trained linear/kernelized and even softmax/GQA improve steadily with larger `n`. Sparse stays flat and GLA is non‑monotonic, reinforcing that additional context is useful only when the mechanism can integrate it.

---

## Impact and Implications
- Why this matters for model design
- Efficiency vs. capability trade-offs
- Potential applications or future work

### Project Scope & Limitations

Within this project, we compare standard, grouped-query, sparse, linear, low-rank, and gated attention mechanisms under the same experimental setup.

Each model is trained and evaluated on **synthetic linear regression tasks**, which allows us to directly compare model behavior with classical optimization methods such as least squares, LASSO, and gradient descent.

However, this setup also introduces several limitations.

First, our experiments are conducted on **synthetic tasks rather than natural language data**, meaning our results may not directly transfer to large-scale language modeling settings.

Second, our analysis is primarily **empirical**. While we measure prediction accuracy and alignment with gradient descent updates, we do not analyze model parameters or derive formal theoretical guarantees explaining the observed behaviors.

Despite these limitations, this controlled setup allows us to isolate the role of the attention mechanism and study how architectural differences influence in-context learning behavior.

Our results are purely experimental, we do not analyze model parameters or have any mathematical arguments to justify our observations.

---

## Team
- **Anish Kasam** — [akasam@ucsd.edu](mailto:akasam@ucsd.edu)
- **Dhanvi Patel** — [dhp003@ucsd.edu](mailto:dhp003@ucsd.edu)
- **Krish Prasad** — [krprasad@ucsd.edu](mailto:krprasad@ucsd.edu)
- **Shou Tai Yue** — [syue@ucsd.edu](mailto:syue@ucsd.edu)

---

## Links
- Project report PDF (to be added)
- [Code repository](https://github.com/Shou-Yue/Incontext-Learning-of-Attention-Mechanisms/tree/main)
- Poster or presentation (to be added)

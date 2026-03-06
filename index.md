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
</style>

## Quick Navigation
- [Overview](#overview)
- [Introduction](#introduction)
- [Background](#background)
- [Research Question](#research-question)
- [Methods](#methods)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Impact and Implications](#impact-and-implications)
- [Team](#team)
- [Links](#links)

---

## Introduction

### Problem Statement

In-context learning is an interesting phenomenon that describes the ability of LLMs (transformers) to learn an input-output mapping from examples provided within the prompt. For example, if you provide a model a [todo][example of in-context learning]. 

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

We compare all attention variants under the same training setup.

1. **Train** each model on synthetic linear regression tasks.
2. **Evaluate** on underparameterized, overparameterized, and sparse regimes.
3. **Compare** prediction error, gradient-descent alignment, and robustness.

We study performance across three regimes:

- **Underparameterized regression:** the number of informative examples is sufficient relative to the input dimension, so the task is well-constrained.
- **Overparameterized regression:** the problem has more degrees of freedom, making generalization and implicit regularization more important.
- **Sparse regression:** the true task depends on only a subset of features, allowing comparison to sparse classical baselines such as LASSO.

For each task, the model receives several context input-output pairs along with a query input, and it must predict the missing query output. We evaluate attention variants using metrics such as prediction error on held-out tasks, alignment with a one-step gradient descent baseline, and robustness under perturbations such as scaling and noise. This setup allows us to test not only whether a model performs well, but also whether its behavior resembles an optimizer operating over the context.

### Data Generation

All models are trained and evaluated on synthetic linear regression tasks generated from the same procedure. For each task, we sample a new ground-truth linear map \(W^\star\) and generate inputs \(x\) from a zero-mean distribution, with outputs defined by \(y = W^\star x\). We then form a prompt consisting of several context pairs \((x_i, y_i)\) and one query input \(x_\ast\), where the model must predict the corresponding target \(y_\ast\).

Following the token construction used in our codebase, each context token concatenates the input and output into a single representation, while the query token contains the query input paired with a zero placeholder for the missing output. A new regression task is sampled for every training example, which prevents memorization and forces the model to infer the task from the prompt itself. This synthetic setup is especially useful because it is simple enough to compare against analytic baselines such as least squares, LASSO, and one-step gradient descent, while still capturing the core structure of in-context learning.

---

## Results
- Comparison of ICL performance across attention types
- Analysis of performance in different regimes (e.g., underparameterized vs. overparameterized)
- Identification of trade-offs and failure modes

---

## Key Takeaways
- Summary of main findings
- What we learned about attention and ICL

---

## Impact and Implications
- Why this matters for model design
- Efficiency vs. capability trade-offs
- Potential applications or future work

### Project Scope & Limitations

Within this project, we compare standard, grouped-query, sparse, linear, low rank, and gated attention. In order to compare them, we evaluated each variant on the same tasks:

[todo][describe the experiments? over/under/sparse?]

However, our results are purely experimental, we do not analyze model parameters or have any mathematical arguments to justify our observations.

Therefore, when interpreting our work, it's important to understand that these experiments are [todo][explain the shortcomings of our work]

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
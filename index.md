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

Previous research has shown that these models are able to perform in-context learning due to their attention mechanism. At a high level, attention is what allows these models to reason & emulate thinking [todo][this isn't completely true, maybe reword]. 

However, one of the main drawbacks of standard attention is that it grows quadratically as the input size increases. For example, take a prompt of 100 tokens, this would require 10,000 "attention connections". This was one of the main limitations with the first iteration of LLMs, the size of the prompt you could give it was capped. 

[todo][generate a visualization showing standard dense attention]

In order to address this, researchers came up with more efficient attention mechanisms that didn't grow quadratically with the input size. Although they were more efficient, with less "attention connections," they weren't as powerful. 

[todo][generate a visualization showing a more efficient attention mechanism]

This begs the question: Do these more efficient attention mechanisms still support the same level of in-context learning as standard attention? What are the tradeoffs of using more efficient attention mechanisms?

Answering this question can influence the architecture of the next generation LLMs [todo][revise this, why should people care?]

### Target User & Stakeholders

The target users and stakeholders for our project are researchers focused on transformer architectures and engineers building or selecting models for applications. For researchers, our experiements can serve as the starting point for assessing which attention mechanisms are best suited for performing in-context learning. Similarly, our experiments can influence design decisions for engineers who are attempting to deploy their own models, they can evaluate the performance of different attention mechanisms in the context of their use case and decide what works best.

---

## Methods

### Attention Variants

Within this project, we compare standard, grouped-query, sparse, linear, low rank, and gated attention.

- Standard Attention: [todo][1-2 sentence explanation]
- Grouped Query Attention (GQA): [todo][1-2 sentence explanation]
- Sparse Attention: [todo][1-2 sentence explanation]
- Linear Attention: [todo][1-2 sentence explanation]
- Low-Rank Attention: [todo][1-2 sentence explanation]
- Gated Attention: [todo][1-2 sentence explanation]

### Experiments

To effectively compare the various attention mechanisms, we selected various tasks, with the attention mechanism being the only changing variable [todo][reword]. This allows us to have a standardized setup to compare the models. The experiments are:

[todo][explain each experiment at a high level, setup, evaluation, etc.]

### Data Generation

[todo][dataset description: synthetic data generation]

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
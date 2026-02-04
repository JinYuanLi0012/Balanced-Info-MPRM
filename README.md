# Training Data Efficiency in Multimodal Process Reward Models

> Make Multimodal Process Reward Model (MPRM) training 10× cheaper by selecting informative Monte Carlo (MC)-annotated rollouts—no extra supervision, no extra model calls.

Check out our [paper](https://arxiv.org/abs/2602.11111)for the details.

## 🔥 Updates

- **[2026-02-04]** We released our [paper](https://arxiv.org/abs/2602.11111) and code. BIS can match full-data performance using as little as **10%** of the training data.

## 🧩 Overview
Training MPRMs usually relies on large-scale MC-annotated corpora, which makes training expensive. Our study shows that random subsampling saturates quickly, implying strong redundancy in existing MC rollouts. 

We identify two key factors that determine whether a rollout yields informative gradient updates:

**Label Mixture**: rollouts that include both positive and negative steps (i.e., “mixed”) provide stronger supervision signals. 

**Label Reliability**: positive steps with higher average MC scores are more reliable; extremely low-MC positives are often noisy pseudo-positives. 

Built on empirical observations and grounded analysis, we propose the Balanced-Information Score (BIS) to rank rollouts by mixture × reliability, using only existing MC signals stored in the dataset.

## ⚡️ Quickstart Guide

Getting started with RelayLLM is straightforward.

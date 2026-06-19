# tinyGPT-AllAboutRunning

A TinyGPT implementation trained on a custom corpus about running and athletics. This project compares multiple tokenization approaches to evaluate their impact on language modeling performance and text generation quality.

---

## Overview

This project was developed as part of a Natural Language Processing (NLP) coursework assignment.

The objective is to train a TinyGPT language model using a custom corpus focused on running and athletics while comparing different tokenization methods:

- Word Tokenization
- Character Tokenization
- Byte Pair Encoding (BPE)

The performance of each tokenizer is analyzed through training loss and generated text quality.

---

## Objectives

- Create a custom corpus containing more than 2000 words.
- Train TinyGPT on the custom corpus.
- Compare multiple tokenization approaches.
- Analyze the impact of tokenization on model performance.
- Generate text using trained language models.

---

## Project Structure

```text
tinyGPT-AllAboutRunning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── corpus.txt
│
├── experiments/
│   ├── word_tokenization.ipynb
│   ├── character_tokenization.ipynb
│   └── bpe_tokenization.ipynb
│
├── models/
│   ├── word/
│   ├── character/
│   └── bpe/
│
├── results/
│   ├── generated_text/
│   ├── plots/
│   └── comparison.csv
│
└── report/
    └── laporan.pdf
```

---

## Model Architecture

The project uses TinyGPT, a lightweight Transformer-based language model designed for next-token prediction tasks.

Main Components:

- Token Embedding
- Positional Encoding
- Multi-Head Self Attention
- Feed Forward Network
- Language Modeling Head

---

## Analysis

The comparison focuses on:

- Vocabulary size
- Training efficiency
- Model convergence
- Generated text quality
- Handling of unseen words

---

## Author

Rifky Danu Asmoro

Student Project – Natural Language Processing

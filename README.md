# 🔊 Environmental Sound Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-EnvSDD-orange.svg)](https://envsdd.github.io/)

## 🚀 Overview

Generative AI can now create highly realistic environmental sounds, such as sirens, footsteps, and nature amibence, raising concerns about misinformation and safety. Unlike speech deepfakes, environmental sounds span diverse categories and exhibit complex generation artifacts.

This repo contains a robust **Deepfake Detection System** designed to distinguish between real recordings and AI-generated audio. Our solution leverages **Self-Supervised Learning (SSL)** to generalize effectively to unseen audio generators (Text-to-Audio and Audio-to-Audio models).

---

## 📚 Dataset

We utilize the **EnvSDD** dataset, a large-scale curated collection designed for benchmarking environmental deepfake detection.

- **Scale:** 45.25 hours of real audio and 316.7 hours of fake audio.
- **Diversity:** Covers a wide range of acoustic scenes beyond human speech.
- **Generators:** Includes samples from multiple diverse generation architectures to test robustness.

> 🔗 [Dataset Info](https://envsdd.github.io/)

---

## 🧠 Model

Our architecture combines a powerful pre-trained front-end with a graph-based back-end to capture subtle deepfake artifacts:

- **Front-End (EAT)**: We use the **Efficient Audio Transformer (EAT)** to extract high-level acoustic representations.
- **Back-End (AASIST)**: These features are processed by an **Integrated Spectro-Temporal Graph Attention Network (AASIST)** to differentiate genuine signals from generated noise.
- Model checkpoint can be downloaded from [here] (https://drive.google.com/file/d/1f26tEVuMwaPQULZnatwgabcVqMyWsxqc/view?usp=sharing)

### 🔧 Key Features

- **Generalization:** Effective against unseen Text-to-Audio (TTA) and Audio-to-Audio (ATA) generators.
- **High-Fidelity Analysis:** Detects artifacts in complex, non-speech environmental sounds.
- **Robust Architecture:** Combines SSL features with graph attention for superior discrimination.

---

## 📊 Results

The model was evaluated using the **Equal Error Rate (EER)** metric, where a lower score indicates better performance.

| Metric | Score |
|--------|-------|
| **Baseline AASIST EER** | ~15.02% |
| **Baseline W2V2+AASIST EER** | ~48.00% |
| **Baseline BEATs+AASIST EER** | ~13.20% |
| **Proposed EAT+AASIST EER** | ~2.48% |

> By combining EAT with AASIST, our solution achieves state-of-the-art performance, drastically reducing error rates compared to standard baselines.

---


# PIFTrack

Official implementation of **PIFTrack**, a novel framework for multi-object tracking in satellite videos via Points-of-Interest (PoIs) based flow modeling and association.

<p align="center">
  <figure>
    <img src="example/FIG1.png" width="50%" alt="Framework Overview">
    <figcaption><b>Figure 1</b>: Nonlinear motion prediction based on diffusion ODE.</figcaption>
  </figure>
</p>

---

## 📰 News

📌 **June 2025**: The manuscript has been submitted to **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**. More details and models will be released upon acceptance. Stay tuned!

---

## 🔍 Introduction

PIFTrack tackles the unique challenges of multi-object tracking (MOT) in satellite videos, such as ambiguous annotations, nonlinear motion, and degraded IoU-based association. To address these issues, PIFTrack introduces:
<!-- - **PIDH Module**: Probabilistic modeling of spatial ambiguity using Points-of-Interest (PoIs).
- **RFMP Module**: Trajectory prediction via Residual Flow-guided Motion Prior, modeling nonlinear dynamics without handcrafted assumptions.
- **SCM Strategy**: An IoU-free association strategy for robust tracking of tiny targets. -->

<p align="center">
  <figure>
    <img src="example/Fig3_a.png" width="85%" alt="Training Structure of PIFTrack">
    <figcaption><b>Figure 2(a)</b>: The training pipeline of PIFTrack.</figcaption>
  </figure>
  
  <figure>
    <img src="example/Fig3_b.png" width="85%" alt="Inference Structure of PIFTrack">
    <figcaption><b>Figure 2(b)</b>: The inference pipeline of PIFTrack.</figcaption>
  </figure>
</p>

---

<!-- ## 🧠 Network Architecture

The architecture of PIFTrack consists of two stages:
1. **Training Stage**: Learns PoI-based representations and motion flows using PIDH and RFMP modules.
2. **Inference Stage**: Applies SCM for effective association, ensuring trajectory continuity under challenging motion and appearance variations.

<p align="center">
  <img src="figures/fig2_network.png" width="85%" alt="Network Structure">
</p>

--- -->

## 🔗 Sparse Cluster Matcher (SCM) Example

The following visualization demonstrates the effectiveness of SCM in associating tiny objects across frames using PoI overlaps, instead of unstable IoU or center distance metrics.

<p align="center">
  <img src="example/Fig4.png" width="60%" alt="SCM Matching Example">
</p>

---

## 🛠️ Engineering Notes

- The **detection** component of PIFTrack is developed based on the [MMDetection](https://github.com/open-mmlab/mmdetection) framework.
- We sincerely thank the **OpenMMLab** team for providing such a powerful detection toolkit.

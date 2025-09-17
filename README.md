# Visual Robustness Based on Faces/Houses Dataset

This repository provides a Python pipeline for analyzing electrocorticography (ECoG) data, aiming to understand how the human brain encodes visual objects under varying levels of sensory noise. The project combines traditional signal processing techniques—such as High-Gamma Activity (HGA) and Event-Related Potentials (ERPs)—with machine learning approaches, including Multivariate Pattern Analysis (MVPA), to map the spatio-temporal dynamics underlying face and house perception.

**Note:** This repository was adapted and expanded from the work of my project peer [Mohammadreza Shahsavari](https://github.com/mohammadrezashahsavari). Since we worked as a team, this repository represents the contributions of all team members, not just me.
---

##  Key Findings

Our analysis shows that the brain recognizes objects in a strong yet flexible way. While different objects have specialized neural patterns, these patterns are affected differently as visual input becomes noisier.

---

## 1. Selective High-Gamma Activity (HGA)

Initial analysis of high-gamma activity (HGA) reveals strong, localized neural responses to specific visual categories. As expected, some cortical areas are highly selective: certain channels respond strongly to faces, while others respond more to houses.

*The figure below shows the average broadband response across 50 ECoG channels for one subject. Note the clear selectivity: channel **ECOG_047** responds strongly to faces (orange line), while channel **ECOG_031** responds to houses (blue line).*

<p align="center">
  <img src="https://github.com/UncleRazavi/Visual-robustness-based-on-ECoG-data/blob/main/results/High%20Gamma%20Activity%20%26%20ERP%20Plots/SubjectIndx1%20-%20NoiseLevel%20All.png" width="800">
</p>

---

## 2. The "Breaking Point" in Neural Decoding

Using a sliding-window MVPA with an SVM classifier, we decoded object categories (face vs. house) from neural activity at different noise levels. The results show that the brain can reliably represent these categories up to a certain point, after which performance drops sharply.

- **Accuracy Drops with Noise:** Decoding accuracy decreases as sensory noise increases. Neural representations are strongest in low-noise conditions and weaken as stimuli become more ambiguous.  

- **The 40–50% Noise "Breaking Point":** Decoding remains above chance until noise reaches ~40–50%, then drops sharply to chance level, revealing a “breaking point” in the brain’s evidence accumulation.

*Left: Decoding accuracy curves for different noise levels. Purple (0–20% noise) and blue (20–40% noise) curves show significant decoding, while higher noise levels fail.  
Right: Peak decoding accuracy decreases with noise, dropping to non-significant levels (red squares) after ~30% noise.*

<p align="center">
  <img src="https://github.com/UncleRazavi/Visual-robustness-based-on-ECoG-data/blob/main/results/Across%20Noise%20Levels%20Across%20Subjects%20Results/noise_levels_comparison.png" width="500">
  <img src="https://github.com/UncleRazavi/Visual-robustness-based-on-ECoG-data/blob/main/results/Across%20Noise%20Levels%20Across%20Subjects%20Results/PeakDecodingAccuracy%20vs%20NoiseLevel.png" width="300">
</p>

This sharp drop is also visible in the channel-wise MVPA results for a single subject, where strong decoding in low-noise bins (1 and 2) disappears at higher noise levels.

<p align="center">
  <img src="https://github.com/UncleRazavi/Visual-robustness-based-on-ECoG-data/blob/main/results/Across%20Noise%20Levels%20Across%20Subjects%20Results/photo_2025-08-03_06-38-05.jpg" width="800">
</p>

---

## 3. Time Delay in Neural Processing

Higher sensory noise introduces a delay in peak neural responses. The grand-average heatmap across all subjects shows that as noise increases, the “hotspot” of maximum decoding accuracy not only weakens but also shifts later in time. This suggests that the brain requires more time to accumulate evidence to identify objects under noisy conditions.

*Decoding accuracy heatmap averaged across subjects. Peak accuracy (yellow/orange) occurs later for the 0.2–0.4 noise level compared to the 0.0–0.2 level, showing a clear temporal delay in processing.*

<p align="center">
  <img src="https://github.com/UncleRazavi/Visual-robustness-based-on-ECoG-data/blob/main/results/Across%20Noise%20Levels%20Across%20Subjects%20Results/noise_levels_heatmap.png" width="700">
</p>

---

##  Data Source

The raw ECoG data (`faceshouses.npz`) used in this project can be downloaded from the Stanford Digital Repository:  
- **Link:** [https://exhibits.stanford.edu/data/catalog/zk881ps0522](https://exhibits.stanford.edu/data/catalog/zk881ps0522)

---

## How to Run

First, clone the repository to your local machine:

```bash
git clone git@github.com:UncleRazavi/Visual-robustness-based-on-ECoG-data.git
cd Visual-robustness-based-on-ECoG-data
```

## About the Scripts

utils.py – The core library containing all essential functions.

main.py – The main entry point that runs the full analysis using utils.py.

Other scripts handle specific tasks (e.g., ERP plotting) or are earlier/partial versions of the analysis.



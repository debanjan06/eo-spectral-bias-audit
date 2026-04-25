# EO-Spectral-Bias-Audit

Independent Research on Multi-Modal AI Robustness in Earth Observation Systems

---

## Project Overview

This is an independent research project investigating and quantifying **Spectral Bias** in multi-modal Earth Observation (EO) models. While many AgTech models report high accuracy on held-out test sets, they often fail silently when deployed across different geographic domains. This repository provides a diagnostic framework to audit those failures.

The core hypothesis: models trained on high-biomass regions develop a hidden over-reliance on meteorological priors, and will confidently produce wrong predictions when presented with out-of-distribution satellite imagery — even when the spatial signal is unambiguous.

---

## The "Smoking Gun": 100% False Positive Detection

This project successfully isolated a critical architectural vulnerability. A Multi-Modal CNN trained on high-biomass regions (California) was subjected to a stress test using arid, high-albedo data (Western Australia).

The result was a complete diagnostic collapse:

| Validation Zone | Environment Type | Accuracy | Status |
|---|---|---|---|
| California (Baseline) | Mediterranean / High Biomass | 72.3% | Learning Confirmed |
| W. Australia (Stress Test) | Arid / Bare Earth / Low NDVI | 0.0% | Spectral Bias Confirmed |

> **Scientific Finding:** Despite spatial inputs showing 100% bare dirt, the model predicted `"Healthy"` with 100% frequency. This proves the Late-Fusion mechanism developed a mathematical over-reliance on meteorological priors — effectively ignoring the satellite imagery entirely when weather conditions appeared acceptable.

---

## Technical Stack & Architecture

| Component | Details |
|---|---|
| **Model** | Multi-Modal Late-Fusion CNN (PyTorch) |
| **Spatial Input** | 4-Channel (RGB-NIR) Sentinel-2 patches |
| **Tabular Input** | 6-feature meteorological vectors (NDVI, SAVI, EVI, Temp, Rainfall, Humidity) |
| **Dataset** | AgriGuard Training Dataset + Real-world OOD samples (Australia & Punjab) |

The Late-Fusion design processes the satellite image stream and the meteorological feature stream independently before combining them at the decision layer — the exact point where domain shift causes catastrophic over-reliance on the tabular branch.

---

## Repository Structure

```
EO-Spectral-Bias-Audit/
│
├── src/
│   ├── train.py                # Training loop for regional domain adaptation
│   ├── evaluate_baseline.py    # Evaluation engine for control-group testing
│   └── evaluate_audit.py       # Scientific core: measures model bias & OOD failure rates
│
├── models/
│   └── *.pth                   # Trained weights used for the audit
│
└── data/
    └── /                       # Raw and processed meteorological time-series
```

- **`train.py`** — Trains the Multi-Modal CNN on a source domain. Configurable for regional dataset inputs.
- **`evaluate_baseline.py`** — Bulletproof evaluation engine for control-group testing on the training distribution.
- **`evaluate_audit.py`** — The scientific core. Injects synthetic OOD spatial signals while holding meteorological inputs constant, isolating and measuring the model's modal bias.

---

## Getting Started

```bash
git clone https://github.com/debanjan06/eo-spectral-bias-audit.git
cd EO-Spectral-Bias-Audit
pip install -r requirements.txt
```

Run the stress test audit:

```bash
python src/evaluate_audit.py --weights models/california_baseline.pth --region australia
```

---

## Why This Matters

Spectral bias of this kind is particularly dangerous in precision agriculture and food security applications, where a model deployed across climate zones may produce confident, incorrect crop health assessments. This audit framework is designed to surface these failures before production deployment.

---

## License

MIT License. See `LICENSE` for details.

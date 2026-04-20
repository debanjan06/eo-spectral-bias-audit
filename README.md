# Addressing Spectral Bias in Agricultural AI
**Making Crop Health Diagnostics Reliable Across the Globe**

## Overview
This repository contains the code and diagnostic dashboard for AgriSight, an AI tool designed to predict crop health from space. 

Machine learning models are increasingly used in agriculture to tell farmers if their crops are stressed or diseased. However, these models are usually trained in lush, high-biomass areas (like the United States). When deployed globally, they often fail. This project builds a crop-health AI and explicitly stress-tests it across three different continents to uncover its hidden flaws before it reaches the farmer.

## How It Works
The model acts as a "digital agronomist" by combining two sources of information:
1. **Spatial Data:** 10-meter resolution satellite images from Sentinel-2 (looking at the actual color and texture of the fields).
2. **Tabular Data:** Historical weather data from Open-Meteo (temperature, rainfall, and humidity).

The AI fuses these two streams of data together to make a final prediction: Healthy, Stressed, or Diseased.

## The Discovery: What is "Spectral Bias"?
To ensure the model works everywhere, it was audited across three global zones:

* **The Baseline (California, USA):** The model performed perfectly, accurately identifying healthy crops in a lush environment.
* **The Scale Test (Punjab, India):** The model successfully processed over 1.1 billion pixels, proving it can handle massive, fragmented networks of small-holder farms.
* **The Stress Test (Western Australia):** This is where the model broke, revealing a critical flaw. 

In Western Australia, the satellite images showed dry, bare earth and stubble. However, because the weather data (temperature and rainfall) looked "normal" to the AI, it completely ignored the satellite pictures of the dirt and confidently predicted the field was a "Healthy Crop." 

This is **Spectral Bias**. The AI learned to trust the weather data so much that it stopped "looking" at the actual fields. Documenting this flaw is the primary contribution of this research, as it highlights the danger of deploying agricultural AI without strict, cross-continental auditing.

## How to Run the Diagnostic Dashboard
This project includes an interactive web dashboard so users can independently load the data and see the Spectral Bias happen in real-time.

**Run via Docker (Recommended)**
```bash
docker build -t eo-audit:v1.0 .
docker run -p 8501:8501 eo-audit:v1.0
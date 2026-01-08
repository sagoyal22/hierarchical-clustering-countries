# Hierarchical Agglomerative Clustering from Scratch: Country Analysis

## Overview
This project implements **Hierarchical Agglomerative Clustering (HAC)** *from scratch* using NumPy and applies it to global country-level socioeconomic data. The goal is to cluster countries based on development indicators and visualize the results both as **dendrograms** and **geographic world maps**.

Unlike library-based clustering, this project manually computes pairwise distances, supports **single and complete linkage**, and constructs the full linkage matrix compatible with SciPy’s dendrogram visualization.

---

## Features
- Manual implementation of **Hierarchical Agglomerative Clustering (HAC)**
- Supports **single linkage** and **complete linkage**
- **Z-score normalization** of feature vectors
- **Dendrogram visualization** using SciPy
- **Geospatial cluster visualization** on a world map using GeoPandas
- Clean pipeline from raw CSV → clustering → visualization

---

## Dataset
The project uses `Country-data.csv`, which contains socioeconomic indicators for countries, including:
- Child mortality
- Exports and imports
- Health expenditure
- Income
- Inflation
- Life expectancy
- Total fertility rate
- GDP per capita

Each country is represented as a **9-dimensional feature vector**.

---

## Project Structure
```text
.
├── Country-data.csv        # Input dataset
├── output.txt              # Normalized feature vectors
├── hac.py                  # Main clustering and visualization code
├── README.md

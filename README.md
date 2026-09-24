# Improving Geological Lineament Detection from Aeromagnetic Data Using Machine Learning: A Case Study in Nigeria

## Overview

This repository contains the data processing and modeling work for a study on ML-based geological lineament detection from aeromagnetic data. The core question is whether machine learning can integrate multiple magnetic attributes to identify lineaments more consistently than individual conventional enhancement techniques, with validation against independent geological evidence rather than the classical interpretation used for training.

## Study Area

Ibadan, Oyo State, Nigeria (NGSA aeromagnetic sheet 261), part of the southwestern Nigerian basement complex.

## Data

High-resolution airborne magnetic data from the Nigerian Geological Survey Agency (NGSA), covering southwestern sheets including Abeokuta, Ibadan, Apomu, Ondo, Akure, Owo, Ijio, Oyo, Iwo, and Ilesha.

## Method

- **Models:** Random Forest and SVM (classical ML, not CNN)
- **Input attributes:** Total Magnetic Intensity (TMI), Reduction to Pole/Equator (RTP/RTE), Analytical Signal, Tilt Derivative, Horizontal Gradient, Vertical Derivative, and related derivatives
- **Validation approach:** Independent geological evidence, avoiding circular validation against the same classical-filter interpretations used for training

## Repository Structure

```
.
├── data/               # Raw and processed aeromagnetic data
├── notebooks/          # Attribute derivation, model training, evaluation
├── src/                # Processing and modeling pipeline
├── figures/            # Maps and result visualizations
└── README.md
```



## Status

First-author manuscript in progress, targeting the Nigerian Journal of Physics (NJP), with Discover Geoscience (Springer) as a backup. Co-authored with a supervisor.

# Spam Detection & Prediction — Streamlit App

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

NLP-powered spam detection app that classifies messages as spam or legitimate in real time — built with scikit-learn text classifiers and deployed as an interactive Streamlit web application.

---

## Overview

This app uses TF-IDF vectorisation and machine learning classifiers to detect fraudulent and spam messages. Users paste any message and instantly receive a spam probability score, classification label, and the key words driving the prediction.

---

## Features

| Feature | Description |
|---------|-------------|
| Real-Time Classification | Instant spam/ham prediction on user-submitted messages |
| TF-IDF Vectorisation | Text feature extraction with n-gram support |
| Multi-Model Comparison | Naive Bayes, Logistic Regression, SVM comparison |
| Prediction Confidence | Probability score per classification |
| Key Feature Display | Top words driving the spam prediction |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP | scikit-learn (TF-IDF), NLTK |
| Machine Learning | Naive Bayes, Logistic Regression, SVM |
| App | Streamlit |
| Data | SMS Spam Collection dataset |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/SPAM-DETECTION-PREDICTION-USING-MACHINE-LEARNING-ON-STREAMLIT.git
cd SPAM-DETECTION-PREDICTION-USING-MACHINE-LEARNING-ON-STREAMLIT
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)

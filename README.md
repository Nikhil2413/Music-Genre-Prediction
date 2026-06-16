# 🎵 Music Genre Prediction

> A machine learning framework for predicting song genres from Spotify audio features — empowering singers and producers to analyze and target musical genres for their compositions.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Spotify API](https://img.shields.io/badge/Spotify-Web%20API-1DB954?logo=spotify&logoColor=white)](https://developer.spotify.com)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Datasets](#-datasets)
- [Pipeline](#-pipeline)
- [Models & Results](#-models--results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results Summary](#-results-summary)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🎯 Overview

This project applies **supervised machine learning** to classify songs into genres based on quantitative audio features extracted from the **Spotify Web API**. Six classification and regression models are evaluated on a dataset of **32,834 tracks** across **6 major genres** (EDM, Latin, Pop, R&B, Rap, Rock), with the goal of creating a practical tool for music creators to understand and target specific genres.

The best-performing model achieves **88.9% accuracy**, demonstrating that audio features alone can reliably distinguish between popular music genres.

---

## 💡 Motivation

Singers and music producers often struggle to identify the genre of their own creations or to deliberately craft songs that fit a target genre. By building a predictive model based on well-known audio features (danceability, energy, valence, tempo, etc.), this project provides:

- **Genre identification** — given a set of audio features, what genre does the song belong to?
- **Targeted production** — producers can adjust specific audio characteristics to move toward a desired genre
- **Viral music analysis** — understanding the audio fingerprint of popular songs across genres

---

## 📊 Datasets

| Dataset | Source | Rows | Columns | Target | Description |
|---|---|---|---|---|---|
| `spotify_songs.csv` | Spotify Playlists | 32,834 | 23 | `playlist_genre` (6 classes) | Primary dataset for model training |
| `spotify-tracks-dataset-detailed.csv` | Spotify Tracks | 114,001 | 20 | `track_genre` (114 classes) | Supplementary dataset (1000 tracks per genre) |
| `final_music_dataset.csv` | Preprocessed | 38,715 | 11 | — | Numeric features for inference |

### Features used

The models are trained on the following **Spotify audio features**:

| Feature | Description | Range |
|---|---|---|
| `danceability` | How suitable a track is for dancing | 0.0 – 1.0 |
| `energy` | Perceptual intensity and activity | 0.0 – 1.0 |
| `key` | Musical key of the track | 0 – 11 |
| `loudness` | Overall loudness in decibels (dB) | -60 – 0 |
| `mode` | Modality (0 = minor, 1 = major) | 0 or 1 |
| `speechiness` | Presence of spoken words | 0.0 – 1.0 |
| `acousticness` | Confidence the track is acoustic | 0.0 – 1.0 |
| `instrumentalness` | Amount of vocal content | 0.0 – 1.0 |
| `liveness` | Presence of audience | 0.0 – 1.0 |
| `valence` | Musical positiveness | 0.0 – 1.0 |
| `tempo` | Beats per minute (BPM) | 0 – 250 |
| `duration_ms` | Track length in milliseconds | — |

### Target genres

| Genre | Samples | Subgenres |
|---|---|---|
| EDM | 6,043 | big room, deep house, electro house, progressive house, techno, trap, trance |
| Rap | 5,746 | hip hop, trap, gangster rap, hardcore rap |
| Pop | 5,507 | dance pop, post-teen pop |
| R&B | 5,431 | contemporary R&B, neo soul |
| Latin | 5,155 | latin pop, reggaeton, latin hip hop, latin rock, salsa, bachata |
| Rock | 4,951 | classic rock, hard rock, indie rock, alternative rock, modern rock |

---

## 🔬 Pipeline

```
                         ┌─────────────────────┐
                         │   spotify_songs.csv   │
                         └──────────┬──────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │      Data Preprocessing       │
                    │  • Drop non-feature columns    │
                    │  • Remove null values          │
                    │  • Label encode target genre   │
                    │  • One-hot encode categoricals │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      Feature Scaling          │
                    │  • MinMaxScaler (NB, LR)      │
                    │  • StandardScaler (SVM)       │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │     Train / Test Split        │
                    │       80% — 20%               │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      Model Training           │
                    │  ┌────────────────────────┐   │
                    │  │ • Random Forest        │   │
                    │  │ • K-Nearest Neighbors  │   │
                    │  │ • SVM (RBF kernel)     │   │
                    │  │ • Naive Bayes          │   │
                    │  │ • Linear Regression    │   │
                    │  └────────────────────────┘   │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │        Evaluation             │
                    │  Accuracy • Precision • F1    │
                    │  Confusion Matrix • Report    │
                    └──────────────────────────────┘
```

---

## 🤖 Models & Results

### Performance Comparison

| Model | Accuracy | Precision (weighted) | F1-Score (weighted) |
|---|---|---|---|
| **Naive Bayes** 🏆 | **88.9%** | **83.9%** | **85.7%** |
| Random Forest | 81.6% | 87.0% | 78.6% |
| SVM (RBF) | 41.5% | 47.4% | 31.4% |
| KNN (k=5) | 32.5% | 30.8% | 30.4% |
| Linear Regression (R²) | — | — | 0.992 / 0.305 |

### Best Model: Multinomial Naive Bayes

| Genre | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| EDM | 0.86 | 1.00 | 0.92 | 66 |
| Latin | 1.00 | 0.74 | 0.85 | 27 |
| Pop | 0.98 | 1.00 | 0.99 | 42 |
| R&B | 1.00 | 0.89 | 0.94 | 27 |
| Rap | 0.00 | 0.00 | 0.00 | 16 |
| Rock | 0.80 | 1.00 | 0.89 | 56 |

> **Key Insight:** The model perfectly identifies EDM, Pop, and Rock. Latin and R&B perform strongly. Rap is consistently the hardest genre to classify — attributed to small test sample size and feature overlap with other genres.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-genre-prediction.git
cd music-genre-prediction

# Install dependencies
pip install pandas scikit-learn jupyter notebook

# Launch the notebook
jupyter notebook Music_Genre_Prediction.ipynb
```

### Requirements

- Python 3.11+
- pandas
- scikit-learn
- Jupyter Notebook / Google Colab

---

## 🚀 Usage

1. **Open** `Music_Genre_Prediction.ipynb` in Jupyter Notebook or Google Colab
2. **Run** all cells to execute the full pipeline — from data loading to model evaluation
3. **To make a prediction**, use the Linear Regression inference section at the end of the notebook:

```python
# Example: Predict genre from audio features
features = [0.75, 0.80, -5.0, 0.65, 120.0, 240000]
# Returns: 'rock'
```

4. **Swap datasets** — replace `spotify_songs.csv` with `spotify-tracks-dataset-detailed.csv` for a 114-genre classification challenge

---

## 📈 Results Summary

- **Naive Bayes** achieves the highest accuracy (**88.9%**) with strong precision and recall across most genres
- **Random Forest** offers the best precision (**87.0%**) and is more robust for genres with limited samples
- **Rap genre** is poorly predicted by all models — potential causes include:
  - Low representation in the test set (only 16 samples)
  - High feature similarity between rap and other genres (especially pop and R&B)
  - Need for additional features (e.g., lyrics, rhythm patterns)
- **SVM and KNN** underperform significantly, suggesting the feature space is not linearly separable and the data has high dimensionality after one-hot encoding

---

## ⚠️ Limitations

- **Genre labels** come from playlist assignments, which can be subjective and noisy
- **Rap prediction** is effectively unusable — needs more data or feature engineering
- **Linear Regression** is used as a classifier (predicting discrete labels via rounding), which is statistically unconventional
- No **hyperparameter tuning** was performed — models use mostly default parameters
- No **cross-validation** was applied beyond a single 80/20 split
- The secondary dataset (114 genres) remains **unused** in the current analysis

---

## 🔮 Future Work

- [ ] Perform hyperparameter optimization (GridSearchCV / RandomizedSearchCV)
- [ ] Implement deep learning models (MLP, CNN on spectrograms, or Transformer-based audio embeddings)
- [ ] Incorporate **lyrical analysis** via NLP for improved rap genre classification
- [ ] Build a **web application** (Flask/Streamlit) for real-time genre prediction
- [ ] Expand to the **114-genre dataset** for fine-grained classification
- [ ] Add **cross-validation** and statistical significance testing
- [ ] Create an **API endpoint** for programmatic genre prediction

---

## 📄 Research Paper

A comprehensive research paper detailing the methodology, experimental results, and analysis is included in this repository:

- [`ResearchPaper.pdf`](./ResearchPaper.pdf) — Formal academic paper
- [`ML Powered Song Genre Prediction A Tool for Singers to Create Viral Music.docx`](./ML%20Powered%20Song%20Genre%20Prediction%20A%20Tool%20for%20Singers%20to%20Create%C2%A0Viral%C2%A0Music.docx) — Detailed project write-up

---

## 🧰 Tech Stack

| Domain | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest, KNN, SVM, Naive Bayes, Linear Regression) |
| Development | Jupyter Notebook, Google Colab |
| Data Source | Spotify Web API |

---

## 📬 Contact

For questions, suggestions, or collaboration, feel free to reach out or open an issue on this repository.

---

<p align="center">
  Built with ❤️ for music creators and data scientists alike
</p>

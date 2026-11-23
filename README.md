# NLP Support Ticket Classification

A multi-class text classification project exploring how text preprocessing techniques affect the performance of machine learning models on IT support tickets.

## Overview

This project uses Natural Language Processing (NLP) and machine learning to automatically classify short IT support messages into **14 predefined categories**, such as login issues, payment problems and account recovery.

Using a dataset of 308 support messages, I developed and compared two text preprocessing pipelines, followed by TF-IDF feature extraction and supervised classification.

## Project Objectives

- Build an automated support ticket classification system.
- Compare stemming and lemmatisation-based preprocessing approaches.
- Investigate the effect of domain-specific text normalisation.
- Evaluate classification performance across multiple ticket categories.

## Methodology

### 1. Text Preprocessing

Two pipelines were implemented:

| Pipeline | Techniques |
|---|---|
| Pipeline 1 – Stemming | Lowercasing, punctuation removal, tokenisation, stopword removal and Porter stemming |
| Pipeline 2 – Lemmatisation | Lemmatisation, domain-specific abbreviation normalisation and text cleaning |

Pipeline 2 also included normalisation of common abbreviations, such as `pls` to `please`, `acct` to `account` and `pw` to `password`.

### 2. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert text into numerical features.

- Pipeline 1: Unigram TF-IDF.
- Pipeline 2: Unigram and bigram TF-IDF with sublinear term frequency scaling.

### 3. Model Development

Logistic Regression was used as the main classifier to compare the two preprocessing approaches.

- Stratified 80:20 train-test split.
- Class weighting to address class imbalance.
- GridSearchCV with five-fold stratified cross-validation.
- Macro-F1 scoring for hyperparameter selection.

Multinomial Naive Bayes was also evaluated using the Pipeline 2 features.

## Results

The main evaluation metric was **macro-F1**, which gives equal importance to each class and is useful for assessing performance on an imbalanced dataset.

| Pipeline | Model | Macro-F1 |
|---|---|---:|
| Pipeline 1 – Stemming | Logistic Regression | 0.87 |
| Pipeline 2 – Lemmatisation | Logistic Regression | **0.90** |
| Pipeline 2 – Lemmatisation | Multinomial Naive Bayes | 0.83 |

Pipeline 2 achieved the strongest macro-F1 score among the evaluated models.

Confusion matrices and classification reports were also used to examine class-level performance and identify misclassifications.

## Key Findings

- Lemmatisation combined with domain-specific normalisation and unigram-bigram features achieved the best overall macro-F1.
- Logistic Regression outperformed Multinomial Naive Bayes in the evaluated experiments.
- Some semantically similar categories, particularly login issues, account recovery and password resets, were difficult to distinguish.
- Less frequent categories presented additional classification challenges.

## Technologies Used

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **NLP:** NLTK
- **Machine Learning:** Scikit-learn
- **Visualisation:** Matplotlib
- **Environment:** Jupyter Notebook

## Repository Structure

```text
nlp-support-ticket-classification/
├── notebooks/
│   ├── pipeline_1_stemming.ipynb
│   ├── pipeline_2_lemmatisation.ipynb
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/BhavanaTR03/nlp-support-ticket-classification.git
cd nlp-support-ticket-classification
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks using Jupyter:

```bash
jupyter notebook
```

The notebooks download the required NLTK resources during execution.

## Limitations and Future Work

The project uses a relatively small dataset of 308 support messages, which limits how well the findings may generalise to larger real-world support systems.

Potential extensions include:

- Evaluating transformer-based models such as BERT.
- Exploring additional classification algorithms.
- Expanding the dataset to improve coverage of underrepresented categories.
- Developing an API for automated ticket classification.

  
A further improvement would be to integrate TF-IDF vectorisation and classification into a single Scikit-learn Pipeline, ensuring feature extraction is fitted independently within each cross-validation fold.

---

**Project context:** Academic coursework – University of Bath, MSc Data Science.

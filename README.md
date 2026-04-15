# NLP Support Ticket Classification System

## Overview
This project builds a machine learning system to automatically classify IT support tickets into predefined categories such as login issues, billing queries, and account recovery.

## Problem Statement
Support teams receive large volumes of user messages daily. Manually categorising these tickets is time-consuming and inefficient.

This project aims to automate ticket classification using natural language processing techniques.

## Approach

### Preprocessing
Two preprocessing pipelines were compared:
- Lemmatisation-based pipeline
- Domain-specific text normalisation pipeline

### Feature Engineering
- TF-IDF vectorisation was used to convert text into numerical features

### Model
- Logistic Regression classifier used for fair comparison between pipelines

## Evaluation
Model performance was evaluated using:
- Precision
- Recall
- F1-score
- Confusion Matrix

This ensured performance was assessed across all classes, not just overall accuracy.

## Key Results
- Improved classification performance through preprocessing optimisation
- Better handling of noisy, real-world support text
- Clear differences observed between preprocessing strategies

## Key Learnings
- Preprocessing choices significantly impact NLP performance
- Domain-aware normalisation improves real-world robustness
- Confusion matrices are essential for multi-class evaluation

## Technologies Used
- Python
- scikit-learn
- Pandas
- NumPy
- NLTK

## Future Improvements
- Try SVM and advanced classifiers
- Explore transformer-based models (BERT)
- Deploy as a simple API or web app

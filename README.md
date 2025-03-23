## Sentiment Analysis on Movie Reviews

This project demonstrates how to build a **sentiment analysis model** using **Logistic Regression** and **NLP techniques** to classify movie reviews as positive or negative. The model is trained on the **IMDB Movie Reviews dataset** and achieves an accuracy of **89%**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Results](#results)

---

## Project Overview
The goal of this project is to classify movie reviews from the **IMDB Movie Reviews dataset** into one of two sentiment categories:
- **Positive**
- **Negative**

The project involves the following steps:
1. **Text Preprocessing**: Cleaning and preparing the text data using **NLTK**.
2. **Feature Extraction**: Converting text into numerical features using **TF-IDF Vectorization**.
3. **Model Training**: Building and training a **Logistic Regression** model.
4. **Evaluation**: Evaluating the model's performance on the test set.

---

## Dataset
The **IMDB Movie Reviews dataset** contains 50,000 movie reviews labeled as positive or negative. The dataset is split into:
- **Training data**: 25,000 reviews
- **Test data**: 25,000 reviews

Each review is a text string, and the sentiment is a binary label (`positive` or `negative`).

---

## Model Architecture
The sentiment analysis model is built using the following steps:

1. **Text Preprocessing**:
   - Remove HTML tags, special characters, and numbers.
   - Convert text to lowercase.
   - Remove stopwords using **NLTK**.

2. **Feature Extraction**:
   - Convert text into numerical features using **TF-IDF Vectorization**.

3. **Model Training**:
   - Train a **Logistic Regression** model on the TF-IDF features.

4. **Evaluation**:
   - Evaluate the model's performance using accuracy, precision, recall, and F1-score.

---

## Results
The model achieves the following performance metrics:
- **Accuracy**: 89%
- **Precision**: 0.89
- **Recall**: 0.89
- **F1-Score**: 0.89

Below is the confusion matrix for the model:

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------|-------------------|
| **Actual Negative** | 11,000            | 1,500             |
| **Actual Positive** | 1,200             | 11,300            |

---

### Project Title

Automated Feedback Analysis for Public Relations Insights

### Brief One Line Summary

An NLP project that uses Naïve Bayes and Logistic Regression to perform sentiment analysis on customer reviews, automating feedback analysis for public relations.

### Overview

This project leverages Natural Language Processing (NLP) and supervised machine learning to build a sentiment analysis model.By training on a dataset of customer product reviews, the model learns to classify text as positive or negative, providing a scalable solution for the Public Relations department to monitor customer satisfaction and brand perception automatically.

### Problem Statement
A multinational corporation's Public Relations department has collected a massive volume of customer feedback in the form of text-based product reviews. The primary challenge is the inability to manually process this data at scale to get a timely understanding of customer sentiment. This project aims to solve this by creating a predictive model to automatically classify reviews, enabling the team to quickly identify if customers are satisfied or not.

### Dataset

The project utilizes a dataset of customer product reviews where the primary feature is the raw text of the review. The goal is to predict the sentiment from this text. This project is based on a proprietary dataset from a private course. The data file is included in this repository for reproducibility.

### Tools and Technologies

* **Python**, **Pandas** for data cleaning and manipulation.
* **NLTK** for Natural Language Processing tasks.
* **scikit-learn** for implementing classification models and feature extraction.
* **Matplotlib** & **Seaborn** for exploratory data analysis (e.g., WordCloud).
* **Colab Notebook** for the development environment.

### Methods

* **Exploratory Data Analysis (EDA):** Performed initial analysis and created visualizations like WordClouds to understand the text data.
* **Text Preprocessing:** Utilized the **NLTK** library to perform **tokenization**, breaking down review sentences into individual words or tokens for analysis.
* **Feature Extraction:** Converted the processed text data into numerical vectors using the **Count Vectorizer** technique from `scikit-learn`.
* **Model Training:** Trained and implemented two supervised learning models to classify sentiment: a **Naïve Bayes classifier** and a **Logistic Regression** model.
* **Model Evaluation:** Assessed model performance using standard metrics including the confusion matrix, precision, recall, and F1-score.

### Key Insights

* NLP provides a powerful framework for extracting meaningful, structured information from unstructured text data like customer reviews.
* Feature extraction techniques like Count Vectorizer are essential for translating qualitative text into a quantitative format that machine learning models can interpret.
* Probabilistic classifiers like Naïve Bayes serve as a strong baseline and are often highly effective for text-based classification tasks.

### Dashboard/Model/Output

* Python scripts and notebooks detailing the end-to-end data processing, training, and evaluation pipeline.
* A trained machine learning model capable of predicting the sentiment of new, unseen product reviews.
* Visualizations from the EDA phase that highlight the most frequent and important terms used in customer feedback.

### How to Run This Project?

To open any notebook from this repository directly in Google Colab:
1.  Copy the notebook's GitHub URL.
2.  Replace `github.com` with `githubtocolab.com` in the URL.
3.  Paste the modified URL into your browser and press Enter; the notebook will open in Colab, ready to execute.

### Results & Conclusion

* The trained models successfully classify customer sentiment from text reviews with measurable accuracy.
* The solution provides the PR department with a scalable and automated tool, replacing a slow, manual feedback analysis process.
* This project serves as a practical demonstration of how NLP can be applied to solve real-world business problems and generate actionable insights from text data.

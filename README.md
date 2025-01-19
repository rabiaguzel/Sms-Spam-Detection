# SMS Spam Detection with Machine Learning

This project focuses on detecting spam messages in SMS data using various machine learning models. The project uses the **SMS Spam Collection** dataset, which contains a collection of labeled SMS messages, either as "spam" or "ham" (non-spam). The objective is to classify SMS messages into these two categories based on the text content.

## Dataset

The dataset used in this project is the **SMS Spam Collection** dataset, which is publicly available on Kaggle. It consists of **5,574** SMS messages with a mix of spam and ham messages. Each message is labeled as either "ham" (non-spam) or "spam". 

You can access the dataset on Kaggle here:
- [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Dataset Features:
- **label**: The label for each message (`spam` or `ham`).
- **message**: The actual text message.

## Project Overview

In this project, we apply several machine learning models to classify SMS messages as spam or ham. The models tested include:

1. **Naive Bayes**  
2. **Logistic Regression**  
3. **Random Forest**  
4. **LSTM (Deep Learning)**  

Additionally, techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** were used to address class imbalance.

## Models & Techniques

### 1. **Naive Bayes**
   - A probabilistic classifier that uses Bayes’ theorem.
   - Great for text classification problems like spam detection.

### 2. **Logistic Regression**
   - A regression-based approach that can be used for binary classification tasks like this one.

### 3. **Random Forest**
   - An ensemble learning method that combines multiple decision trees to improve accuracy.

### 4. **LSTM (Long Short-Term Memory)**
   - A type of Recurrent Neural Network (RNN) that works well with sequential data like text.
   - This model helps capture long-term dependencies in the messages for more accurate classification.

### 5. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - A technique to balance the dataset by generating synthetic samples for the minority class (spam in this case).

## Project Workflow

1. **Data Preprocessing**
   - Load the dataset and clean the text (removing special characters, converting to lowercase, etc.).
   - Tokenize the messages and transform the text into numerical features using techniques like **TF-IDF** (Term Frequency-Inverse Document Frequency).
   
2. **Model Training and Evaluation**
   - Train multiple models and evaluate their performance using common metrics like **accuracy**, **precision**, **recall**, and **f1-score**.
   - For unbalanced data, SMOTE was applied to generate synthetic samples for the minority class to improve model performance.

3. **Results Comparison**
   - Evaluate the models using various metrics and compare their performance.
   - The Random Forest model achieved the best results, outperforming others in terms of precision and recall for the spam class.

## Requirements

- Python 3.x
- `pandas`
- `numpy`
- `sklearn`
- `imbalanced-learn` (for SMOTE)
- `tensorflow` (for LSTM model)

## Results

### Model Performance on Test Set:
The models were evaluated on a test set, and here are the results:

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Naive Bayes        | 97%      | 97%       | 100%   | 98%      |
| Logistic Regression| 96%      | 97%       | 71%    | 82%      |
| Random Forest      | 98%      | 99%       | 85%    | 92%      |
| LSTM               | 87%      | 0%        | 0%     | 0%       |

**Note:** The LSTM model performed poorly in this case, which might be due to the simple nature of the dataset, requiring a more complex architecture or better hyperparameter tuning.


## Acknowledgments

- **UCI Machine Learning Repository** for providing the SMS Spam Collection dataset.
- **Kaggle** for hosting the dataset and offering a platform for data science competitions and collaboration.
  

# Fake News Detection using Machine Learning

A machine learning project for university that uses Multinomial Naive Bayes with TF-IDF vectorization to classify news articles as fake or real, achieving ~95% accuracy through feature engineering and hyperparameter optimization.

## 📊 Project Overview

This project implements a binary classification system to detect fake news articles using natural language processing (NLP) and machine learning techniques. The model analyzes textual content combined with selected numerical metadata features to make predictions.

### Key Features

- **TF-IDF Vectorization** for text representation
- **Feature Engineering** with 4 selected numerical features (weak predictors dropped after EDA)
- **Hyperparameter Tuning** using GridSearchCV with 5-fold cross-validation
- **Comprehensive EDA** with correlation analysis and visualizations
- **Performance Metrics** including confusion matrix and classification report

## 🎯 Performance Metrics

The model achieves strong performance on the test set:

- **Accuracy:** ~95%
- **Precision:** 95% for both classes
- **Recall:** 96% fake / 95% real
- **F1-Score:** 96% fake / 95% real

## 📁 Project Structure

```
├── algorithm.py   # Main classification model
├── graphics.py    # EDA and visualizations
├── Fake csv       # Fake news dataset
├── True.csv       # Real news dataset
└── README.md      # Project documentation
```
## 🔧 Technologies Used

- **Python**
- **scikit-learn** - ML algorithms and preprocessing
- **pandas** - Data manipulation
- **matplotlib** & **seaborn** - Visualizations
- **TextBlob** - Sentiment analysis
- **scipy** - Sparse matrix operations

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/gurowamuki/fake-news-detection.git
cd fake-news-detection

# Install required packages
pip install pandas scikit-learn matplotlib seaborn textblob scipy
```
## 🚀 Usage

### 1. Run Exploratory Data Analysis

```bash
python graphics_.py
```

This will:
- Display dataset statistics and info
- Show correlation matrix
- Generate visualizations (histograms, boxplots, scatter plots)
- Analyze missing values

### 2. Train and Evaluate the Model

```bash
python algorithm_.py
```

This will:
- Load and preprocess the data
- Perform hyperparameter tuning
- Train the best model
- Display classification report and confusion matrix

## 🧪 Methodology

### Data Preprocessing

1. **Data Loading:** Combine Fake.csv and True.csv with appropriate labels
2. **Missing Value Handling:** Remove rows with null values
3. **Feature Engineering:** Create 8 candidate features, then select the 4 strongest predictors
4. **Normalization:** Scale selected numerical features using MinMaxScaler

### Feature Engineering

The model uses the following features:

**Textual Features (TF-IDF):**
- Combined title and text content
- Top 5000 words by TF-IDF score


**Numerical Features used in model (selected by correlation with label):**
 
| Feature | Correlation with label | Notes |
|---|---|---|
| `title_length` | -0.58 | Strong — fake articles have longer titles |
| `year` | -0.57 | Strong — fake articles cluster in earlier years |
| `title_exclamations` | -0.25 | Moderate — fake articles use more `!` in titles |
| `text_exclamations` | -0.23 | Moderate — fake articles use more `!` in text |
 
**Dropped features (weak predictors):**
 
| Feature | Correlation with label | Reason dropped |
|---|---|---|
| `text_length` | -0.05 | Near-zero correlation |
| `text_sentiment` | -0.03 | Near-zero correlation |
| `title_sentiment` | +0.04 | Near-zero correlation |
| `title_text_ratio` | -0.12 | Low discriminative power |
 
> Dropping weak features reduces model complexity and avoids noise without sacrificing accuracy.

### Training Process

1. **Data Split:** 80% training, 20% testing (stratified)
2. **Cross-Validation:** 5-fold CV on training set
3. **Scoring Metric:** F1-score (balances precision and recall)
4. **Optimization:** GridSearchCV tests all alpha values `[0.1, 0.5, 1.0, 1.5, 2.0]`
5. **Final Model:** Best estimator selected based on CV performance

## 📈 Exploratory Data Analysis Insights

### Key Findings

### Dataset
 
- **Total entries:** 44,898 (no missing values)
- **Fake (label=0):** 23,481 articles — 52%
- **Real (label=1):** 21,417 articles — 48%
- The dataset is slightly imbalanced but still suitable for balanced training.
 
### Key Findings from Correlation Analysis
 
- `title_length` and `year` are the strongest predictors of fake news
- Fake articles use noticeably more exclamation marks in both title and body
- Sentiment scores and raw text length have negligible predictive power

### Visualizations Generated

1. **Label Distribution** - Bar chart showing class balance
2. **Text Length Distribution** - Histogram by label
3. **Title Length Distribution** - Histogram by label
4. **Boxplots** - Length distributions by category
5. **Sentiment Distribution** - Histogram of polarity scores
6. **Scatter Plot** - Text vs. title length relationship
7. **Exclamation Analysis** - Average counts by label
8. **Correlation Heatmap** - Feature relationships

## 📊 Results Interpretation

### Classification Report Metrics

- **Precision:** Of all predicted fake/real, what % were correct?
- **Recall:** Of all actual fake/real, what % did we catch?
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of samples in each class

### Confusion Matrix

```
           Predicted
           Fake  True
Actual Fake  TN    FP
       True  FN    TP
```

- **True Negatives (TN):** Correctly identified fake news
- **True Positives (TP):** Correctly identified real news
- **False Positives (FP):** Real news misclassified as fake
- **False Negatives (FN):** Fake news misclassified as real

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **Machine Learning Pipeline:** Data loading → preprocessing → training → evaluation
2. **Feature Selection:** Dropping weak predictors based on correlation analysis
3. **NLP Techniques:** TF-IDF, text preprocessing, sentiment analysis
4. **Model Optimization:** Hyperparameter tuning with cross-validation
5. **Data Visualization:** Comprehensive EDA with multiple plot types
6. **Best Practices:** Code organization, reproducibility (`random_state=42`), documentation

## 📚 Dataset Information

**Source:** Kaggle Fake News Detection datasets
- `Fake.csv` - Collection of fake news articles
- `True.csv` - Collection of verified real news articles

**Features:**
- `title` - Article headline
- `text` - Article body content
- `subject` - News category
- `date` - Publication date

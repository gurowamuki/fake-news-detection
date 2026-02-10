# Fake News Detection using Machine Learning

A comprehensive machine learning project for university that uses Multinomial Naive Bayes with TF-IDF vectorization to classify news articles as fake or real, achieving high accuracy through feature engineering and hyperparameter optimization.

## 📊 Project Overview

This project implements a binary classification system to detect fake news articles using natural language processing (NLP) and machine learning techniques. The model analyzes both textual content and metadata features to make predictions.

### Key Features

- **TF-IDF Vectorization** for text representation
- **Feature Engineering** with 8 custom features
- **Hyperparameter Tuning** using GridSearchCV with 5-fold cross-validation
- **Comprehensive EDA** with correlation analysis and visualizations
- **Performance Metrics** including confusion matrix and classification report

## 🎯 Performance Metrics

The model achieves strong performance on the test set:

- **Accuracy:** ~99%
- **F1-Score:** Optimized through hyperparameter tuning
- **Best Alpha Parameter:** Determined via GridSearchCV

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
3. **Feature Engineering:** Create 8 custom features
4. **Normalization:** Scale numerical features using MinMaxScaler/StandardScaler

### Feature Engineering

The model uses the following features:

**Textual Features (TF-IDF):**
- Combined title and text content
- Top 5000 words by TF-IDF score


**Numerical Features:**
1. `text_length` - Word count in article body
2. `title_length` - Word count in title
3. `title_exclamations` - Number of '!' in title
4. `text_exclamations` - Number of '!' in text
5. `year` - Publication year extracted from date
6. `text_sentiment` - Sentiment polarity score (-1 to 1)
7. `title_sentiment` - Title sentiment polarity
8. `title_text_ratio` - Ratio of title to text length

### Training Process

1. **Data Split:** 80% training, 20% testing (stratified)
2. **Cross-Validation:** 5-fold CV on training set
3. **Scoring Metric:** F1-score (balances precision and recall)
4. **Optimization:** GridSearchCV tests all alpha combinations
5. **Final Model:** Best estimator selected based on CV performance

## 📈 Exploratory Data Analysis Insights

### Key Findings

**Correlation Analysis:**
- Text/title length show moderate correlation with label
- Sentiment scores reveal differences between fake/real news
- Exclamation usage patterns differ between categories

**Distribution Analysis:**
- Balanced dataset (roughly equal fake/real samples)
- Text length varies significantly between categories
- Sentiment distributions show distinct patterns

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
2. **Feature Engineering:** Creating meaningful features from raw data
3. **NLP Techniques:** TF-IDF, text preprocessing, sentiment analysis
4. **Model Optimization:** Hyperparameter tuning with cross-validation
5. **Data Visualization:** Comprehensive EDA with multiple plot types
6. **Best Practices:** Code organization, reproducibility (random_state), documentation

## 📚 Dataset Information

**Source:** Kaggle Fake News Detection datasets
- `Fake.csv` - Collection of fake news articles
- `True.csv` - Collection of verified real news articles

**Features:**
- `title` - Article headline
- `text` - Article body content
- `subject` - News category
- `date` - Publication date

## 👤 Contact

**gurowamuki**
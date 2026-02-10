import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob  # For sentiment analysis
from sklearn.preprocessing import StandardScaler

sns.set_theme()  # Set style

# Reading data from files
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Ensure consistent labels (0 for Fake, 1 for True)
fake_df['label'] = 0
true_df['label'] = 1

# Combine datasets and remove rows with missing values
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.dropna()

# Add basic features
df['text_length'] = df['text'].str.split().str.len().fillna(0)  # Word count
df['title_length'] = df['title'].str.split().str.len().fillna(0)

# Add new features for correlation matrix
# Sentiment scores for text and title
df['text_sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity) #  polarity give a score from -1 to 1
df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity) #  lambda is a function that takes a single text value as input

# Count exclamation marks in text and title
df['text_exclamations'] = df['text'].str.count('!').fillna(0)
df['title_exclamations'] = df['title'].str.count('!').fillna(0)

# Title-to-text length ratio:
df['title_text_ratio'] = df['title_length'] / (df['text_length'] + 1)  # Add 1 to avoid division by zero

# Extract year from date (assuming date format like 'December 20, 2017')
df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year.fillna(0).astype(int) #  errors='coerce' tells to put NaT instead of error

# Standardize numerical features
numerical_features = ['text_length', 'title_length', 'text_sentiment', 'title_sentiment',
                      'text_exclamations', 'title_exclamations', 'title_text_ratio', 'year']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# STATISTIC!
print("----------------------BASIC-INFO--------------------------------------")
print("Dataset info:")
print(df.info())
print("-----------------------------------")

print("Numerical Features:")
print(df[numerical_features].describe())
print("-----------------------------------")

print("Subject quantity:")
print(df['subject'].value_counts())
print("-----------------------------------")

print("Label quantity:")
print(df['label'].value_counts())
print("-----------------------------------")

# MISSING VALUE ANALYSIS!
print("----------------------MISSING-VALUE-ANALYSIS--------------------------------------")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing values': missing_values,
    'Percentage': missing_percentage
})
print(missing_df)

# CORRELATION MATRIX
print("----------------------CORRELATION-MATRIX----------------------------------------------")
corr_matrix = df[numerical_features + ['label']].corr()
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# VISUALIZATION!
# Bar chart for labels (0=Fake, 1=True)
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color=['orange', 'blue'])
plt.title('Label Distribution (0=Fake, 1=True)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Histogram for text length (standardized)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='text_length', hue='label', bins=40, palette=['orange', 'blue'])
plt.title('Text Length (Standardized Word Count) Distribution')
plt.xlabel('Text Length (Standardized)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Histogram for title length (standardized)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='title_length', hue='label', bins=40, palette=['orange', 'blue'])
plt.title('Title Length (Standardized Word Count) Distribution')
plt.xlabel('Title Length (Standardized)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Boxplot for text length
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='label', y='text_length', hue='label', palette=['orange', 'blue'], legend=False)
plt.title('Text Length (Standardized) by Label')
plt.xlabel('Label (0=Fake, 1=True)')
plt.ylabel('Text Length (Standardized)')
plt.tight_layout()
plt.show()

# Boxplot for title length
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='label', y='title_length', hue='label', palette=['orange', 'blue'], legend=False)
plt.title('Title Length (Standardized) by Label')
plt.xlabel('Label (0=Fake, 1=True)')
plt.ylabel('Title Length (Standardized)')
plt.tight_layout()
plt.show()

# Histogram for text sentiment
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='text_sentiment', hue='label', bins=40, palette=['orange', 'blue'])
plt.title('Text Sentiment (Standardized) Distribution')
plt.xlabel('Text Sentiment (Standardized)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Scatter plot of text length vs. title length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='text_length', y='title_length', hue='label', palette=['orange', 'blue'])
plt.title('Text Length vs. Title Length (Standardized)')
plt.xlabel('Text Length (Standardized)')
plt.ylabel('Title Length (Standardized)')
plt.tight_layout()
plt.show()

# Bar chart for average exclamation counts
exclamation_means = df.groupby('label')[['text_exclamations', 'title_exclamations']].mean()
exclamation_means.plot(kind='bar', figsize=(8, 5), color=['#ff9999', '#66b3ff'])
plt.title('Average Exclamation Counts by Label')
plt.xlabel('Label (0=Fake, 1=True)')
plt.ylabel('Average Count (Standardized)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
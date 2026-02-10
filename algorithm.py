import pandas as pd
from sklearn.preprocessing import MinMaxScaler #  Scales numerical values to a (0,1) rang
from sklearn.feature_extraction.text import TfidfVectorizer #  Convert text into numerical features using TF-IDF
from sklearn.model_selection import train_test_split, GridSearchCV #   For hyperparameter tuning using cross-validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay #  used to make a report
import scipy.sparse as sp #  used to handle sparse matrix (contains mostly 0)
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data from files
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# 0 for Fake, 1 for True
fake_df['label'] = 0
true_df['label'] = 1

# Combine datasets and remove rows with missing values
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.dropna()

# Add selected features
df['text_length'] = df['text'].str.split().str.len().fillna(0)  # Word count
df['title_length'] = df['title'].str.split().str.len().fillna(0)
df['title_exclamations'] = df['title'].str.count('!').fillna(0)
df['text_exclamations'] = df['title'].str.count('!').fillna(0)

# Extract year from date (assuming date format like 'December 20, 2017')
df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year.fillna(0)

# Standardize numerical features
numerical_features = ['title_length', 'title_exclamations', 'text_exclamations', 'year']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Combine text and title for TF-IDF
df['combined'] = df['title'] + " " + df['text']

# Vectorize text using TF-IDF: method to represent text as numbers while reflecting how important a word is
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') #  max_features=5000->keep top 5000 words with the highest TF-IDF scores; stop_word='english'-> ignore common english words; <5000 lower accuracy; >5000 overfitting
X_text = vectorizer.fit_transform(df['combined']) #  learn the vocabulary from dataset and transform into matrix of TF-IDF score

# Combine TF-IDF features with numerical features
X_numerical = df[numerical_features].values #  extract features from dataset as a numpy array
X = sp.hstack([X_text, X_numerical]) # used to handle sparse matrix
y = df['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test 20%, train 80%; random_state ensures reproducibility

# Naive Bayes with hyperparameter optimization
nb_model = MultinomialNB()
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]} #  Dictionary of parameters to test (alpha values in this case), more can underfit
#  alpha is the laplace smoothing parameter: it's avoid zero probabilities for words that didn't appear in training data;
#  If alpha = 0, and a word never appeared in real news but suddenly appears in test data, the model could give it a 0
#  probability, which can throw off predictions
grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='f1', n_jobs=-1) #  method to utomatically try multiple hyperparameter combinations and find the best one;
#  cv=5 -> splits training data into 5 folds, trains on 4, tests on 1; scoring='f1' -> uses the F1-score to evaluate model performance
#  n_jobs=-1 -> run ein parallel for speed
grid_search.fit(X_train, y_train) #  for each value of alpha perform 5-fold cross-validation and computes the average F1-score across folds
#  after stores best model and parameters
#  F1-score balanced both precision and recall

#  Best model and predictions
print("Best Parameters:", grid_search.best_params_)
best_nb_model = grid_search.best_estimator_
y_pred = best_nb_model.predict(X_test)

#  Evaluate the model
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred))

#  Confusion  Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
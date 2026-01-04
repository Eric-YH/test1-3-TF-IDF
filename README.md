# Dental Pain Detection from Clinical Notes (Traditional NLP)

Detect pain-related symptoms from clinical notes using **classic NLP** and traditional machine learning.

## Features
- Text preprocessing: tokenization, lemmatization, stop-word removal, negation handling
- TF-IDF vectorization (unigrams + bigrams)
- Classification with Logistic Regression & Linear SVM
- Model interpretability: examine key pain vs. non-pain words

## Special Tools Used
- **SpaCy** – tokenization, lemmatization, stop-word handling
- **NLTK** – optional tokenization support
- **scikit-learn** – TF-IDF, Logistic Regression, LinearSVC
- **NumPy** – numerical computations

## Learning Path
1. Basic NLP: tokenization, lemmatization, stopwords
2. Text representation: TF-IDF, n-grams
3. Classical ML: Logistic Regression, Linear SVM
4. Handle negation & domain-specific rules
5. Interpret model coefficients

## Quick Usage

```python
clean_docs = [clean_text(text) for text in raw_docs]
X = tfidf.fit_transform(clean_docs)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X, y)
X_test = tfidf.transform(test_texts)
preds = model.predict(X_test)

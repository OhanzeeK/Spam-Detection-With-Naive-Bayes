# Spam-Detection-With-Naive-Bayes
A spam detection mockup I created to test the Naive Bayes classifier in Scala/Spark for a class project.

**Objective:** Implement a Naïve Bayes classifier to distinguish spam emails from legitimate
emails using word-frequency features.

**Data Description:**
 - **Columns:**
     - email_id: The identification column for each email. String.
     - text: The contents of an email. String.
     - label: A binary column representing whether the email is spam (0 = False, 1 = True). Int.
       
 - Row Count: 1200

**Data Pipeline:**

1. **Load data:** Read naive_bayes_spam.csv into a Spark DataFrame with headers and schema inference.
2. **Clean:** Drop rows where every column is null.
3. **Tokenize:** Split each email's text column on whitespace with RegexTokenizer, lowercasing the tokens, into a words array column.
4. **Split:** Use the first half of the data (email_id <= 'EMAIL00601') for training and the second half (email_id > 'EMAIL00601') for testing.
5. **Vectorize:** Fit a CountVectorizer on the training set (vocabulary size capped at 300) to convert token arrays into term-count feature vectors.
6. **Train:** Fit a multinomial NaiveBayes model on the training features.
7. **Evaluate:** Score the test set and compute accuracy with MulticlassClassificationEvaluator.
8. **Predict custom emails:** Run five handwritten emails (labeled 1.0 for spam, 0.0 for not spam) through the same tokenizer, vectorizer, and model, then report predictions and accuracy.

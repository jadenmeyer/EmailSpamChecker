from xml.sax.handler import all_properties

import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  # make a training split
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #data analytics stuff


DOCUMENT_ONE = "emails.csv"

def extract_email_body(email_text):
    """
    :param email_text: This is a column of the dataframe in which there is an email message.
    :return: a spliced version of the message where the header is cut out.
    """
    email_body = re.search(r"\n\n", email_text)
    if email_body:
        body = email_text[email_body.end():].strip()  # splicer for characters after header text
        return body.strip()  # get rid of extra newlines and stuff
    return ""

def preprocess_email_text(email_text):
    email_text = email_text.lower()
    translation = str.maketrans('', '', string.punctuation)
    email_text = email_text.translate(translation)
    return email_text

def initial_training_spam_words(email_text):
    spam_keywords = ['free', 'win', 'offer', 'urgent', 'money', 'lottery', 'qualify', 'failure']
    # should overly redundant email_text = email_text.lower()
    return 1 if any(keyword in email_text for keyword in spam_keywords) else 0

# Read the email data
email_dataframe = pd.read_csv(DOCUMENT_ONE)

# Extract the body and preprocess the email text
email_dataframe['body'] = email_dataframe['message'].apply(extract_email_body)
email_dataframe['processed_body'] = email_dataframe['body'].apply(lambda x: preprocess_email_text(x))
email_dataframe['spam'] = email_dataframe['processed_body'].apply(initial_training_spam_words)

X = email_dataframe['processed_body']
y = email_dataframe['spam']

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train-test splitter
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

model = LogisticRegression() #create a logistic regression model object

model.fit(X_train, y_train) #tell it what data i want

y_prediction = model.predict(X_test) #make a prediction on my data

print(f'Accuracy: {accuracy_score(y_test, y_prediction)}')

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_prediction))

print('Classification Report:')
print(classification_report(y_test, y_prediction))

data_vector = tfidf_vectorizer.transform(email_dataframe['processed_body'])

probabilities = model.predict_proba(data_vector)[:, 1]

graded_emails = email_dataframe.copy()
graded_emails['predicted_spam_probability'] = probabilities
graded_emails['predicted_spam'] = (probabilities > 0.6).astype(int)

graded_emails.to_csv("graded_emails.csv", index=False)
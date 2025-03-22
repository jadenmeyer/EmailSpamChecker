import pandas as pd
import nltk as nl
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split #make a training split
from sklearn.linear_model import LogisticRegression #logistic regression
DOCUMENT_ONE = "emails.csv"

def extract_email_body(email_text):
    """
    :param email_text: This is a column of the dataframe in which there is an email message.
    :return: a spliced version of the message where the header is cut out.
    """
    email_body = re.search(r"\n\n", email_text)
    if email_body:
        body = email_text[email_body.end():].strip() #splicer for characters after header text
        return body.strip() #get rid of extra newlines and stuff
    return ""
pd.options.display.max_rows = 60 #redefine the number of rows we can print with pandas

email_dataframe = pd.read_csv(DOCUMENT_ONE)

location_one = extract_email_body(email_dataframe.loc[1, 'message'])

#print(location_one)
#print(pd.options.display.max_rows)

"""Now to Preprocess the data"""

def preprocess_email_text(email_text):
    email_text = email_text.lower()
    translation = str.maketrans('', '', string.punctuation)
    email_text = email_text.translate(translation)
    return email_text

tester = preprocess_email_text(location_one)
#print(tester)

"""Now onto Term frequency using tfdifVectorizer in sklearn"""
listy = [tester]
testing_TF = TfidfVectorizer() #object
result = testing_TF.fit_transform(listy) #send in document can send in array of documents though
#running into a type issue with fit_transform

#loc = "This is a tester EXAMPLE!!!"
#fin = preprocess_email_text(loc)
#tester = [fin]

print(result)

"""Onto Training and Testing splitting of data"""

"""Onto choosing a model
going to explicity use a logistic regression model
"""

"""
TODO:
Build up the model
analyze the model and its accuracy
polish up stuff
add a spam probablity checker
tune the model
get into a more workign form
"""

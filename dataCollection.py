import pandas as pd
import nltk as nl
import re

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

email_dataframe = pd.read_csv('emails.csv')

location_one = extract_email_body(email_dataframe.loc[1, 'message'])
print(location_one)
print(pd.options.display.max_rows)

#print(email_dataframe)

"""Now to Preprocess the data"""



import pandas
import requests
from bs4 import BeautifulSoup as bs


# Making a GET request
r = requests.get('https://www.geeksforgeeks.org/python-programming-language/')

# check status code for response received
# success code - 200
print(r)

# Parsing the HTML
soup = bs(r.content, 'html.parser')
print(soup.prettify())
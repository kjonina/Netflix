"""
Name:               Karina Jonina 
Github:             https://github.com/kjonina/
Data Gathered:      https://www.kaggle.com/shivamb/netflix-shows
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from io import StringIO # This is used for fast string concatination
import nltk # Use nltk for valid words
import collections as co # Need to make hash 'dictionaries' from nltk for fast processing
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer #Bag of Words



# The dataset used was collected from the following website:
# https://www.kaggle.com/shivamb/netflix-shows

# read the CSV file
dataset = pd.read_csv('netflix_titles.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(dataset.head())

# checking the dataset shape
print(dataset.shape)
# (6234, 12)


# prints out names of columns
print(dataset.columns)

# This tells us which variables are object, int64 and float 64. This would mean that 
# some of the object variables might have to be changed into a categorical variables and int64 to float64 
# depending on our analysis.
print(dataset.info())


# checking for missing data
dataset.isnull().sum() 

# making object into categorical variables
dataset['type'] = dataset['type'].astype('category')
dataset['country'] = dataset['country'].astype('category')
dataset['listed_in'] = dataset['listed_in'].astype('category')
dataset['rating'] = dataset['rating'].astype('category')

# checking data to check that all objects have been changed to categorical variables.
dataset.info()


# =============================================================================
# Converting 'date_added' to datetime
# =============================================================================

dataset['date_added'] = dataset['date_added'].astype('datetime64')

# Complete the call to convert the date column
# Displayed  September 9, 2019 
dataset['date_added'] =  pd.to_datetime(dataset['date_added'],
                              format='%B %d, %Y')


print(dataset.info())



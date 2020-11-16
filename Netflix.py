'''
Name:               Karina Jonina 
Github:             https://github.com/kjonina/
Data Gathered:      https://www.kaggle.com/shivamb/netflix-shows
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import calendar
from io import StringIO # This is used for fast string concatination
import nltk # Use nltk for valid words
import collections as co # Need to make hash 'dictionaries' from nltk for fast processing
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings('ignore')
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer #Bag of Words


from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

import re
from sklearn.feature_extraction.text import CountVectorizer #Bag of Words



# The df used was collected from the following website:
# https://www.kaggle.com/shivamb/netflix-shows

# read the CSV file
df = pd.read_csv('netflix_titles.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(df.head())

# checking the df shape
print(df.shape)
# (6234, 12)


# prints out names of columns
print(df.columns)

# This tells us which variables are object, int64 and float 64. This would mean that 
# some of the object variables might have to be changed into a categorical variables and int64 to float64 
# depending on our analysis.
print(df.info())


# checking for missing data
df.isnull().sum() 

# dropping null value columns to avoid errors 
# You cant create a code to autofill the string data
#df.dropna(inplace = True) 

# checking the df shape
print(df.shape)
# (3774, 12)

# making object into categorical variables
df['type'] = df['type'].astype('category')
df['country'] = df['country'].astype('category')
df['listed_in'] = df['listed_in'].astype('category')
df['rating'] = df['rating'].astype('category')

# checking data to check that all objects have been changed to categorical variables.
df.info()


# =============================================================================
# Converting 'date_added' to datetime
# =============================================================================
# changing date to dateime
df['date_added'] = pd.to_datetime(df['date_added'], utc = True)

# just getting the date
df['date_added'] = df['date_added'].dt.date

#Get the year from date_added
df['year_added'] = pd.DatetimeIndex(df['date_added']).year

# creating a new variable, examining how new are the films on Netflix
df['new_or_old'] = df['year_added'] - df['release_year']

# =============================================================================
# Splitting Movie and TV shows
# =============================================================================

movie = df[df['type'] == 'Movie']
movie.isnull().sum() 

tv_show = df[df['type'] == 'TV Show']
tv_show .isnull().sum() 


movie['duration'] = movie['duration'].str.strip(' min')
movie['duration'] = movie['duration'].astype(int)


# =============================================================================
# Examining Rating
# =============================================================================
df['rating'].unique()

# create a table
df['rating'].value_counts()


# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'rating', data = df, palette = 'viridis', order = df['rating'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Rating', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Ratings', fontsize = 14)
plt.show()


# =============================================================================
# Examining Ratings by Type
# =============================================================================

ratings_type = pd.DataFrame({'A': df['type'],
                   'B': df['rating']})

print(ratings_type)

ratings_type_df_pivot = pd.pivot_table(
        ratings_type, index = ['A'], columns = ['B'], aggfunc=len)

print(ratings_type_df_pivot)


plt.figure(figsize = (12, 8))
sns.catplot(x = 'A', hue = 'B', kind = 'count', palette = 'pastel', edgecolor = '.6', data = ratings_type)
plt.title('Breakdown of Ratings by Content Type', fontsize = 16)
plt.ylabel('Rating', fontsize = 14)
plt.xlabel('count', fontsize = 14)


plt.figure(figsize = (12, 8))
sns.countplot(y = 'B', hue = 'A', palette = 'pastel', data = ratings_type)
plt.title('Breakdown of Ratings by Content Type', fontsize = 20)
plt.ylabel('Rating', fontsize = 14)
plt.xlabel('count', fontsize = 14)

# =============================================================================
# Examining type
# =============================================================================
df['type'].unique()

# tables
df.groupby(['type']).size().sort_values(ascending=False)
# Movie      4265
# TV Show    1969

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'type', data = df, palette = 'viridis')
plt.title('Breakdown of Content Type', fontsize = 20)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Content Type', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = df['type'].value_counts().head(25).index.tolist()
sizes = df['type'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of Type', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()

# =============================================================================
# Examining release_year
# =============================================================================
df['release_year'].unique()

# create a table for the view
df.groupby(['release_year']).size().sort_values(ascending=False)


# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(y = 'release_year', data = df, palette = 'magma', order = df['release_year'].value_counts().head(25).index)
plt.title('How much fresh content does Netflix have?', fontsize = 16)
plt.ylabel('Release Year', fontsize = 14)
plt.xlabel('count', fontsize = 14)
plt.show()


# =============================================================================
# Examining how quick are the 
# =============================================================================
df['new_or_old'].unique()

df['new_or_old'].value_counts()

# create a table for the view
df.groupby(['new_or_old']).size().sort_values(ascending=False)

df.info()
## there are missing values so I decided to give it a value of -1 because our data does not have
## negative numbers and it will be easy to see if it is really NA
df['new_or_old'] = df['new_or_old'].fillna(-1)

df['new_or_old'] = df['new_or_old'].astype(int)

# Examines the ratings 
sns.set()
plt.figure(figsize = (12, 8))
sns.countplot(y = 'new_or_old', data = df, palette = 'magma', order = df['new_or_old'].value_counts().head(20).index)
plt.title('How long does it take Netflix to add content to its website?', fontsize = 20)
plt.ylabel('Difference in Years', fontsize = 14)
plt.xlabel('count', fontsize = 14)
plt.show()

# =============================================================================
# How long are most movies?
# =============================================================================
# creating a diagrams
movie['duration'].hist() 

# examining the mean and sd, max and min
movie['duration'].describe()


movie_duration = movie['duration'].value_counts()

fig = plt.figure(figsize = (20,10))
sns.lineplot(data = movie_duration)
plt.title('Movie Durations', fontsize = 20)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Duration in Minutes', fontsize = 14)
plt.show()



# trying to find shortest Movie 
if movie['duration'] == 12:
    print(movie['title'])

# =============================================================================
# How many Seasons do most TV Shows have?
# =============================================================================
# examining TV show seasons
plt.figure(figsize = (12, 8))
sns.countplot(y = 'duration', data = df, palette = 'viridis', order = tv_show['duration'].value_counts().head(25).index)
plt.title('How many Seasons do most TV Shows have?', fontsize = 20)
plt.ylabel('Seasons', fontsize = 14)
plt.xlabel('count', fontsize = 14)
plt.show()






    


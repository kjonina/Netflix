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
import calendar
from io import StringIO # This is used for fast string concatination
import nltk # Use nltk for valid words
import collections as co # Need to make hash 'dictionaries' from nltk for fast processing
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
from wordcloud import WordCloud
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
df.dropna(inplace = True) 
#show_id            0
# Type               0
# Title              0
# director        1969
# cast             570
# country          476
# date_added        11
# release_year       0
# rating            10
# duration           0
# listed_in          0
# description        0

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
tv_show = df[df['type'] == 'TV Show']

movie['duration'] = movie['duration'].str.strip(' min')
movie['duration'] = movie['duration'].astype(int)


# =============================================================================
# Examining Rating
# =============================================================================
df['rating'].unique()

# table to cretat
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
plt.xticks(rotation = 90)
plt.title('Breakdown of Type', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Type', fontsize = 14)
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
# Examining listed_in
# =============================================================================
df['listed_in'].unique()

df.groupby(['listed_in']).size().sort_values(ascending=False)
# Movie      4265
# TV Show    1969

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'listed_in', data = df, palette = 'viridis', order = df['listed_in'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Top 25 listed_in', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('listed_in', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = df['listed_in'].value_counts().head(25).index.tolist()
sizes = df['listed_in'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of listed_in', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()


# =============================================================================
# Examining release_year
# =============================================================================
df['release_year'].unique()

df.groupby(['release_year']).size().sort_values(ascending=False)
# Movie      4265
# TV Show    1969

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'release_year', data = df, palette = 'viridis', order = df['release_year'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Top 25 listed_in', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('release_year', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = df['release_year'].value_counts().head(25).index.tolist()
sizes = df['release_year'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of release_year', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()






# =============================================================================
# How long are most movies?
# =============================================================================


# creating a diagrams
movie['duration'].hist() 

# examining the mean and sd, max and min
movie['duration'].describe()

shortest_film = 12
if movie['duration'] == 12 :
    print(movie['title'])

# =============================================================================
# How many Seasons do most TV Shows have?
# =============================================================================
# examining TV show seasons
plt.figure(figsize = (12, 8))
sns.countplot(x = 'duration', data = df, palette = 'viridis', order = tv_show['duration'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('How many Seasons do most TV Shows have?', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Seasons', fontsize = 14)
plt.show()


# =============================================================================
# Preparing Genres
# =============================================================================

# new data frame with split value columns 
new_listed_in= df["listed_in"].str.split(", ", n = 6, expand = True) 
# making separate first listed_in column from new data frame 
df["first listed_in"]= new_listed_in[0]
# making separate second listed_in column from new data frame 
df["second listed_in"]= new_listed_in[1] 
# making separate third listed_in column from new data frame 
df["third listed_in"]= new_listed_in[2] 


'''
Not efficient. Need to create a loop for the function. 
got the idea from https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
'''

'''
# After many hours of (what seemed like) an impossible tasks of splitting the column hunting, 
# I ran into this code: https://www.kaggle.com/subinium/storytelling-with-data-netflix-ver
# the use of LAMBDA is important!!!!
# Most practise the use of Lamdba more!
'''


df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
Types = []
for i in df['genre']: Types += i
Types = set(Types)

print(df['genre'])

# =============================================================================
# Examining countries that produced films
# =============================================================================
df['country'].unique()

country = df['country'].value_counts().head(25)


country.head(15).plot(kind = 'barh')

# Examines the top 25 countries that have complaints
plt.figure(figsize = (12, 8))
sns.countplot(x = 'country', data = df, palette = 'viridis', order = df['country'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Countries', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Countries', fontsize = 14)
plt.show()


# =============================================================================
# Other interesting points to identify
# =============================================================================
#Upon further inspection of the data,it was noted that the following columns need to be split.
## countries,
## director
## cast,
## listed_in
#
# THere are also a lot of NULL values for director and cast members, which I will have to decided what to do with 

# do a stacked barchart by country and Type

# create a map of the world with saturation

# split duration by Seasons and minutes for Movies and TV Shows

# examine release dates
# examine release dates and titles (Christmas Movies released at Christmas etc.)
# examine the description

# run a Machine Learning Algorithm to predict Rating based on listed_in and title and description
    # so kids movies get catergorised as kids movies and adult movies are categorised as kids movies.
#Split the director and cast and analyse by them. (eg. Jennifer Aniston mostly stars in comedies)
    


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

# dropping null value columns to avoid errors 
# You cant create a code to autofill the string data
dataset.dropna(inplace = True) 
#show_id            0
#type               0
#title              0
#director        1969
#cast             570
#country          476
#date_added        11
#release_year       0
#rating            10
#duration           0
#listed_in          0
#description        0

# checking the dataset shape
print(dataset.shape)
# (3774, 12)

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



# =============================================================================
# Preparing data
# =============================================================================
'''
Upon further inspection of the data,it was noted that the following columns need to be split.
# countries,
# director
# cast,
# listed_in

Also date_added and release_year are not the same year.

THere are also a lot of NULL values for director and cast members, which I will have to decided what to do with 

'''

# =============================================================================
# Preparing Directors
# =============================================================================


# new data frame with split value columns 
new_director = dataset["director"].str.split(", ", n = 6, expand = True) 

# making separate first director column from new data frame 
dataset["first director"]= new_director[0]
# making separate second director column from new data frame 
dataset["second director"]= new_director[1] 
# making separate third director column from new data frame 
dataset["third director"]= new_director[2] 
# making separate fourth director column from new data frame 
dataset["fourth director"]= new_director[3] 
# making separate fifth director column from new data frame 
dataset["fifth director"]= new_director[4] 
# making separate fifth director column from new data frame 
dataset["sixth director"]= new_director[5] 



# new data frame with split value columns 
new_cast = dataset["cast"].str.split(", ", n = 10, expand = True) 
# making separate first cast column from new data frame 
dataset["first cast"]= new_cast[0]
# making separate second cast column from new data frame 
dataset["second cast"]= new_cast[1] 
# making separate third cast column from new data frame 
dataset["third cast"]= new_cast[2] 
# making separate fourth cast column from new data frame 
dataset["fourth cast"]= new_cast[3] 
# making separate fifth cast column from new data frame 
dataset["fifth cast"]= new_cast[4] 
# making separate sixth cast column from new data frame 
dataset["sixth cast"]= new_cast[5] 
# making separate seventh cast column from new data frame 
dataset["seventh cast"]= new_cast[6] 
# making separate eighth cast column from new data frame 
dataset["eighth cast"]= new_cast[7] 
# making separate ninth cast column from new data frame 
dataset["ninth cast"]= new_cast[8] 
# making separate tenth cast column from new data frame 
dataset["tenth cast"]= new_cast[9] 
# making separate eleventh cast column from new data frame 
dataset["eleventh cast"]= new_cast[10] 





# new data frame with split value columns 
new_countries = dataset["country"].str.split(", ", n = 6, expand = True) 
# making separate first director column from new data frame 
dataset["first country"]= new_countries[0]
# making separate second director column from new data frame 
dataset["second country"]= new_countries[1] 
# making separate third director column from new data frame 
dataset["third country"]= new_countries[2] 
# making separate fourth director column from new data frame 
dataset["fourth country"]= new_countries[3] 
# making separate fifth director column from new data frame 
dataset["fifth country"]= new_countries[4] 
# making separate fifth director column from new data frame 
dataset["sixth country"]= new_countries[5] 






# new data frame with split value columns 
new_listed_in= dataset["listed_in"].str.split(", ", n = 6, expand = True) 
# making separate first listed_in column from new data frame 
dataset["first listed_in"]= new_listed_in[0]
# making separate second listed_in column from new data frame 
dataset["second listed_in"]= new_listed_in[1] 
# making separate third listed_in column from new data frame 
dataset["third listed_in"]= new_listed_in[2] 



# Dropping old Name columns 
dataset.drop(columns =['director'], inplace = True) 
# Dropping old Name columns 
dataset.drop(columns =["cast"], inplace = True) 
# Dropping old Name columns 
dataset.drop(columns =["listed_in"], inplace = True) 
# Dropping old Name columns 
dataset.drop(columns =["country"], inplace = True) 


dataset['result'] = pd.concat([dataset["first listed_in", dataset["second listed_in"], dataset["third listed_in"]], axis=1, sort=False)

'''
Not efficient. Need to create a loop for the function. 
got the idea from https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
'''

# =============================================================================
# Examining countries that produced films
# =============================================================================
dataset['country'].unique()

dataset.groupby(['country']).size().sort_values(ascending=False)

# Examines the top 25 countries that have complaints
plt.figure(figsize = (12, 8))
sns.countplot(x = 'country', data = dataset, palette = 'viridis', order = dataset['country'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Countries', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('States', fontsize = 14)
plt.show()

# =============================================================================
# Examining Rating
# =============================================================================
dataset['rating'].unique()

dataset.groupby(['rating']).size().sort_values(ascending=False)
#TV-MA       2027
#TV-14       1698
#TV-PG        701
#R            508
#PG-13        286
#NR           218
#PG           184
#TV-Y7        169
#TV-G         149
#TV-Y         143
#TV-Y7-FV      95
#G             37
#UR             7
#NC-17          2

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'rating', data = dataset, palette = 'viridis', order = dataset['rating'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Rating', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Ratings', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['rating'].value_counts().head(25).index.tolist()
sizes = dataset['rating'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of rating', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()



# =============================================================================
# Examining type
# =============================================================================
dataset['type'].unique()

dataset.groupby(['type']).size().sort_values(ascending=False)
#Movie      4265
#TV Show    1969

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'type', data = dataset, palette = 'viridis', order = dataset['type'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Type', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Type', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['type'].value_counts().head(25).index.tolist()
sizes = dataset['type'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of Type', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()


# =============================================================================
# Examining listed_in
# =============================================================================
dataset['listed_in'].unique()

dataset.groupby(['listed_in']).size().sort_values(ascending=False)
#Movie      4265
#TV Show    1969

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'listed_in', data = dataset, palette = 'viridis', order = dataset['listed_in'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Top 25 listed_in', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('listed_in', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['listed_in'].value_counts().head(25).index.tolist()
sizes = dataset['listed_in'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of listed_in', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()


# =============================================================================
# Examining release_year
# =============================================================================
dataset['release_year'].unique()

dataset.groupby(['release_year']).size().sort_values(ascending=False)
#Movie      4265
#TV Show    1969

# Examines the ratings 
plt.figure(figsize = (12, 8))
sns.countplot(x = 'release_year', data = dataset, palette = 'viridis', order = dataset['release_year'].value_counts().head(25).index)
plt.xticks(rotation = 90)
plt.title('Breakdown of Top 25 listed_in', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('release_year', fontsize = 14)
plt.show()

# creates a pie chart 
fig = plt.figure(figsize = (20,10))
labels = dataset['release_year'].value_counts().head(25).index.tolist()
sizes = dataset['release_year'].value_counts().head(25).tolist()
plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
        shadow = False, startangle = 30)
plt.title('Breakdown of release_year', fontdict = None, position = [0.48,1], size = 'xx-large')
plt.show()


# =============================================================================
# Attempting to draw a map
# =============================================================================
# libraries
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Always start witht the basemap function to initialize a map
m=Basemap()

# Then add element: draw coast line, map boundary, and fill continents:
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents()


# =============================================================================
# Other interesting points to identify
# =============================================================================

# do a stacked barchart by country and Type

# create a map of the world with saturation

# split duration by Seasons and minutes for Movies and TV Shows

# examine release dates
# examine release dates and titles (Christmas Movies released at Christmas etc.)
# examine the description

# run a Machine Learning Algorithm to predict Rating based on listed_in and title and description
    # so kids movies get catergorised as kids movies and adult movies are categorised as kids movies.
#Split the director and cast and analyse by them. (eg. Jennifer Aniston mostly stars in comedies)
    


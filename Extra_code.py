# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:11:31 2020

@author: Karina
"""

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
# Preparing Genres
# =============================================================================

# new data frame with split value columns 
new_listed_in= df['listed_in'].str.split(', ', n = 6, expand = True) 
# making separate first listed_in column from new data frame 
df['first listed_in']= new_listed_in[0]
# making separate second listed_in column from new data frame 
df['second listed_in']= new_listed_in[1] 
# making separate third listed_in column from new data frame 
df['third listed_in']= new_listed_in[2] 


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
# Countries
# =============================================================================
## The issue with countries is that there are some films and Movies that have a
## several locations and it is impossible just to do a barchart with the data.

sns.set(style="white", color_codes=True)

# We want a very fast way to concat strings.
s = StringIO()
df['country'].apply(lambda x: s.write(x))

k = s.getvalue()
s.close()
k = k.split()

# Next only want valid strings
words = co.Counter(nltk.corpus.words.words())
stopWords = co.Counter( nltk.corpus.stopwords.words() )
k = [i for i in k if i in words and i not in stopWords]
s = " ".join(k)
c = co.Counter(k)

# At this point we have k,s and c
# k Array of words, with stop words removed
# s Concatinated string of all countries
# c Collection of words
# Take a look at the 14 most common words
c.most_common(14)

s[0:100]

print(k[0:10],"\n\nLength of k %s" % len(k))


# Read the whole text.
movie['country'] = movie['country'].astype(str)

text = df['country']

# Generate a word cloud image
wordcloud = WordCloud().generate(text)


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",max_words=len(k),max_font_size=40, relative_scaling=.8).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
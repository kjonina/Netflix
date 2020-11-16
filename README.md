# Netflix
This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset was collected from [Kaggle](https://www.kaggle.com/shivamb/netflix-shows).


# Learning Outcomes
Following my analysis of my [Netflix Viewing Habits](https://github.com/kjonina/personal_Netflix/blob/main/README.md), I decided to continue my quest to analyse data and update README.md on datasets that I have not yet finished.
I have a lot of holes in my knowledge so I decided to outline some learning outcomes for myself:

The purpose of this analysis for myself is to: 
- [ ] deal with parts of code I struggle with: loc, iloc splitting text data, creating linegraph and horizontal barcharts, barcharts by a category. 
- [ ] filter and split data (['Director',  'Cast', 'Genre']
- [ ] run analysis on ['Title', 'Description']
- [ ] create a predictive model for Ratings 
- [ ] continue fancy-schmancy code to update README.md such as links to external url and links, etc. 
- [ ] use GITHUB regularly to update code as I work on this project
- [ ] Hopefully update it to Kaggle?

### Variables
['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description', 'year_added', 'new_or_old']

New Variables:
['new_or_old'] compares the number of years between the original release data and year added to Netflix.

The dataset was also split by **'Movie'** and **'TV_Shows'**

## Exploring EDA

**What is more commonly found: Movies or TV Shows?**

![Types](https://github.com/kjonina/Netflix/blob/master/Graphs/Types.png)

TV Shows make up over two-thirds of contents on Netflix (68.4% - TV Shows, 31.6% - Movies). 

**What are some of the most common TV-Ratings?**

![Ratings](https://github.com/kjonina/Netflix/blob/master/Graphs/Ratings.png)

**When were the most TV shows/ Movies released?**

![release](https://github.com/kjonina/Netflix/blob/master/Graphs/release.png)

**How new are additions on Netflix?**

![new_or_old](https://github.com/kjonina/Netflix/blob/master/Graphs/new_or_old.png)

Most content is uploaded onto Netflix within 0 - 2 years of its initial release. Unfortunatly, month of the release_year was not supplied so it is not very accurate representation.
For Example: a movie could have been release in November and uploaded on Netflix in February, which means the time difference is 4 months, not 1 year.

Furthermore: it is unclear whether the content with 0 years between release_year and year_added is Netflix produced.

One way to found it is to create code to examine that compares contents to Netflix produced. 
- but that is for later. 

**How long are most movies?**
![movie_d](https://github.com/kjonina/Netflix/blob/master/Graphs/movie_d.png)

An average film on Netflix rather long (M = 102 minutes, SD = 26 minutes).
The shortest film on Netflix is 12 minutes and the longest is 228 minutes.

**How many Seasons do most TV Shows have?**

![seasons](https://github.com/kjonina/Netflix/blob/master/Graphs/seasons.png)



## Next Questions to anwer: 

**What are the most common words in TV Show Movie description?**


**What are the most genre?**







 =============================================================================
 Other interesting points to identify
 =============================================================================
Upon further inspection of the data,it was noted that the following columns need to be split.
# countries,
# director
# cast,
# listed_in

 THere are also a lot of NULL values for director and cast members, which I will have to decided what to do with 

 do a stacked barchart by country and Type

 create a map of the world with saturation

 split duration by Seasons and minutes for Movies and TV Shows

 examine release dates
 examine release dates and titles (Christmas Movies released at Christmas etc.)
 examine the description

 run a Machine Learning Algorithm to predict Rating based on listed_in and title and description
     so kids movies get catergorised as kids movies and adult movies are categorised as kids movies.
Split the director and cast and analyse by them. (eg. Jennifer Aniston mostly stars in comedies)
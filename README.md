# Netflix
This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset was collected from [Kaggle](https://www.kaggle.com/shivamb/netflix-shows).


# Learning Outcomes
Following my analysis of my [Netflix Viewing Habits](https://github.com/kjonina/personal_Netflix/blob/main/README.md), I decided to continue my quest to analyse data and update README.md on datasets that I have not yet finished.
I have a lot of holes in my knowledge

I am aiming to:
- [ ] deal with parts of code I struggle with: loc, iloc splitting text data, creating linegraph and horizontal barcharts. 
- [ ] filter and split data (['Director',  'Cast', 'Genre']
- [ ] run analysis on ['Title', 'Description']
- [ ] create a predictive model for Ratings 

- [ ] Hopefully update it to Kaggle?

### Variables
['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description', 'year_added', 'new_or_old']

New Variables:
['new_or_old'] compares the number of years between the original release data and year added to Netflix.

The dataset was also split by **'Movie'** and **'TV_Shows'**


## Exploring EDA

**What are some of the most common TV-Ratings?**

![Ratings](https://github.com/kjonina/Netflix/blob/master/Graphs/Ratings.png)

**What is more commonly found: Movies or TV Shows?**

![Types](https://github.com/kjonina/Netflix/blob/master/Graphs/Types.png)

![Types](https://github.com/kjonina/Netflix/blob/master/Graphs/Pie_Types.png)

**When were the most TV shows/ Movies released?**

![release](https://github.com/kjonina/Netflix/blob/master/Graphs/release.png)

**How new are additions on Netflix?**

![new_or_old](https://github.com/kjonina/Netflix/blob/master/Graphs/new_or_old.png)

**How long are most movies?**
![movie_d](https://github.com/kjonina/Netflix/blob/master/Graphs/movie_d.png)

An average film on Netflix rather long (M = 102 minutes, SD = 26 minutes).
The shortest film on Netflix is 12 minutes and the longest is 228 minutes.

**How many Seasons do most TV Shows have?**
![seasons](https://github.com/kjonina/Netflix/blob/master/Graphs/seasons.png)
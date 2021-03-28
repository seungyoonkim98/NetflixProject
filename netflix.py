import numpy as np
import pandas as pd
from collections import Counter
from rake_nltk import Rake
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##########################################################################################################
### Clean data
##########################################################################################################

### Read csv file
netflix_df = pd.read_csv('netflix_titles.csv')

### Shape of dataframe
print(netflix_df.shape)

### Missing values
print(pd.isnull(netflix_df).sum())

### Remove rows where date_added and rating contain missing values
### We can do this because number of rows containing missing values in these columns is 
### small compared to total number of rows, minimizing effect on data
netflix_df.dropna(subset=['date_added', 'rating'])

### Remove unneccessary columns
netflix_df = netflix_df.drop(columns=['show_id', 'date_added'])

### Print df columns to determine interesting visualizations
print(netflix_df.columns)

##########################################################################################################
### Process and visualize data
##########################################################################################################

### Most frequently occuring directors (director)
netflix_df['director'] = netflix_df['director'].fillna('Unknown')
netflix_df['director'] = netflix_df['director'].astype(str)
netflix_df['director'] = netflix_df['director'].map(lambda x: x.split(','))

directors_temp = list(netflix_df['director'])
directors = [director.replace(' ', '')  for directors_list in directors_temp for director in directors_list if director is not 'Unknown']

director_counter = Counter(directors).most_common(10)
directors = [director for director, directed_movies_count in director_counter]
directed_movies_count = [directed_movies_count for director, directed_movies_count in director_counter]

plt.bar(directors, directed_movies_count)
plt.show()

### Most frequently occuring cast members (cast)
netflix_df['cast'] = netflix_df['cast'].fillna('Unknown')
netflix_df['cast'] = netflix_df['cast'].astype(str)
netflix_df['cast'] = netflix_df['cast'].map(lambda x: x.split(','))

cast_temp = list(netflix_df['cast'])
cast = [actor.replace(' ', '') for cast_list in cast_temp for actor in cast_list if actor is not 'Unknown']

cast_counter = Counter(cast).most_common(10)
cast = [cast for cast, appearances in cast_counter]
appearances = [appearances for cast, appearances in cast_counter]

plt.bar(cast, appearances)
plt.show()

### Distribution of shows across countries (country)
netflix_df['country'] = netflix_df['country'].fillna('Unknown')
netflix_df['country'] = netflix_df['country'].astype(str)
netflix_df['country'] = netflix_df['country'].map(lambda x: x.split(','))

country_temp = list(netflix_df['country'])
countries = [country.replace(' ', '') for country_list in country_temp for country in country_list if country is not 'Unknown']

countries_counter = Counter(countries).most_common(10)
countries = [country for country, appearances in countries_counter]
appearances = [appearances for country, appearances in countries_counter]

plt.pie(appearances, labels=countries)
plt.show()

### Distribution of shows across time (release_year)
release_years = list(netflix_df['release_year'])
span = max(release_years) - min(release_years)

plt.hist(release_years, bins=span)
plt.show()

### Distribution of ratings (rating)
netflix_df['rating'] = netflix_df['rating'].astype(str)
netflix_df['rating'] = netflix_df['rating'].map(lambda x: x.split(','))

ratings_temp = list(netflix_df['rating'])
ratings = [rating.replace(' ', '') for ratings_list in ratings_temp for rating in ratings_list if rating is not 'Unknown']

ratings_counter = Counter(ratings).most_common()
ratings = [rating for rating, count in ratings_counter]
counts = [count for rating, count in ratings_counter]

plt.bar(ratings, counts)
plt.show()

### Distribution of genres (listed_in)
netflix_df['listed_in'] = netflix_df['listed_in'].astype(str)
netflix_df['listed_in'] = netflix_df['listed_in'].map(lambda x: x.split(','))
genres_temp = netflix_df['listed_in']

genres = [genre.replace(' ', '') for genres_list in genres_temp for genre in genres_list]

genres_counter = Counter(genres).most_common()
genres = [genre for genre, count in genres_counter]
counts = [count for genre, count in genres_counter]

plt.bar(genres, counts)
plt.show()

##########################################################################################################
### RAKE text preprocessor based recommender
##########################################################################################################

### Extract keywords from Netflix title descriptions
r = Rake()

def implement_rake(x, r):
    r.extract_keywords_from_text(x)
    key_words_dict_scores = r.get_word_degrees()
    return list(key_words_dict_scores.keys())

netflix_df['description_keywords'] = netflix_df['description'].apply(lambda x: implement_rake(x, r))
netflix_df['director'] = netflix_df['director'].map(lambda directors: [director.lower().replace(' ', '') for director in directors if director is not 'Unknown'])
netflix_df['cast'] = netflix_df['cast'].map(lambda cast: [actor.lower().replace(' ', '') for actor in cast if actor is not 'Unknown'])
netflix_df['listed_in'] = netflix_df['listed_in'].map(lambda listed_in: [genre.lower().replace(' ', '') for genre in listed_in])

### Aggregate list of words into one big column
netflix_df['combined_keywords'] = netflix_df['description_keywords'] + netflix_df['director'] + netflix_df['cast'] + netflix_df['listed_in']
netflix_df['combined_keywords'] = netflix_df['description_keywords'].map(lambda x: ' '.join(x))
relevant_columns = ['title', 'combined_keywords']
netflix_df = netflix_df[relevant_columns]

### Create vectorizer and cosine similarity matrix
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(netflix_df['combined_keywords'])
cosine_similarity = cosine_similarity(sparse_matrix, sparse_matrix)

indices = pd.Series(netflix_df['title'])

### Define and call recommend function
def recommend(title, cosine_sim = cosine_similarity):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:11].index)
                    
    for i in top_10_indices:
        recommended_movies.append(list(netflix_df['title'])[i])
    
    return recommended_movies

print(netflix_df['title'].head(1))
print(recommend('3%'))

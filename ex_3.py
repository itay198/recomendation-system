
import pandas as pd
import numpy as np
import heapq

from sklearn.metrics.pairwise import pairwise_distances



books = pd.read_csv('./books.csv', sep=',', header=0 , encoding='latin-1')
books_tags = pd.read_csv('./books_tags.csv', encoding='utf-8', low_memory=False)
ratings = pd.read_csv('./ratings.csv', sep=',', header=0 , encoding='latin-1')
tags = pd.read_csv('./tags.csv', encoding='utf-8', low_memory=False)
users = pd.read_csv('./users.csv', encoding='utf-8', low_memory=False)

##################################################################################################################
# part 1
def data_by_age(books, ratings, users, age):
    def age_ids(age, users):
        bottom = int(age / 10) * 10
        up = (int(age / 10) + 1) * 10
        temp = users.where((users['age'] > bottom) & (users['age'] <= up))  # ['user_id']
        return temp.dropna(how='all').sort_values(by=['user_id'], ascending=False)

    user_age = age_ids(age, users)
    user_age['user_id'] = user_age['user_id'].apply(np.int64)
    rate_sort = ratings.sort_index(inplace=True)
    rate_age = ratings[ratings['user_id'].isin(
        user_age['user_id'])]  # rate_sort.where[rate_sort['user_id'] == user_place['user_id']]
    books_age = books[books['book_id'].isin(rate_age['book_id'])]
    return books_age, rate_age

def data_by_place(books, ratings, users, place):
    def place_ids(place, users):
        temp = users.where(users['location'] == place)  # ['user_id']
        return temp.dropna(how='all').sort_values(by=['user_id'], ascending=False)

    user_place = place_ids(place, users)
    user_place['user_id'] = user_place['user_id'].apply(np.int64)
    rate_place = ratings[ratings['user_id'].isin(user_place['user_id'])].sort_values(by=['book_id'],
                                                                                     ascending=False)  # rate_sort.where[rate_sort['user_id'] == user_place['user_id']]
    books_place = books[books['book_id'].isin(rate_place['book_id'])].sort_values(by=['book_id'], ascending=False)
    return books_place, rate_place

def get_simply_recommendation(k, books=books.copy(deep=True), ratings=ratings.copy(deep=True)):
    def vote_count(book, ratings=ratings):
        count = ratings[ratings['book_id'] == book['book_id']]
        return len(count)

    books['vote_count'] = books.apply(vote_count, axis=1)

    def vote_average(book, ratings=ratings):
        avg = ratings.loc[ratings['book_id'] == book['book_id'], 'rating'].mean()
        return avg

    books['vote_average'] = books.apply(vote_average, axis=1)
    C = books['vote_average'].mean()
    m = books['vote_count'].quantile(0.9)
    q_books = books.copy().loc[books['vote_count'] >= m]

    def weighted_avg_rate(book, m=m, C=C):
        v = book['vote_count']
        r = book['vote_average']
        return (v / (v + m) * r) + (m / (m + v) * C)


    q_books['score'] = q_books.apply(weighted_avg_rate, axis=1)
    q_books = q_books.sort_values('score', ascending=False)

    return q_books.head(k)

def get_simply_place_recommendation(place, k, books=books.copy(deep=True), ratings=ratings.copy(deep=True)):
    place_books, place_rate = data_by_place(books.copy(deep=True), ratings.copy(deep=True), users.copy(deep=True),
                                            place)
    return get_simply_recommendation(k, place_books, place_rate)

def get_simply_age_recommendation(age, k, books=books.copy(deep=True), ratings=ratings.copy(deep=True)):
    age_books, age_rate = data_by_age(books.copy(deep=True), ratings.copy(deep=True), users.copy(deep=True), age)
    return get_simply_recommendation(k, age_books, age_rate)

print("1: \n", get_simply_recommendation(10)[
    ['book_id', 'title', 'score']])  # q1 returns 10 top books by weighted avg rating

print("2: \n",
      get_simply_place_recommendation('Ohio', 10)[['book_id', 'title', 'score']])

print("3: \n", get_simply_age_recommendation(28, 10)[['book_id', 'title', 'score']])

###################################################################################################################
# part2
# returns the array with the top k values, and all the rest as 0's
def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr

# For debug purposes - appends rating r for items on behalf of a debug user

def build_CF_prediction_matrix(sim):
    n_users = ratings.user_id.unique().shape[0]
    n_items = ratings.book_id.unique().shape[0]

    users_dict = {}
    sort_users = ratings.user_id.unique()
    sort_users.sort()
    for i_user, id_user in enumerate(sort_users):
        users_dict[id_user] = i_user
    books_dict = {}
    sort_books = ratings.book_id.unique()
    sort_books.sort()
    for i_books, id_books in enumerate(sort_books):
        books_dict[id_books] = i_books

    # create ranking table - that table is sparse
    data_matrix1 = np.empty((n_users, n_items))
    data_matrix1[:] = np.nan
    for line in ratings.itertuples():
        user = line[1]
        movie = line[2]
        rating = line[3]
        data_matrix1[users_dict[user], books_dict[movie]] = rating

    # calc mean
    mean_user_rating = np.nanmean(data_matrix1, axis=1).reshape(-1, 1)

    ratings_diff = (data_matrix1 - mean_user_rating)
    # replace nan -> 0
    ratings_diff[np.isnan(ratings_diff)] = 0
    # if sim == 'jaccard':
    #    ratings_diff = np.array(ratings_diff, dtype=bool)
    # calculate user x user similarity matrix
    user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)

    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    k = 10
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])

    # since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

    return data_matrix1, users_dict, books_dict, pred

# Function that takes in book title as input and outputs most similar movies
def get_recommendations(predicted_ratings_row, data_matrix_row, items, k=5):
    predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
    idx = np.argsort(-predicted_ratings_row)
    sim_scores = idx[0:k]

    # Return top k movies
    return items[['title', 'book_id']].iloc[sim_scores]
# data_matrix_e, users_dict_e, books_dict_e, pred_e = build_CF_prediction_matrix('euclidean')
# data_matrix, users_dict, books_dict, pred = build_CF_prediction_matrix('cosine')
# data_matrix_j, users_dict_j, books_dict_j, pred_j = build_CF_prediction_matrix('jaccard')
def get_CF_recommendations(user_id, k, sim):
    ratings_path = 'ratings.csv'
    items_path = 'books.csv'

    # Reading ratings file:
    r_cols = ['user_id', 'book_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(ratings_path, skiprows=[0], sep=',', names=r_cols, encoding='latin-1')

    # Reading items file:
    i_cols = ['book_id', 'goodreads_book_id', 'best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'authors',
              'original_publication_year', 'original_title', 'title', 'language_code', 'image_url',
              'small_image_url']

    items = pd.read_csv(items_path, skiprows=[0], sep=',', names=i_cols, encoding='latin-1')


    data_matrix, users_dict, books_dict, pred = build_CF_prediction_matrix(sim)
    user = users_dict[user_id]
    predicted_ratings_row = pred[user]
    data_matrix_row = data_matrix[user]
    return get_recommendations(predicted_ratings_row, data_matrix_row, items, k), pred, users_dict, books_dict


cosine_1 = get_CF_recommendations(1, 10, 'cosine')
print(cosine_1)
# #build_CF_prediction_matrix('euclidean')
# #build_CF_prediction_matrix('jaccard')

####################################################################################################################
#part3:

# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Movies Metadata
metadata = pd.read_csv('./books.csv', sep=',', header=0, encoding='latin-1')

# Print plot overviews of the first 5 movies.
# print(metadata[['title', 'authors', 'language_code']].head())

### Add 1
# Load books_tags and tags
#books_tags = pd.read_csv('./books_tags.csv', encoding='utf-8', low_memory=False)
#tags = pd.read_csv('./tags.csv', encoding='utf-8', low_memory=False)

# Convert IDs to int. Required for merging
books_tags['tag_id'] = books_tags['tag_id'].astype('int')
tags['tag_id'] = tags['tag_id'].astype('int')
metadata['goodreads_book_id'] = metadata['goodreads_book_id'].astype('int')

# Merge books_tags and tags into your main metadata dataframe
# all_tags = books_tags.merge(tags, on='tag_id')
# goods = all_tags['goodreads_book_id'].tolist()
# tags_dict = {}
# for tag in goods:
#     temp = all_tags.loc[all_tags['goodreads_book_id'] == tag]['tag_name'].tolist()
#     str1 = ''
#     for word in temp:
#         str1 += word
#     tags_dict[tag] = str1
# metadata =metadata.merge(all_tags, on='goodreads_book_id')

# metadata = metadata.merge(credits, on='id')
# metadata = metadata.merge(keywords, on='id')

# Print the first two movies of your newly merged metadata
print(metadata.head(2))

# Print the new features of the first 3 films
print(metadata[['title', 'authors', 'language_code']].head(3))  # ,'tag_name']].head(3))

# Add 5
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['title', 'authors', 'language_code']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

# Add 6
def create_soup(x):
    return ' '.join(x['title']) + x['authors'] + x[
        'language_code']  # + tags_dict[x['goodreads_book_id']]# + ' ' + ' '.join(x['tag_name'])

# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)

print(metadata[['soup']].head(2))

# Add 7
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
print(count_matrix.shape)

# Add 8
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['original_title'])
print(indices[:10])

# # Function that takes in movie title as input and outputs most similar movies
def get_recommendations1(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    temp = indices[title]
    if temp.size > 1:
        idx = temp.iloc[0]
    else:
        idx = temp

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (the first is the movie we asked)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['original_title'].iloc[movie_indices]

# "Twilight
# temp = metadata.loc[lambda df: df['title'] == 'Twilight']['original_title']
# str = temp.iloc[0]['']
# print('********************** Twilight ****************************')
# print(get_recommendations1('Twilight'))
'''
print('********************** The Dark Knight Rises ****************************')
print(get_recommendations('The Dark Knight Rises'))
print('************************ Mean Girls **************************')
print(get_recommendations('Mean Girls'))
print('*********************** Star Wars ***************************')
print(get_recommendations('Star Wars'))
print('*********************** Robots ***************************')
print(get_recommendations('Robots'))
print('*********************** The Princess Diaries ***************************')
print(get_recommendations('The Princess Diaries'))
'''

####################################################################################################################
def filter_test():
    k = 10
    test = pd.read_csv('./test.csv', sep=',', header=0, encoding='latin-1')
    test_filtered = test.copy(deep=True).loc[test['rating'] > 3]
    test_filtered['freq'] = test_filtered.copy(deep=True).groupby('user_id')['user_id'].transform('count')
    test_filtered = test_filtered.loc[test_filtered['freq'] >= k]
    return test_filtered

test_filtered = filter_test()

def precision_k(k, test_filtered=test_filtered):
    sort_users = test_filtered['user_id'].unique()
    sort_users.sort()
    sims = ['cosine', 'euclidean', 'jaccard']
    perk_list = []
    for sim in sims:
        total_hits = 0
        for i, id in enumerate(sort_users):
            y_hat_temp, data_matrix, users_dict, books_dict = get_CF_recommendations(id, k, sim)
            y_hat = y_hat_temp['book_id']
            temp_y =test_filtered.loc[test_filtered['user_id'] == id]
            y = temp_y[['book_id', 'rating']]
            y_list = y['book_id'].tolist()
            y_hat_list = y_hat.tolist()
            user_hits = 0
            for rate in y_hat_list:
                if rate in y_list:
                    user_hits += 1
            user_hits /= k
            total_hits += user_hits
        perk_list.append(total_hits / len(sort_users))
    return perk_list

def ARHR(k, test_filtered=test_filtered):
    sort_users = test_filtered['user_id'].unique()
    sort_users.sort()
    sims = ['cosine', 'euclidean', 'jaccard']
    arhr_list = []
    for sim in sims:
        total_hits = 0
        for i, id in enumerate(sort_users):
            y_hat_temp, data_matrix, users_dict, books_dict = get_CF_recommendations(id, k, sim)
            y_hat = y_hat_temp['book_id']
            temp_y = test_filtered.loc[test_filtered['user_id'] == id]
            y = temp_y[['book_id', 'rating']]
            y_list = y['book_id'].tolist()
            y_hat_list = y_hat.tolist()
            # count hits for user id and adds according to the location of hit in user y_list (j)
            user_hits = 0
            for j, rate in enumerate(y_hat_list):
                if rate in y_list:
                    user_hits += 1 / (j+1)
            total_hits += user_hits

        arhr_list.append(total_hits / len(sort_users))
    return arhr_list


test2 = pd.read_csv('./test.csv', sep=',', header=0, encoding='latin-1')


def RMSE(test2=test2):
    sims = ['cosine', 'euclidean', 'jaccard']
    rmse_list = []
    for sim in sims:
        data_matrix, users_dict, books_dict, pred = build_CF_prediction_matrix(sim)
        loss = 0
        N = 0
        for index, row in test2.iterrows():
            y_hat = pred[users_dict[row['user_id']], books_dict[row['book_id']]]
            loss += (row['rating'] - y_hat) ** 2
            N += 1
        rmse_list.append((loss / N) **0.5)
    return rmse_list


print("ARHR, cosine: ", ARHR(10))
print("precision_k, cosine: ", precision_k(10))
print("RMSE, cosine: ", RMSE())

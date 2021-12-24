# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 20:28:45 2021

@author: abhil
"""

# importing the libararies
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#import seaborn as sns

from scipy.sparse import csr_matrix # to create user-item matrix

from sklearn.neighbors import NearestNeighbors #finding nearest ratings using nearestneighbour algo
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import CountVectorizer


# importing the data
movies_data = pd.read_csv('E:/Projects/Python/recommendation/ml-25m/ml-25m/movies.csv')
movies_ratings_data = pd.read_csv('E:/Projects/Python/recommendation/ml-25m/ml-25m/ratings.csv')
# checking data
#print(movies_data.head(3))
rating_length = len(movies_ratings_data)
movies_length = len(movies_data)
unique_movies = len(movies_ratings_data['movieId'].unique())
unique_users = len(movies_ratings_data['userId'].unique())
#print(unique_movies,unique_users)

user_freq = movies_ratings_data[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
#print(user_freq.head(2))

# creating a function to generate user-item matrix
def create_matrix(df):
    N= len(df['userId'].unique())
    M = len(df['movieId'].unique())
    
    # mapping Ids to indices(means indexing)
    user_mapping = dict(zip(np.unique(df['userId']),list(range(N))))
    movie_mapping =dict(zip(np.unique(df['movieId']),list(range(M))))
    
    #inverse mappping of user and movie
    user_inver_mapping = dict(zip(list(range(N)),np.unique(df['userId'])))
    movie_inver_mapping = dict(zip(list(range(M)),np.unique(df['movieId'])))
    
    user_index=[user_mapping[i] for i in df['userId']]
    movie_index = [movie_mapping[i] for i in df['movieId']]
    
    X=  csr_matrix((df['rating'], (movie_index,user_index)), shape=(M,N))
    
    return X, user_mapping, movie_mapping, user_inver_mapping, movie_inver_mapping,
# function ends here
#here we are calling create_matrix function
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(movies_ratings_data)

# creating function to find similar movies using nearest neighbour algo with cosine metric

def find_similar(movie_id, X, k, metric='cosine',show_distance=False):
    neibour_list=[] #or we can say simlar movie list
    movie_ind = movie_mapper[movie_id]
    movie_vec= X[movie_ind]
    k +=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric)
    kNN.fit(X)
    movie_vec= movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec,  return_distance=show_distance)
    for i in range(0,k):
        n= neighbour.item(i)
        neibour_list.append(movie_inv_mapper[n])
    neibour_list.pop(0)
    return neibour_list
# function ends here
def main_call(titles):
    movie_titles = dict(zip(movies_data['movieId'], movies_data['title']))
    movie_title = dict(zip(movies_data['title'], movies_data['movieId']))
#movie_id =3
# titles ='Spider-Man (2002)'

    Id = movie_title[titles]
    similar_ids =find_similar(Id, X, k=10) 
    recomend_movies=[]
#movie_title= movie_titles[titles]
#movie_title= movie_titles[movie_id]
# print('movie id for {} is: '.format(titles),Id)
# print(f"Since you watched {titles}")
    for i in similar_ids:
        recomend_movies.append(movie_titles[i])       
    return recomend_movies

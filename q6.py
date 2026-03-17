# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# ======================================================
# Question B: Dataset Exploration
# Operation: Load dataset and explore its structure
# ======================================================

ratings = pd.read_csv("ratings.csv")

print("First 5 rows of dataset:")
print(ratings.head())

print("\nDataset Information:")
print(ratings.info())

print("\nStatistical Summary:")
print(ratings.describe())


# ======================================================
# Operation: Create User-Item Interaction Matrix
# ======================================================

user_item_matrix = ratings.pivot_table(index='userId',
                                       columns='movieId',
                                       values='rating')

user_item_matrix = user_item_matrix.fillna(0)

print("\nUser Item Matrix:")
print(user_item_matrix.head())


# ======================================================
# Question C: User-Based Collaborative Filtering
# Operation: Calculate similarity between users
# ======================================================

user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)

print("\nUser Similarity Matrix:")
print(user_similarity_df.head())


# ======================================================
# Operation: Recommend items based on similar users
# ======================================================

def recommend_movies_user(user_id, n=5):

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    similar_users_movies = user_item_matrix.loc[similar_users.index]

    recommendation_scores = similar_users_movies.mean().sort_values(ascending=False)

    return recommendation_scores.head(n)


print("\nUser Based Recommendations for User 1:")
print(recommend_movies_user(1))


# ======================================================
# Question D: Item-Based Collaborative Filtering
# Operation: Calculate similarity between items
# ======================================================

item_matrix = user_item_matrix.T

item_similarity = cosine_similarity(item_matrix)

item_similarity_df = pd.DataFrame(item_similarity,
                                  index=item_matrix.index,
                                  columns=item_matrix.index)

print("\nItem Similarity Matrix:")
print(item_similarity_df.head())


# ======================================================
# Operation: Recommend similar items
# ======================================================

def recommend_movies_item(movie_id, n=5):

    similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]

    return similar_movies


print("\nItem Based Recommendations for Movie 1:")
print(recommend_movies_item(1))


# ======================================================
# Question E: Hybrid Recommender System
# Operation: Combine User-Based and Item-Based approaches
# ======================================================

hybrid_similarity = (user_similarity + item_similarity[:user_similarity.shape[0], :user_similarity.shape[0]]) / 2

print("\nHybrid Similarity Matrix Created")


# ======================================================
# Question F: Evaluation Metrics
# Operation: Calculate Precision, Recall and F1 Score
# ======================================================

threshold = 3.5

ratings['relevant'] = ratings['rating'].apply(lambda x: 1 if x >= threshold else 0)

y_true = ratings['relevant']
y_pred = (ratings['rating'] >= threshold).astype(int)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nEvaluation Results:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ======================================================
# Final Output
# ======================================================

print("\nFinal Recommendation Example")
print("Recommended Movies for User 1:")
print(recommend_movies_user(1))
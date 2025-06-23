# RECOMMENDATION SYSTEM USING KNNs

# Import libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load CSVs
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create a matrix containing both movies and user ratings
user_movie_matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

# Compute user similarity
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index = user_movie_matrix.index, columns = user_movie_matrix.index)

# Recommendation function
def knn_recommend(user_id, k = 5, n_recs = 10):
    if user_id not in user_similarity_df.index:
        return f"User {user_id} not found."
    
    similar_users = user_similarity_df[user_id].drop(user_id).nlargest(k).index
    neighbor_ratings = user_movie_matrix.loc[similar_users]
    
    user_sim_scores = user_similarity_df.loc[user_id, similar_users]
    weighted_ratings = neighbor_ratings.T.dot(user_sim_scores) / user_sim_scores.sum()
    
    seen_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    weighted_ratings = weighted_ratings.drop(seen_movies, errors = "ignore")
    
    top_movies = weighted_ratings.sort_values(ascending=False).head(n_recs)
    
    return pd.DataFrame({"title" : movies.set_index("movieId").loc[top_movies.index]["title"],
                        "predicted_rating" : top_movies.values})

# Displaying the recommendations
i = 1
while i > 0:
    try:
        selected_user = int(input(print("Enter User ID to display recommendations: ")))
        recs = knn_recommend(user_id = selected_user, k = 5, n_recs = 10)
        print(recs)
    
    except ValueError:
        print("Please enter a valid integer user ID.")
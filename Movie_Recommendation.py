### Toy example of LoRMA usage. Given data about users and their movie preferences, we recommend movies that the users have not seen and would like.
### Using MovieLens dataset and pandas library.

# Imports for loading and splitting data
from sklearn.model_selection import train_test_split
import pandas as pd
import LoRMA_GD

# Download the MovieLens dataset into a pandas dataframe and display it
movie_ratings_path = 'https://raw.githubusercontent.com/ameet-1997/Machine-Learning/master/MovieLens/ratings.csv'
movielens_df = pd.read_csv(movie_ratings_path)
print(movielens_df.head()) 

# Convert pandas DataFrame to matrix
matrix_df = movielens_df.pivot(index='userId', columns='movieId', values='rating')
print(matrix_df.head())

# Replace NaN values in matrix and display
missing_value = -1
movielens_matrix = matrix_df.fillna(missing_value).to_numpy()
print(movielens_matrix)

# check data processing step
def check_data_processing(M):
  assert(np.sum(M == -1) == 5830804)
  assert(M.shape[0] == 610)
  assert(M.shape[1] == 9724)
  assert(M.dtype == 'float64')
  assert(M[0][0] == 4)
  assert(M[0][1] == -1)
  print("Matrix passes basic data processing checks.")

check_data_processing(movielens_matrix)

# Create the observed matrix with 1s where we have user ratings for a given movie and 0s otherwise
movielens_observed = np.where(movielens_matrix == -1, 0, 1)

"""
Run LORMA on the MovieLens dataset.
"""
rand_seed = 10
np.random.seed(rand_seed)

# Normalize copied matrix
movielens_normalized = get_normalized_matrix(movielens_matrix, movielens_observed)

# Define parameters for LORMA
rank = 40
num_epochs = 2000
etas = np.ones(2000) * 10.0
params = k, num_epochs, etas
# Run LORMA using lorma_learn
A, B, losses = lorma_learn(movielens_normalized, movielens_observed, params)

# plot losses to make sure they are decreasing
_ = plt.plot(losses, '-o')

# Load the movies
# The data consists of rows which are movie IDs
movie_info_path = 'https://raw.githubusercontent.com/ameet-1997/Machine-Learning/master/MovieLens/movies.csv'
movielens_df = pd.read_csv(movie_info_path)
print(movielens_df.head())

"""
pick three users and see what movies they rated highly before, and what movies the model predicted

print out the top-p movies they rated most highly
"""

# Pick three users to recommend movies to
users = [1, 13, 111]

# Print top-p movies they have rated highly
p = 10
for user in users:
    print("\nUser {} liked the following:\n".format(user))

    # Sort the movies for this user in descending order based on the rating
    movie_order = np.argsort(-movielens_matrix[user])
    top_p = movie_order[:p]

    # Print the top p movies
    for movie in top_p:
        # Store the movie title, user rating, and movie genre
        movie_title = movielens_df.iloc[movie]['title']
        user_rating = movielens_matrix[user][movie]
        movie_genre = movielens_df.iloc[movie]['genres']
        print("\t{:<50} rated {:.1f}  genres {:<30}".format(
            movie_title,
            user_rating,
            movie_genre))

            
"""

Now let's make our predictions on the test data and see what movies we can recommend

"""

for user in users:
    print("\nRecommend the following movies to User {}\n".format(user))

    # Predict ratings
    predicted_ratings = A[user,:] @ B

    # If the movie review was observed in the matrix, set it to (-infinity) so that we don't predict it
    # We want to predict only from a set of movies which the user has not seen
    predicted_ratings[movielens_observed[user] == 1] = -np.inf

    # Choose the top_p movies
    predicted_movie_order = np.argsort(-predicted_ratings)
    top_p = predicted_movie_order[:p]

    # Print the recommended movies
    for movie in top_p:
      movie_title = movielens_df.iloc[movie]['title']
      movie_genre = movielens_df.iloc[movie]['genres']
      print("\t{:<60} genres {:<30}".format(movie_title[:60], # cap length of movie title to 60 char
                                            movie_genre))
            

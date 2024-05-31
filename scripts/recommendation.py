import pandas as pd

def get_movie_recommendations(model, user_id, ratings, movies, top_n=10):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_unrated_movies = movies[~movies['movieId'].isin(user_ratings['movieId'])]
    user_unrated_movies['est_rating'] = user_unrated_movies['movieId'].apply(lambda x: model.predict(user_id, x).est)
    recommendations = user_unrated_movies.sort_values('est_rating', ascending=False).head(top_n)
    recommendations = recommendations.merge(movies, on='movieId')[['movieId', 'title', 'est_rating']]
    return recommendations

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    from model_building import build_and_train_model
    dataset, movies, ratings, tags, links = load_and_preprocess_data()
    model = build_and_train_model(dataset)
    user_id = 1  # Example user
    recommendations = get_movie_recommendations(model, user_id, ratings, movies)
    print(recommendations)
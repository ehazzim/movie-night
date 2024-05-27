def get_movie_reco(model, user_id, data, top_n=10):
    user_ratings = data[data['userID'] == user_id]
    user_unrated_movies = data[~data['movieID'].isin(user_ratings['movieID'])]
    user_unrated_movies['est_rating'] = user_unrated_movies['movieID'].apply(lambda x: model.predict(user_id, x).est)
    recommendations = user_unrated_movies.sort_values('est_rating', ascending=False).head(top_n)
    return recommendations

if __name__ == "__main__":
    from data_preprocessing import load_preprocess_data
    from model_building import build_train_model
    dataset = load_preprocess_data('../data/movielens_dataset.csv')
    model = build_train_model(dataset)
    user_id = 1 # example
    recommendations = get_movie_reco(model, user_id, dataset.df)
    print(recommendations)
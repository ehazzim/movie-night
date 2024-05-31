import pandas as pd
from surprise import Dataset, Reader

def load_and_preprocess_data():
    # Load datasets
    movies = pd.read_csv('./data/raw/movies.csv')
    ratings = pd.read_csv('./data/raw/ratings.csv')
    tags = pd.read_csv('./data/raw/tags.csv')
    links = pd.read_csv('./data/raw/links.csv')

    # Preprocess ratings data for Surprise
    ratings = ratings[['userId', 'movieId', 'rating']]
    reader = Reader(rating_scale=(0.5, 5))
    dataset = Dataset.load_from_df(ratings, reader)
    
    return dataset, movies, ratings, tags, links

if __name__ == "__main__":
    dataset, movies, ratings, tags, links = load_and_preprocess_data()
    processed_data_path = './data/processed/processed_data.csv'
    ratings.to_csv(processed_data_path, index=False)
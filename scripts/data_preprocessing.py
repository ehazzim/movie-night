import pandas as pd
from surprise import Dataset, Reader

def load_preprocess_data(file):
    data = pd.read_csv(file)
    data = data[['userID', 'movieID', 'rating']]
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data, reader)
    return dataset

if __name__ == "__main__":
    dataset = load_preprocess_data('../data/movielens_dataset.csv')
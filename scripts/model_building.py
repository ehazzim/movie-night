from surprise import SVD, accuracy
from surprise.model_selection import train_test_split

def build_train_model(dataset):
    trainset, testset = train_test_split(dataset, test_size=0.2)
    algorithm = SVD()
    algorithm.fit(trainset)
    predictions = algorithm.test(testset)
    print('RMSE: ', accuracy.rmse(predictions))
    return algorithm

if __name__ == "__main__":
    from data_preprocessing import load_preprocess_data
    dataset = load_preprocess_data('../data/movielens_dataset.csv')
    model = build_train_model(dataset)
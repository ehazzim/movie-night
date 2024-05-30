from flask import Flask, request, jsonify, render_template
import pandas as pd
from data_preprocessing import load_preprocess_data
from model_building import build_train_model
from recommendation import get_movie_reco

app = Flask(__name__)

# loading data and model at startup
dataset = load_preprocess_data('./data/movielens_dataset.csv')
model = build_train_model
data = pd.read_csv('./data/movielens_dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommendations = get_movie_reco(model, user_id, data)
    recommendations_list = recommendations[['movieId', 'est_rating']].to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations_list)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from model_building import build_train_model
from recommendation import get_movie_recommendations

app = Flask(__name__)

# Load data and model once at startup
dataset, movies, ratings, tags, links = load_and_preprocess_data()
model = build_train_model(dataset)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommendations = get_movie_recommendations(model, user_id, ratings, movies)
    recommendations_list = recommendations[['movieId', 'title', 'est_rating']].to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations_list)

if __name__ == '__main__':
    app.run(debug=True)
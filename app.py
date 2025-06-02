from flask import Flask, render_template, request, jsonify
import json
import time
import pickle as pkl
import numpy as np
from algorithms.apriori import Apriori
from algorithms.hashtree_apriori import HashTreeApriori
from algorithms.fp_growth import FPGrowth
from algorithms.content_based_filtering import ContentBasedFiltering
from algorithms.colab_filtering import ColabFiltering
from utils.preprocess import create_transactions
import pandas as pd

app = Flask(__name__)

# Global variables to store data
transaction_data = None
movie_mapping_1m = None
movie_mapping_20m = None
content_based_model = None
collaborative_model = None
user_item_matrix = None
user_id_list = None
movie_id_list = None
movies_df = None
ratings_df = None

def load_movie_mapping():
    """Load movie ID to name mapping for both datasets"""
    global movies_df, movie_mapping_1m, movie_mapping_20m
    
    # Load 1M dataset for frequent itemset mining
    try:
        movies_1m_df = pd.read_csv('data/movielens_1m/movies.csv')
        movie_mapping_1m = dict(zip(movies_1m_df['movieId'], movies_1m_df['title']))
        print(f"Loaded {len(movie_mapping_1m)} movies from 1M dataset")
    except Exception as e:
        print(f"Failed to load 1M movies: {e}")
        movie_mapping_1m = {}
    
    # Load 20M dataset for recommendation systems
    try:
        movies_df = pd.read_csv('data/movielens_20m/movie.csv')
        movie_mapping_20m = dict(zip(movies_df['movieId'], movies_df['title']))
        print(f"Loaded {len(movie_mapping_20m)} movies from 20M dataset")
    except Exception as e:
        print(f"Failed to load 20M movies: {e}")
        movies_df = None
        movie_mapping_20m = {}
    
    # movie_mapping = movie_mapping_20m if movie_mapping_20m else movie_mapping_1m

def load_transactions():
    """Load transaction data from 1M dataset for frequent itemset mining"""
    global transaction_data
    try:
        transaction_data = create_transactions('data/movielens_1m/ratings.csv',
                                               save_path='data/movielens_1m/transaction.pkl')
        print(f"Loaded {len(transaction_data)} transactions from 1M dataset")
    except Exception as e:
        print(f"Failed to load transactions from 1M dataset: {e}")
        transaction_data = []

def load_recommendation_models():
    """Load pre-trained recommendation models and data"""
    global content_based_model, collaborative_model, user_item_matrix, user_id_list, movie_id_list, ratings_df
    
    try:
        # Load content-based filtering model
        content_based_model = ContentBasedFiltering(
            user_embedding_path='data/movielens_20m/user_embedding.pkl', 
            item_embedding_path='data/movielens_20m/movie_embedding.pkl',
            user_item_interaction_path='data/movielens_20m/user_movies.pkl'
        )
        print("Content-based filtering model loaded successfully")
    except Exception as e:
        print(f"Failed to load content-based filtering model: {e}")
        content_based_model = None
    
    try:
        # Load collaborative filtering data
        with open('data/movielens_20m/user_item_matrix.pkl', 'rb') as f:
            user_item_matrix = pkl.load(f)
        with open('data/movielens_20m/user_id_list.pkl', 'rb') as f:
            user_id_list = pkl.load(f)
        with open('data/movielens_20m/movie_id_list.pkl', 'rb') as f:
            movie_id_list = pkl.load(f)
        
        # Initialize and fit collaborative filtering model
        collaborative_model = ColabFiltering(strategy='item_based', similarity_metric='cosine', k=5)
        # Load inferred user-item matrix
        collaborative_model.user_item_matrix = user_item_matrix.copy()
        collaborative_model.mask_matrix = ~np.isnan(collaborative_model.user_item_matrix)
        print("Collaborative filtering model loaded and fitted successfully")
    except Exception as e:
        print(f"Failed to load collaborative filtering model: {e}")
        collaborative_model = None
    
    try:
        # Load ratings data for getting user's top rated movies
        ratings_df = pd.read_csv('data/movielens_20m/ratings.csv')
        print("Ratings data loaded successfully")
    except Exception as e:
        print(f"Failed to load ratings data: {e}")
        ratings_df = None

def get_top_rated_movies_for_user(user_id, top_n=10):
    """Get top rated movies for a specific user"""
    global ratings_df, movies_df
    
    if ratings_df is None or movies_df is None:
        return []
    
    try:
        # Try to load movies_processed.csv for genres
        try:
            movies_processed_df = pd.read_csv('data/movielens_20m/movies_processed.csv')
        except:
            movies_processed_df = movies_df
        
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        user_ratings = user_ratings.sort_values(by='rating', ascending=False)
        
        top_rated = user_ratings.head(top_n)
        
        result = pd.merge(top_rated, movies_processed_df, on='movieId', how='left')
        
        # Ensure we have the required columns
        if 'genres' not in result.columns and 'genres' in movies_df.columns:
            result = pd.merge(result, movies_df[['movieId', 'genres']], on='movieId', how='left')
        
        result = result[['movieId', 'title', 'genres', 'rating']].fillna('Unknown')
        
        return result.to_dict('records')
    except Exception as e:
        print(f"Error getting top rated movies: {e}")
        return []

def convert_to_implicit_user_id(user_id):
    """Convert explicit user ID to implicit (0-indexed)"""
    return user_id - 1

def get_movie_genres(movie_ids):
    """Get genres for a list of movie IDs"""
    global movies_df
    
    if movies_df is None:
        return {movie_id: 'Unknown' for movie_id in movie_ids}
    
    try:
        # Try to load movies_processed.csv for genres
        try:
            movies_processed_df = pd.read_csv('data/movielens_20m/movies_processed.csv')
            genre_mapping = dict(zip(movies_processed_df['movieId'], movies_processed_df['genres']))
        except:
            genre_mapping = dict(zip(movies_df['movieId'], movies_df.get('genres', 'Unknown')))
        
        return {movie_id: genre_mapping.get(movie_id, 'Unknown') for movie_id in movie_ids}
    except Exception as e:
        print(f"Error getting movie genres: {e}")
        return {movie_id: 'Unknown' for movie_id in movie_ids}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_user_profile', methods=['POST'])
def get_user_profile():
    """Get user's top rated movies"""
    try:
        data = request.json
        user_id = int(data['user_id'])
        top_n = int(data.get('top_n', 10))
        
        top_movies = get_top_rated_movies_for_user(user_id, top_n)
        
        return jsonify({
            'success': True,
            'top_movies': top_movies,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations from both content-based and collaborative filtering"""
    try:
        data = request.json
        user_id = int(data['user_id'])
        top_k = int(data.get('top_k', 10))
        
        implicit_user_id = convert_to_implicit_user_id(user_id)
        
        recommendations = {
            'content_based': [],
            'collaborative': []
        }
        
        # Content-based recommendations
        if content_based_model is not None:
            try:
                cb_movie_ids = content_based_model.recommend(user_id=implicit_user_id, top_k=top_k)
                cb_genres = get_movie_genres(cb_movie_ids)
                
                recommendations['content_based'] = [
                    {
                        'movieId': movie_id,
                        'title': movie_mapping_20m.get(movie_id, f"Movie {movie_id}"),
                        'genres': cb_genres.get(movie_id, 'Unknown')
                    }
                    for movie_id in cb_movie_ids
                ]
            except Exception as e:
                print(f"Content-based recommendation error: {e}")
        
        # Collaborative filtering recommendations
        if collaborative_model is not None and user_id_list is not None and movie_id_list is not None:
            try:
                cf_movie_implicit_ids = collaborative_model.recommend(user_id=implicit_user_id, top_k=top_k)
                cf_movie_ids = [movie_id_list[i] for i in cf_movie_implicit_ids]
                cf_genres = get_movie_genres(cf_movie_ids)
                
                recommendations['collaborative'] = [
                    {
                        'movieId': movie_id,
                        'title': movie_mapping_20m.get(movie_id, f"Movie {movie_id}"),
                        'genres': cf_genres.get(movie_id, 'Unknown')
                    }
                    for movie_id in cf_movie_ids
                ]
            except Exception as e:
                print(f"Collaborative filtering recommendation error: {e}")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    try:
        data = request.json
        algorithm = data['algorithm']
        num_transactions = int(data['num_transactions'])
        min_support = float(data['min_support'])
        min_confidence = float(data['min_confidence'])
        
        # Limit transactions
        limited_transactions = transaction_data[:num_transactions] if transaction_data else []
        
        start_time = time.time()
        
        if algorithm == 'apriori':
            alg = Apriori(min_support=min_support, min_confidence=min_confidence)
            alg.fit(limited_transactions)
            
        elif algorithm == 'hashtree_apriori':
            max_leaf_size = int(data.get('max_leaf_size', 64))
            max_depth = int(data.get('max_depth', 4))
            alg = HashTreeApriori(
                min_support=min_support, 
                min_confidence=min_confidence,
                max_leaf_size=max_leaf_size,
                max_depth=max_depth
            )
            alg.fit(limited_transactions)
            
        elif algorithm == 'fp_growth':
            alg = FPGrowth(min_support=min_support, min_confidence=min_confidence)
            alg.fit(limited_transactions)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Format results using 1M dataset movie mapping
        frequent_itemsets = []
        for itemset, support in alg.get_frequent_itemsets():
            # Convert movie IDs to names using 1M dataset mapping
            movie_names = []
            for movie_id in itemset:
                movie_name = movie_mapping_1m.get(movie_id, f"Movie {movie_id}")
                movie_names.append(movie_name)
            
            frequent_itemsets.append({
                'itemset': movie_names,
                'support': round(support, 4),
                'size': len(itemset)
            })
        
        return jsonify({
            'success': True,
            'frequent_itemsets': frequent_itemsets,
            'runtime': round(runtime, 2),
            'total_itemsets': len(frequent_itemsets)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("Loading data...")
    load_movie_mapping()
    load_transactions()
    load_recommendation_models()
    print(f"Loaded {len(transaction_data) if transaction_data else 0} transactions from 1M dataset")
    print(f"Loaded {len(movie_mapping_1m) if movie_mapping_1m else 0} movie mappings from 1M dataset")
    print(f"Loaded {len(movie_mapping_20m) if movie_mapping_20m else 0} movie mappings from 20M dataset")
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001)
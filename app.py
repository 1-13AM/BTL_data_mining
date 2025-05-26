from flask import Flask, render_template, request, jsonify
import json
import time
from algorithms.apriori import Apriori
from algorithms.hashtree_apriori import HashTreeApriori
from algorithms.fp_growth import FPGrowth
from utils.preprocess import create_transactions
import pandas as pd

app = Flask(__name__)

# Global variables to store data
transaction_data = None
movie_mapping = None

def load_movie_mapping():
    """Load movie ID to name mapping"""
    global movie_mapping
    try:
        movies_df = pd.read_csv('data/movielens_1m/movies.csv')
        movie_mapping = dict(zip(movies_df['movieId'], movies_df['title']))
    except:
        try:
            movies_df = pd.read_csv('data/movielens_20m/movie.csv')
            movie_mapping = dict(zip(movies_df['movieId'], movies_df['title']))
        except:
            movie_mapping = {}

def load_transactions():
    """Load transaction data"""
    global transaction_data
    try:
        transaction_data = create_transactions('data/movielens_1m/ratings.csv',
                                               save_path='data/movielens_1m/transaction.pkl')
    except:
        try:
            transaction_data = create_transactions('data/movielens_20m/ratings.csv',
                                                   save_path='data/movielens_20m/transaction.pkl')
        except:
            transaction_data = []

@app.route('/')
def index():
    return render_template('index.html')

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
        
        # Format results
        frequent_itemsets = []
        for itemset, support in alg.get_frequent_itemsets():
            # Convert movie IDs to names
            movie_names = []
            for movie_id in itemset:
                movie_name = movie_mapping.get(movie_id, f"Movie {movie_id}")
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
    print(f"Loaded {len(transaction_data) if transaction_data else 0} transactions")
    print(f"Loaded {len(movie_mapping) if movie_mapping else 0} movie mappings")
    app.run(debug=True, host='0.0.0.0', port=5001)
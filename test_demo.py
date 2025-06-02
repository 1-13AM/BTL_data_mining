#!/usr/bin/env python3
"""
Test script to verify the demo setup works correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test that all required data files can be loaded"""
    print("Testing data loading...")
    
    # Test 1M dataset files
    print("\n1. Testing 1M dataset files:")
    try:
        import pandas as pd
        movies_1m = pd.read_csv('data/movielens_1m/movies.csv')
        print(f"✓ 1M movies.csv: {len(movies_1m)} movies")
        
        ratings_1m = pd.read_csv('data/movielens_1m/ratings.csv')
        print(f"✓ 1M ratings.csv: {len(ratings_1m)} ratings")
    except Exception as e:
        print(f"✗ Error loading 1M dataset: {e}")
    
    # Test 20M dataset files
    print("\n2. Testing 20M dataset files:")
    try:
        movies_20m = pd.read_csv('data/movielens_20m/movie.csv')
        print(f"✓ 20M movie.csv: {len(movies_20m)} movies")
        
        ratings_20m = pd.read_csv('data/movielens_20m/ratings.csv')
        print(f"✓ 20M ratings.csv: {len(ratings_20m)} ratings")
    except Exception as e:
        print(f"✗ Error loading 20M dataset: {e}")
    
    # Test recommendation system files
    print("\n3. Testing recommendation system files:")
    required_files = [
        'data/movielens_20m/user_embedding.pkl',
        'data/movielens_20m/movie_embedding.pkl',
        'data/movielens_20m/user_item_matrix.pkl',
        'data/movielens_20m/user_id_list.pkl',
        'data/movielens_20m/movie_id_list.pkl',
        'data/movielens_20m/user_movies.pkl'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file_path}: {size_mb:.1f} MB")
        else:
            print(f"✗ Missing: {file_path}")

def test_algorithms():
    """Test that algorithms can be imported"""
    print("\n4. Testing algorithm imports:")
    
    try:
        from algorithms.apriori import Apriori
        print("✓ Apriori algorithm")
    except Exception as e:
        print(f"✗ Apriori algorithm: {e}")
    
    try:
        from algorithms.hashtree_apriori import HashTreeApriori
        print("✓ HashTree Apriori algorithm")
    except Exception as e:
        print(f"✗ HashTree Apriori algorithm: {e}")
    
    try:
        from algorithms.fp_growth import FPGrowth
        print("✓ FP-Growth algorithm")
    except Exception as e:
        print(f"✗ FP-Growth algorithm: {e}")
    
    try:
        from algorithms.content_based_filtering import ContentBasedFiltering
        print("✓ Content-based filtering algorithm")
    except Exception as e:
        print(f"✗ Content-based filtering algorithm: {e}")
    
    try:
        from algorithms.colab_filtering import ColabFiltering
        print("✓ Collaborative filtering algorithm")
    except Exception as e:
        print(f"✗ Collaborative filtering algorithm: {e}")

def test_app_setup():
    """Test that the Flask app can be set up"""
    print("\n5. Testing Flask app setup:")
    
    try:
        from app import load_movie_mapping, load_transactions
        
        print("Testing movie mapping loading...")
        load_movie_mapping()
        print("✓ Movie mappings loaded successfully")
        
        print("Testing transaction loading...")
        load_transactions()
        print("✓ Transactions loaded successfully")
        
    except Exception as e:
        print(f"✗ Flask app setup error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Data Mining Algorithms Demo - Setup Test")
    print("=" * 60)
    
    test_data_loading()
    test_algorithms()
    test_app_setup()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("If all tests pass, you can run: python app.py")
    print("=" * 60) 
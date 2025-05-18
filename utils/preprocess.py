# contains functions for data preprocessing
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def concatenate_movie_tags(
    movies_file:str='data/movie.csv',
    user_tags_file:str='data/tag.csv',
    genome_scores_file:str='data/genome_scores.csv',
    genome_tags_file:str='data/genome_tags.csv',
    min_relevance:float=0.2
) -> pd.DataFrame:
    """
    Concatenate user tags and genome tags to movie data.
    
    Parameters:
    -----------
    movies_file : str
        Path to movie.csv file
    user_tags_file : str
        Path to tag.csv file containing user-provided tags
    genome_scores_file : str
        Path to genome_scores.csv file
    genome_tags_file : str
        Path to genome_tags.csv file
    min_relevance : float
        Minimum relevance score to include a genome tag (default: 0.2)
        
    Returns:
    --------
    pandas.DataFrame
        Movies dataframe with concatenated tag columns
    """
    # Load DataFrames
    movies_df = pd.read_csv(movies_file)
    user_tags_df = pd.read_csv(user_tags_file)
    genome_scores_df = pd.read_csv(genome_scores_file)
    genome_tags_info_df = pd.read_csv(genome_tags_file)
    
    # process user tags
    user_tags_aggregated = (user_tags_df.groupby('movieId')['tag']
                           .apply(lambda x: '|'.join(x.astype(str)))
                           .reset_index()
                           .rename(columns={'tag': 'user_tags'}))
    
    # process genome tags
    genome_tags_merged = pd.merge(genome_scores_df, genome_tags_info_df, on='tagId')
    genome_tags_filtered = genome_tags_merged[genome_tags_merged['relevance'] >= min_relevance]
    genome_tags_aggregated = (genome_tags_filtered.groupby('movieId')['tag']
                             .apply(lambda x: '|'.join(x.astype(str)))
                             .reset_index()
                             .rename(columns={'tag': 'genome_tags'}))
    
    # merge tag columns into movie dataframe
    result_df = (movies_df
                .merge(user_tags_aggregated, on='movieId', how='left')
                .merge(genome_tags_aggregated, on='movieId', how='left'))
    
    # fill nan values with empty strings
    result_df['user_tags'] = result_df['user_tags'].fillna('')
    result_df['genome_tags'] = result_df['genome_tags'].fillna('')
    
    return result_df

def summarize_missing_values(file_paths:dict[str, str]) -> pd.DataFrame:
    """
    Create a summary table of missing values for each file.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary mapping file names to their file paths
        
    Returns:
    --------
    pandas.DataFrame
        Summary table of missing values
    """
    summary_rows = []
    
    for name, path in file_paths.items():
        # load the dataframe
        df = pd.read_csv(path)
        
        # calculate overall stats
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        missing_cells = df.isna().sum().sum()
        
        # calculate per-column missing values
        col_missing = df.isna().sum()
        col_missing_pct = col_missing / total_rows * 100
        
        # column details formatted as strings
        column_details = [f"{col}: {miss}/{total_rows} ({pct:.1f}%)" 
                         for col, miss, pct in zip(col_missing.index, 
                                                 col_missing.values, 
                                                 col_missing_pct.values) 
                         if miss > 0]
        
        if not column_details:
            column_details = ["No missing values"]
            
        # Create summary row
        summary_rows.append({
            'File': name,
            'Total Rows': total_rows,
            'Total Columns': total_cols,
            'Missing Cells': f"{missing_cells}/{total_cells} ({missing_cells/total_cells*100:.1f}%)",
            'Columns with Missing Values': len([c for c in col_missing.values if c > 0]),
            'Details': '\n'.join(column_details)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df

def create_user_item_matrix(
    ratings_file:str='data/ratings.csv',
    movies_file:str='data/movies.csv'
    # min_ratings:int=50
) -> np.ndarray:
    """
    return a user-item matrix with shape (num_users, num_items)
    """
    ratings = pd.read_csv(ratings_file, nrows=100000)
    
    
    # filter movies with less than min_ratings
    # movies_with_ratings = ratings['movieId'].value_counts() >= min_ratings
    # movies_with_ratings = movies_with_ratings[movies_with_ratings].index.tolist()
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    
    # user_item_matrix = user_item_matrix[movies_with_ratings]
    
    return user_item_matrix.to_numpy()

def embed_text(model:SentenceTransformer, text:str) -> np.ndarray:
    return model.encode(text)
    
    
    

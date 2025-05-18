import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Calculate the cosine similarity between 2 2-d arrays
    Args:
        a: np.ndarray, the first array # (a, d)
        b: np.ndarray, the second array # (b, d)
    Returns:
        float, the cosine similarity between pairwise rows of 2 arrays # (a, b)
    """
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    return a @ b.T / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)

def pearson_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise Pearson correlation between rows of two 2D arrays
    using np.corrcoef. Handles NaNs by pairwise deletion.
    Returns 0 for pairs with undefined correlation (e.g., zero variance or <2 common points).

    Args:
        a: np.ndarray, the first array, shape (n_a, d)
        b: np.ndarray, the second array, shape (n_b, d)

    Returns:
        np.ndarray, the Pearson correlation matrix of shape (n_a, n_b).
    """
    n_a, d_a = a.shape
    n_b, d_b = b.shape
    
    stacked_arr = np.vstack((a, b))

    # suppress runtime warnings for division by zero or invalid values
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix_full = np.corrcoef(stacked_arr)

    # extract the relevant block: correlations between rows of 'a' and rows of 'b'
    correlation_matrix_with_nans = corr_matrix_full[:n_a, n_a:]

    # replace NaNs (from undefined correlations) with 0.0.
    correlation_matrix = np.nan_to_num(correlation_matrix_with_nans, nan=0.0)
    
    return correlation_matrix

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 7]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(pearson_correlation(a, b))
    

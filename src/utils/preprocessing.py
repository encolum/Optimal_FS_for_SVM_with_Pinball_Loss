import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold

def standardize_data(X_train, X_test):
    """
    Standardize training and test data using the StandardScaler
    
    Parameters:
    -----------
    X_train : numpy array
        Training feature matrix
    X_test : numpy array
        Test feature matrix
        
    Returns:
    --------
    X_train_scaled : numpy array
        Standardized training data
    X_test_scaled : numpy array
        Standardized test data
    scaler : StandardScaler
        Fitted scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def create_kfolds(X, y, n_splits=10, random_state=42):
    """
    Create k-fold indices for cross-validation
    
    Parameters:
    -----------
    X : numpy array
        Feature matrix
    y : numpy array
        Target labels
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    list of tuples
        Each tuple contains (train_indices, val_indices) for a fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(X,y))
    return folds

def calculate_f_score(X, y):
    """
    Calculate F-score for feature selection
    
    Parameters:
    -----------
    X : numpy array
        Feature matrix
    y : numpy array
        Target labels (-1/1)
        
    Returns:
    --------
    f_scores : numpy array
        F-score for each feature
    """
    pos_indices = y == 1
    neg_indices = y == -1
    
    X_positive = X[pos_indices]
    X_negative = X[neg_indices]
    
    n_positive = X_positive.shape[0]
    n_negative = X_negative.shape[0]
    
    pos_mean = np.mean(X_positive, axis=0)
    neg_mean = np.mean(X_negative, axis=0)
    total_mean = np.mean(X, axis=0)
    
    # Handle division by zero when calculating variance
    variance_positive = np.sum((X_positive - pos_mean)**2, axis=0) / (n_positive - 1) if n_positive > 1 else 1e-10
    variance_negative = np.sum((X_negative - neg_mean)**2, axis=0) / (n_negative - 1) if n_negative > 1 else 1e-10
    
    numerator = (pos_mean - total_mean)**2 + (neg_mean - total_mean)**2
    denominator = variance_positive + variance_negative
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, np.inf, denominator)
    f_scores = numerator / denominator
    
    # Replace NaN and inf values with 0
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return f_scores

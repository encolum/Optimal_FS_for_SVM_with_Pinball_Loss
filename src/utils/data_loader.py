import os
import pandas as pd
import numpy as np

def load_dataset(dataset_name, dataset_type="original"):
    """
    Load a dataset with specified type (original, noise, outlier, or both)
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('wdbc', 'diabetes', 'cleveland', 'ionosphere', 'sonar', 'australia')
    dataset_type : str
        Type of dataset ('original', 'noise', 'outlier', 'both')
        
    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Target labels (-1/1)
    """
    
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset')
    
    dataset_paths = {
        'wdbc': {
            'original': os.path.join(base_dir, 'Dataset', 'wdbc.data.txt'),
            'noise': os.path.join(base_dir, 'Dataset', 'wdbc_noisy_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'Dataset', 'wdbc_noisy_label_outlier.txt'),
            'both': os.path.join(base_dir, 'Dataset', 'wdbc_both_noise_outlier.txt')
        },
        'diabetes': {
            'original': os.path.join(base_dir, 'Dataset', 'diabetes.csv'),
            'noise': os.path.join(base_dir, 'Dataset', 'diabetes_noise_label_feature.csv'),
            'outlier': os.path.join(base_dir, 'Dataset', 'diabetes_outlier.csv'),
            'both': os.path.join(base_dir, 'Dataset', 'diabetes_both_noise_outlier.csv')
        },
        'cleveland': {
            'original': os.path.join(base_dir, 'Dataset', 'Heart_disease_cleveland_new.csv'),
            'noise': os.path.join(base_dir, 'Dataset', 'clevaland_noise_label_feature.csv'),
            'outlier': os.path.join(base_dir, 'Dataset', 'clevaland_outlier.csv'),
            'both': os.path.join(base_dir, 'Dataset', 'cleveland_both_noise_outlier.csv')
        },
        'ionosphere': {
            'original': os.path.join(base_dir, 'Dataset', 'ionosphere.data'),
            'noise': os.path.join(base_dir, 'Dataset', 'ionosphere_noise_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'Dataset', 'ionosphere_outlier.txt'),
            'both': os.path.join(base_dir, 'Dataset', 'ionosphere_both_noise_outlier.txt')
        },
        'sonar': {
            'original': os.path.join(base_dir, 'Dataset', 'sonar.txt'),
            'noise': os.path.join(base_dir, 'Dataset', 'sonar_noise_label_feature.txt'),
            'outlier': os.path.join(base_dir, 'Dataset', 'sonar_outlier.txt'),
            'both': os.path.join(base_dir, 'Dataset', 'sonar_both_noise_outlier.txt')
        },
        'colon': {
            'original': os.path.join(base_dir, 'Dataset', 'colon.csv'),
            'noise': os.path.join(base_dir, 'Dataset', 'colon_noise_label_feature.csv'),
            'outlier': os.path.join(base_dir, 'Dataset', 'colon_outlier.csv'),
            'both': os.path.join(base_dir, 'Dataset', 'colon_both_noise_outlier.csv')
        }
    }
    
    try:
        file_path = dataset_paths[dataset_name][dataset_type]
        
        # Process different dataset formats
        if dataset_name == 'wdbc':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y == 'M', 1, -1)  # Convert B/M to -1/1 ----> 1:Maglinant, -1:Benign
        
        elif dataset_name == 'diabetes':
            df = pd.read_csv(file_path)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            y = np.where(y == 0, -1, 1)  # Convert 0/1 to -1/1 ----> 1:Diabetes, -1:Non-Diabetes
        
        elif dataset_name == 'cleveland':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[1:, 0:13].values.astype(float)
            y = df.iloc[1:, 13].values.astype(float)
            y = np.where(y == 0, -1, 1)  # Convert 0/1 to -1/1 ----> 1:Heart Disease, -1:No Heart Disease
        
        elif dataset_name == 'ionosphere':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            y = np.where(y == 'g', 1, -1)  # Convert g/b to 1/-1 ----> 1:Good, -1:Bad
        
        elif dataset_name == 'sonar':
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, 0:60].values
            y = df.iloc[:, 60].values
            y = np.where(y == 'M', 1, -1)  # Convert to -1/1 ----> 1:Mine, -1:Rock
        
        elif dataset_name == 'colon':
            df = pd.read_csv(file_path, header = None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            y = np.where(y == 2, -1, 1) 
        return X, y
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name} ({dataset_type}): {e}")
        print(f"Please ensure the Dataset folder exists at: {base_dir}")
        print(f"And the file exists at: {dataset_paths.get(dataset_name, {}).get(dataset_type, 'Unknown path')}")
        
        # Return empty data as fallback
        return np.array([]), np.array([])
    
def get_shape(dataset_name, dataset_type="original"):

    """
    Get the shape of the dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('wdbc', 'diabetes', 'cleveland', 'ionosphere', 'sonar', 'colon')
    """
    
    X, y = load_dataset(dataset_name, dataset_type)
    return X.shape[0], X.shape[1]
def get_ratio_class(dataset_name, dataset_type="original"):
    """
    Get the ratio of classes in the dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('wdbc', 'diabetes', 'cleveland', 'ionosphere', 'sonar', 'colon')
    
    Returns:
    --------
    dict
        Dictionary containing the ratio of classes
    """
    
    X, y = load_dataset(dataset_name, dataset_type)
    
    if y.size == 0:
        return {}
    
    count_1 = np.sum(y == 1)
    count_neg1 = np.sum(y == -1)
    
    return f'Ratio of positive to negative samples: {count_1}/{count_neg1}'
if __name__ == "__main__":
    # Example usage
    dataset_name = 'sonar'
    dataset_type = 'original'
    
    X, y = load_dataset(dataset_name, dataset_type)
    
    if X.size > 0 and y.size > 0:
        print(f"Loaded {dataset_name} ({dataset_type}) dataset successfully.")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target labels shape: {y.shape}")
        print(f'Values in y: {np.unique(y)}')
        print(get_ratio_class(dataset_name, dataset_type))
    else:
        print("Failed to load the dataset.")
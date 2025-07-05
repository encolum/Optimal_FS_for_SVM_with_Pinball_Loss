import numpy as np
import time
from docplex.mp.model import Model
from sklearn.metrics import accuracy_score, roc_auc_score

class L1SVM:
    """
    L1-SVM: Support Vector Machine with L1 regularization
    """
    
    def __init__(self, C=1.0, time_limit=None):
        """
        Initialize L1-SVM model
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        time_limit : int or None
            Time limit for optimization in seconds
        """
        self.C = C
        self.time_limit = time_limit
        self.w = None
        self.b = None
        self.train_time = None
    
    def fit(self, X, y):
        """
        Fit the L1-SVM model
        
        Parameters:
        -----------
        X : numpy array
            Training feature matrix
        y : numpy array
            Training labels (-1/1)
            
        Returns:
        --------
        self
        """
        start_time = time.time()
        
        m, n = X.shape
        
        # Initialize the model
        model = Model(name='L1-SVM')
        if self.time_limit:
            model.set_time_limit(self.time_limit)
        
        # Define decision variables
        w = model.continuous_var_list(n, name='w')
        b = model.continuous_var(name='b')
        v = model.continuous_var_list(n, name='v', lb=0)  # L1 norm variables
        xi = model.continuous_var_list(m, lb=0, name='xi')
        
        # Objective function: L1 regularization + hinge loss
        model.minimize(model.sum(v[j] for j in range(n)) + self.C * model.sum(xi[i] for i in range(m)))
        
        # Constraints
        # Classification constraints
        for i in range(m):
            model.add_constraint(y[i] * (model.sum(w[j] * X[i, j] for j in range(n)) + b) >= 1 - xi[i])
        
        # L1 norm constraints
        for j in range(n):
            model.add_constraint(w[j] <= v[j])
            model.add_constraint(-v[j] <= w[j])
        
        # Solve the model
        solution = model.solve()
        
        # Extract solution
        if solution:
            self.w = np.array([solution.get_value(w[j]) for j in range(n)])
            self.b = solution.get_value(b)
        else:
            print("No solution found")
        
        self.train_time = time.time() - start_time
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        numpy array
            Predicted labels (-1/1)
        """
        if self.w is None:
            raise ValueError("Model not fitted yet")
        
        scores = np.dot(X, self.w) + self.b
        return np.sign(scores)

    
    def get_selected_features(self):
        """
        Get indices of selected features
        
        Returns:
        --------
        list
            Indices of selected features (1-indexed)
        """
        if self.w is None:
            raise ValueError("Model not fitted yet")
        
        return [j + 1 for j in range(len(self.w)) if abs(self.w[j]) > 1e-3]
    
    def get_num_selected_features(self):
        """
        Get the number of selected features
        
        Returns:
        --------
        int
            Number of selected features
        """
        return len(self.get_selected_features())

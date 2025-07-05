import numpy as np
import time
from docplex.mp.model import Model

class L2SVM:
    """
    Standard L2-regularized Support Vector Machine using DOCPLEX
    """
    
    def __init__(self, C= None, time_limit=None):
        """
        Initialize L2-SVM model
        
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
        Fit the L2-SVM model using DOCPLEX
        
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
        
        # Initialize model
        opt_mod = Model(name='L2-SVM')
        
        # Get dimensions
        m, n = X.shape
        
        # Define decision variables
        w = opt_mod.continuous_var_list(n, name='w')
        b = opt_mod.continuous_var(name='b')
        xi = opt_mod.continuous_var_list(m, lb=0, name='xi')
        
        # Define objective function
        opt_mod.minimize(0.5 * opt_mod.sum(w[j] ** 2 for j in range(n)) + 
                         self.C * opt_mod.sum(xi[i] for i in range(m)))
        
        # Add constraints
        for i in range(m):
            opt_mod.add_constraint(y[i] * (opt_mod.sum(w[j] * X[i, j] for j in range(n)) + b) >= 1 - xi[i])
        
        # Set time limit if specified
        if self.time_limit is not None:
            opt_mod.set_time_limit(self.time_limit)
        
        # Solve the model
        solution = opt_mod.solve()
        
        if solution:
            # Extract weights and bias
            self.w = np.array([solution.get_value(w[j]) for j in range(n)])
            self.b = solution.get_value(b)
        else:
            # Handle the case when no solution is found
            self.w = np.zeros(n)
            self.b = 0
            print("Warning: No solution found for L2-SVM")
        
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
        
        return np.sign(np.dot(X, self.w) + self.b)
    
    def get_selected_features(self):
        """
        Get indices of selected features
        For L2-SVM, features with non-zero coefficients are selected
        
        Returns:
        --------
        list
            Indices of selected features (1-indexed)
        """
        if self.w is None:
            raise ValueError("Model not fitted yet")
        
        # Return indices of features with non-zero weights (1-indexed)
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
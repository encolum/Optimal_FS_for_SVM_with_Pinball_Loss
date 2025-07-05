import numpy as np
import time
from docplex.mp.model import Model
from sklearn.metrics import accuracy_score, roc_auc_score

class PinballSVM:
    """
    Support Vector Machine with Pinball Loss (Pin-SVM)
    A robust SVM model using pinball loss function
    """
    
    def __init__(self, C=1.0, tau=0.5, time_limit=None, cpu_threads=None):
        """
        Initialize Pinball SVM model
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        tau : float
            Pinball loss parameter (0 < tau <= 1)
        time_limit : int or None
            Time limit for optimization in seconds
        """
        self.C = C
        self.tau = tau
        self.time_limit = time_limit
        self.cpu_threads = cpu_threads
        self.w = None
        self.b = None
        self.train_time = None
    
    def fit(self, X, y):
        """
        Fit the Pinball SVM model
        
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
        model = Model(name='Pinball-SVM')
        if self.time_limit:
            model.set_time_limit(self.time_limit)
        if self.cpu_threads:
            model.context.cplex_parameters.threads = self.cpu_threads
        # Define decision variables
        w = model.continuous_var_list(n, name='w')
        b = model.continuous_var(name='b')
        xi = model.continuous_var_list(m, lb=0, name='xi')
        
        # Objective function: L2 regularization + pinball loss
        model.minimize(0.5 * model.sum(w[j] ** 2 for j in range(n)) + self.C * model.sum(xi[i] for i in range(m)))
        
        # Pinball loss constraints
        for i in range(m):
            model.add_constraint(y[i] * (model.sum(w[j] * X[i,j] for j in range(n)) + b) >= 1 - xi[i])
            model.add_constraint(y[i] * (model.sum(w[j] * X[i,j] for j in range(n)) + b) <= 1 + xi[i] * (1/self.tau))
        
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

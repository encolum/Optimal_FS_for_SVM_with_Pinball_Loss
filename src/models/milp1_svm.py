import numpy as np
import time
from docplex.mp.model import Model
from sklearn.metrics import accuracy_score, roc_auc_score

class MILP1:
    """
    Mixed Integer Linear Programming SVM 1 (MILP1) - feature selection SVM based on L1-norm
    """
    
    def __init__(self, B=None, C=None, l_bound=-2, u_bound=2, time_limit=None, cpu_threads = None):
        """
        Initialize MILP1 model
        
        Parameters:
        -----------
        B : int or None
            Maximum number of features to select. If None, no restriction is applied.
        C : float
            Regularization parameter
        l_bound : float
            Lower bound for feature weights
        u_bound : float
            Upper bound for feature weights
        time_limit : int or None
            Time limit for optimization in seconds
        """
        self.B = B
        self.C = C
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.time_limit = time_limit
        self.w = None
        self.b = None
        self.cpu_threads = cpu_threads
        self.v = None  # Binary variables for feature selection
        self.train_time = None
    
    def fit(self, X, y):
        """
        Fit the MILP1 model
        
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
        B = n if self.B is None else self.B
        
        # Initialize the model
        model = Model(name='MILP1')
        if self.time_limit:
            model.set_time_limit(self.time_limit)
        if self.cpu_threads:
            model.context.cplex_parameters.threads = self.cpu_threads
        # Define decision variables
        w = model.continuous_var_list(n, name='w')
        b = model.continuous_var(name='b')
        v = model.binary_var_list(n, name='v')
        xi = model.continuous_var_list(m, lb=0, name='xi')
        z = model.continuous_var_list(n, lb=0, name='z')
        
        # Objective function: minimize the empirical risk
        model.minimize(model.sum(xi[i] for i in range(m)))
        
        # Constraints
        # Classification constraints
        for i in range(m):
            model.add_constraint(y[i] * (model.sum(w[j] * X[i,j] for j in range(n)) + b) >= 1 - xi[i])
        
        # Feature selection and bound constraints
        for j in range(n):
            model.add_constraint(w[j] <= v[j] * self.u_bound)
            model.add_constraint(w[j] >= self.l_bound * v[j])
            model.add_constraint(w[j] >= -z[j])
            model.add_constraint(w[j] <= z[j])
        
        # Limit on number of selected features
        model.add_constraint(model.sum(v[j] for j in range(n)) <= B)
        
        # Solve the model
        solution = model.solve()
        self.optimization_gap = None
        self.objective_value = None
        self.status = None
        # Extract solution
        if solution:
            self.w = np.array([solution.get_value(w[j]) for j in range(n)])
            self.b = solution.get_value(b)
            self.v = np.array([solution.get_value(v[j]) for j in range(n)])
            self.objective_value = solution.objective_value
            self.status = solution.solve_details.status
            self.optimization_gap = model.get_solve_details().gap
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

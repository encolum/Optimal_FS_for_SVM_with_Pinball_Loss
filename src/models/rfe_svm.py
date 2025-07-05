import numpy as np
import time
from docplex.mp.model import Model

class RFESVM:
    """
    Standard SVM-RFE baseline (Guyon et al.)
    Eliminates one feature per iteration based on w^2 ranking.
    """
    def __init__(self, n_features, C=1.0, time_limit=None):
        if n_features is None or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        self.n_features = n_features
        self.C = C
        self.time_limit = time_limit
        self.selected_indices = None
        self.w = None
        self.b = None
        self.train_time = None

    def _svm_train(self, X, y):
        # QP solver for L1-regularized linear SVM
        opt_mod = Model(name='L1-SVM')
        if self.time_limit:
            opt_mod.set_time_limit(self.time_limit)
        m, n = X.shape
        w = opt_mod.continuous_var_list(n, name='w')
        b = opt_mod.continuous_var(name='b')
        v = opt_mod.continuous_var_list(n, name='v', lb=0)
        xi = opt_mod.continuous_var_list(m, name='xi', lb=0)

        opt_mod.minimize(opt_mod.sum(v[j] for j in range(n)) +
                         self.C * opt_mod.sum(xi[i] for i in range(m)))

        for i in range(m):
            opt_mod.add_constraint(
                y[i] * (opt_mod.sum(w[j] * X[i, j] for j in range(n)) + b)
                >= 1 - xi[i]
            )
        for j in range(n):
            opt_mod.add_constraint(w[j] <= v[j])
            opt_mod.add_constraint(-w[j] <= v[j])

        sol = opt_mod.solve()
        
        # Initialize model
        if sol is None:
            return None, None
        w_opt = np.array([sol.get_value(w[j]) for j in range(n)])
        b_opt = sol.get_value(b)
        return w_opt, b_opt

    def fit(self, X, y):
        start = time.time()
        d = X.shape[1]
        remaining = list(range(d))
        k = min(self.n_features, d)

        # Eliminate one feature at a time
        while len(remaining) > k:
            X_sub = X[:, remaining]
            w_sub, b_sub = self._svm_train(X_sub, y)
            if w_sub is None:
                break
            # Remove least important feature
            drop_idx = int(np.argmin(w_sub**2))
            remaining.pop(drop_idx)

        self.selected_indices = remaining
        # Final training
        X_fin = X[:, remaining]
        w_fin, b_fin = self._svm_train(X_fin, y)
        self.w = np.zeros(d)
        if w_fin is not None:
            for i, idx in enumerate(remaining):
                self.w[idx] = w_fin[i]
            self.b = b_fin
        else:
            self.b = 0.0
        self.train_time = time.time() - start
        return self

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model not fitted yet")
        X_sel = X[:, self.selected_indices]
        scores = X_sel.dot(self.w[self.selected_indices]) + self.b
        return np.sign(scores)

    def get_selected_features(self):
        return [i + 1 for i in self.selected_indices]
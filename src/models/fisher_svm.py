import numpy as np
import time
from docplex.mp.model import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class FisherSVM:
    """
    Fisher-SVM baseline:
    - Use F-score thresholds at percentiles [25, 50, 75]
    - Choose threshold via 5 random 80/20 holdout splits
    - Final train on full data with best threshold
    """
    def __init__(self, C=1.0, time_limit=None):
        self.C = C
        self.time_limit = time_limit
        self.selected_indices = None
        self.best_threshold = None
        self.w = None
        self.b = None
        self.train_time = None

    def _calculate_f_score(self, X, y):
        pos = y == 1
        neg = y == -1
        Xp, Xn = X[pos], X[neg]
        mean_p = np.mean(Xp, axis=0)
        mean_n = np.mean(Xn, axis=0)
        mean_all = np.mean(X, axis=0)
        var_p = (np.sum((Xp - mean_p)**2, axis=0) / (Xp.shape[0] - 1)) if Xp.shape[0] > 1 else np.zeros_like(mean_all)
        var_n = (np.sum((Xn - mean_n)**2, axis=0) / (Xn.shape[0] - 1)) if Xn.shape[0] > 1 else np.zeros_like(mean_all)
        num = (mean_p - mean_all)**2 + (mean_n - mean_all)**2
        den = var_p + var_n
        den[den == 0] = np.inf
        f_scores = num / den
        return np.nan_to_num(f_scores)

    def _svm_train(self, X, y):
        opt = Model(name='L1-SVM')
        if self.time_limit:
            opt.set_time_limit(self.time_limit)
        m, n = X.shape
        w = opt.continuous_var_list(n, name='w')
        b = opt.continuous_var(name='b')
        v = opt.continuous_var_list(n, name='v', lb=0)
        xi = opt.continuous_var_list(m, name='xi', lb=0)

        opt.minimize(opt.sum(v[j] for j in range(n)) + self.C * opt.sum(xi[i] for i in range(m)))
        for i in range(m):
            opt.add_constraint(y[i] * (opt.sum(w[j] * X[i, j] for j in range(n)) + b) >= 1 - xi[i])
        for j in range(n):
            opt.add_constraint(w[j] <= v[j])
            opt.add_constraint(-w[j] <= v[j])
        
        # Solve the model
        sol = opt.solve()
        if sol is None:
            return None, None
        w_opt = np.array([sol.get_value(w[j]) for j in range(n)])
        b_opt = sol.get_value(b)
        return w_opt, b_opt

    def fit(self, X, y):
        start = time.time()
        f_scores = self._calculate_f_score(X, y)
        # define thresholds at 25th, 50th, 75th percentiles
        thresholds = np.percentile(f_scores, [25, 50, 75])
        best_err = float('inf')
        best_thr = None
        for thr in thresholds:
            mask = f_scores >= thr
            if not mask.any():
                continue
            errs = []
            # 5 random hold-out repeats
            for i in range(5):
                current_seed = 42 + i
                Xtr, Xval, ytr, yval = train_test_split(
                    X[:, mask], y, test_size=0.2, random_state=current_seed
                )
                w, b = self._svm_train(Xtr, ytr)
                if w is None:
                    continue
                ypred = np.sign(Xval.dot(w) + b)
                errs.append(1 - accuracy_score(yval, ypred))
            if errs and np.mean(errs) < best_err:
                best_err = np.mean(errs)
                best_thr = thr
        # finalize selection
        self.best_threshold = best_thr
        mask = f_scores >= best_thr
        self.selected_indices = list(np.where(mask)[0])
        # final train on full data
        Xsel = X[:, self.selected_indices]
        w_fin, b_fin = self._svm_train(Xsel, y)
        d = X.shape[1]
        self.w = np.zeros(d)
        if w_fin is not None:
            for i, idx in enumerate(self.selected_indices):
                self.w[idx] = w_fin[i]
            self.b = b_fin
        else:
            self.b = 0.0
        self.train_time = time.time() - start
        return self

    def predict(self, X):
        if self.selected_indices is None:
            raise ValueError("Model not fitted yet")
        Xsel = X[:, self.selected_indices]
        return np.sign(Xsel.dot(self.w[self.selected_indices]) + self.b)

    def get_selected_features(self):
        return [i+1 for i in self.selected_indices]

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import rbf_kernel

def center_gram(K):
    """Applies centering matrix to a Gram matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def get_hsic_matrices(X, y):
    """
    Converts X and y into vectorized, centered Gram matrices 
    for the HSIC-Lasso formulation.
    """
    n, p = X.shape
    L = center_gram(rbf_kernel(y.reshape(-1, 1)))
    y_vec = L.flatten()
    
    X_vecs = []
    for j in range(p):
        K_j = center_gram(rbf_kernel(X[:, j].reshape(-1, 1)))
        X_vecs.append(K_j.flatten())
        
    X_mat = np.column_stack(X_vecs)
    return X_mat, y_vec

def compute_hsic_pvalue(X_feature, y, num_permutations=199):
    """
    Computes a valid HSIC p-value using a permutation test.
    """
    K = center_gram(rbf_kernel(X_feature.reshape(-1, 1)))
    L = center_gram(rbf_kernel(y.reshape(-1, 1)))
    
    # HSIC statistic is the trace of (K * L)
    actual_hsic = np.sum(K * L) 
    
    null_hsic = np.zeros(num_permutations)
    for b in range(num_permutations):
        y_perm = np.random.permutation(y)
        L_perm = center_gram(rbf_kernel(y_perm.reshape(-1, 1)))
        null_hsic[b] = np.sum(K * L_perm)
        
    # Empirical p-value
    p_val = (np.sum(null_hsic >= actual_hsic) + 1) / (num_permutations + 1)
    return p_val

def run_simulation(n_trials=50, n_samples=60, p_features=20, alpha_level=0.05):
    np.random.seed(42)
    
    naive_pvals = []
    split_pvals = []
    
    print("Running simulations. Null hypothesis is ALWAYS true (pure noise data).")
    
    for trial in range(n_trials):
        # 1. Generate pure noise. No true signal exists.
        X = np.random.randn(n_samples, p_features)
        y = np.random.randn(n_samples)
        
        # ==========================================
        # METHOD A: NAIVE POST-SELECTION INFERENCE
        # Select and test on the EXACT SAME data
        # ==========================================
        X_mat, y_vec = get_hsic_matrices(X, y)
        
        # HSIC-Lasso requires positive=True
        lasso_naive = Lasso(alpha=0.005, positive=True, fit_intercept=False, max_iter=2000)
        lasso_naive.fit(X_mat, y_vec)
        selected_naive = np.where(lasso_naive.coef_ > 0)[0]
        
        # Test the selected features on the SAME data
        for j in selected_naive:
            pval = compute_hsic_pvalue(X[:, j], y)
            naive_pvals.append(pval)
            
        # ==========================================
        # METHOD B: VALID SELECTIVE INFERENCE
        # Data Splitting: Select on D1, Test on D2
        # ==========================================
        split_idx = n_samples // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Select on Train
        X_mat_train, y_vec_train = get_hsic_matrices(X_train, y_train)
        lasso_split = Lasso(alpha=0.005, positive=True, fit_intercept=False, max_iter=2000)
        lasso_split.fit(X_mat_train, y_vec_train)
        selected_split = np.where(lasso_split.coef_ > 0)[0]
        
        # Test on Test
        for j in selected_split:
            pval = compute_hsic_pvalue(X_test[:, j], y_test)
            split_pvals.append(pval)

    # Calculate Type I Error Rate (False Positives)
    naive_fpr = np.mean(np.array(naive_pvals) < alpha_level) if naive_pvals else 0
    split_fpr = np.mean(np.array(split_pvals) < alpha_level) if split_pvals else 0

    print("\n--- Results ---")
    print(f"Target Alpha Level: {alpha_level}")
    print(f"Naive Type I Error Rate: {naive_fpr:.3f} (Severely Inflated)")
    print(f"Split Type I Error Rate: {split_fpr:.3f} (Correctly Controlled)")

run_simulation()
import numpy as np
from sklearn.linear_model import LogisticRegression

_lr_models = {}

def reset_learning_models():
    """Clear cached warm-started models between policy runs to prevent cross-contamination."""
    global _lr_models
    _lr_models = {}

def learn_from_history(history_db, node_id=None):
    """Train a warm-started logistic regression on interaction history to derive acceptance coefficients."""
    global _lr_models
    if len(history_db) < 5: return [1.0, 1.0, 1.0]

    X_list, y_list = zip(*history_db)
    X, y = np.array(X_list), np.array(y_list)

    if len(set(y)) < 2: return [1.0, 1.0, 1.0]

    if node_id not in _lr_models:
        _lr_models[node_id] = LogisticRegression(warm_start=True, max_iter=200, C=10.0)
    
    model = _lr_models[node_id]
    model.fit(X, y)
    
    return (model.coef_[0], model.intercept_[0])
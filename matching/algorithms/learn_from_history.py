import numpy as np
from sklearn.linear_model import LogisticRegression

def learn_from_history(history_db):
    if len(history_db) < 5: return [1.0, 1.0, 1.0]

    X, y = np.array([i[0] for i in history_db]), np.array([i[1] for i in history_db])

    if len(set(y)) < 2: return [1.0, 1.0, 1.0]

    model = LogisticRegression()
    model.fit(X, y)
    
    return model.coef_[0]
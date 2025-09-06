from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.01, random_state=42)

    def train(self, X):
        self.model.fit(X)

    def score(self, X):
        return -self.model.decision_function(X)

  

from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.01, random_state=42)

    def train(self, X):
        self.model.fit(X)

    def score(self, X):
        return -self.model.decision_function(X)

class AutoencoderModel:
    def __init__(self, input_dim):
        from tensorflow.keras import layers, models
        inp = layers.Input(shape=(input_dim,))
        enc = layers.Dense(32, activation="relu")(inp)
        dec = layers.Dense(input_dim, activation="linear")(enc)
        self.model = models.Model(inp, dec)
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, X, epochs=3, batch_size=32):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)

    def score(self, X):
        recon = self.model.predict(X, verbose=0)
        return ((X - recon) ** 2).mean(axis=1)

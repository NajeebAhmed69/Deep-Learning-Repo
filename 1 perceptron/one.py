import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.errors_per_epoch = []

    def activation(self, z):
        # Unit-step activation function
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        X = np.asarray(X)
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.activation(weighted_sum)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # One weight per input feature
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for _ in range(self.epochs):
            errors = 0

            for x_i, target in zip(X, y):
                prediction = self.predict(x_i)

                # Perceptron update
                update = self.learning_rate * (target - prediction)

                self.weights += update * x_i
                self.bias += update

                if update != 0:
                    errors += 1

            self.errors_per_epoch.append(errors)

            # Stop early when every training example is classified correctly
            if errors == 0:
                break

        return self
    
    
    
    
    # Inputs: x1, x2
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# AND labels
y = np.array([0, 0, 0, 1])

model = Perceptron(learning_rate=0.1, epochs=20)
model.fit(X, y)

predictions = model.predict(X)

print("Predictions:", predictions)
print("Actual labels:", y)
print("Learned weights:", model.weights)
print("Learned bias:", model.bias)
print("Errors per epoch:", model.errors_per_epoch)


new_input = np.array([[1, 0]])

prediction = model.predict(new_input)[0]

print("Prediction for [1, 0]:", prediction)
import math
from regression_models import RegressionModel

class LogisticRegression(RegressionModel):
    def __init__(self, learning_rate=0.1, epochs=1000):
        super().__init__(learning_rate, epochs)
        self.__w = 0.0  # Encapsulated variable for weight
        self.__b = 0.0  # Encapsulated variable for bias

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, x_data, y_data):
        n = len(x_data)
        for epoch in range(self._max_iterations):
            dw = db = loss = 0
            for i in range(n):
                x = x_data[i]
                y = y_data[i]
                z = self.__w * x + self.__b
                y_hat = self.sigmoid(z)

                error = y_hat - y
                dw += error * x
                db += error

                loss += - (y * math.log(y_hat + 1e-8) + (1 - y) * math.log(1 - y_hat + 1e-8))

            dw /= n
            db /= n
            loss /= n

            self.__w -= self._learning_rate * dw
            self.__b -= self._learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f} | w: {self.__w:.4f} | b: {self.__b:.4f}")

        return self

    def predict(self, x_data):
        return [1 if self.sigmoid(self.__w * xi + self.__b) >= 0.5 else 0 for xi in x_data]

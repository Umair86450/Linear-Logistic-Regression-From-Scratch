from regression_models import RegressionModel

class LinearRegression(RegressionModel):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=0.000001):
        super().__init__(learning_rate, max_iterations, tolerance)
        self.__w = 0.0  # Encapsulated weight
        self.__b = 0.0  # Encapsulated bias

    # Method for calculating metrics (MSE, MAE, etc.)
    def mse(self, y_true, y_pred):
        n = len(y_true)
        return sum(abs(y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

    def mae(self, y_true, y_pred):
        n = len(y_true)
        return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

    def r2_score(self, y_true, y_pred):
        n = len(y_true)
        y_mean = sum(y_true) / n
        ss_tot = sum((y_true[i] - y_mean) ** 2 for i in range(n)) 
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) 
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def fit(self, x, y):
        n = len(x)
        prev_mse = float('inf')
        iteration = 0

        while iteration < self._max_iterations:
            y_pred = [self.__w * xi + self.__b for xi in x]
            dw = db = 0
            for i in range(n):
                error = y[i] - y_pred[i]
                dw += -2 * x[i] * error
                db += -2 * error 
            dw /= n
            db /= n

            self.__w -= self._learning_rate * dw
            self.__b -= self._learning_rate * db

            current_mse = self.mse(y, y_pred)
            if abs(prev_mse - current_mse) < self._tolerance:
                print(f"Converged at iteration {iteration}")
                break
            prev_mse = current_mse
            iteration += 1
        return self
    
    def predict(self, x):
        return [self.__w * xi + self.__b for xi in x]

    def print_final_model(self, x, y):
        y_pred = self.predict(x)
        final_mse = self.mse(y, y_pred)
        final_mae = self.mae(y, y_pred)
        final_r2 = self.r2_score(y, y_pred)

        print("\nFinal Model:")
        print(f"y = {self.__w:.4f} * x + {self.__b:.4f}")
        print(f"Final MSE: {final_mse:.6f}")
        print(f"Final MAE: {final_mae:.6f}")
        print(f"Final R2 Score: {final_r2:.6f}")

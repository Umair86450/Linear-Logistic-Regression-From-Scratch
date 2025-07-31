from linear_regression import LinearRegression
from logistic_regression import LogisticRegression

class MainClass:
    def __init__(self, model):
        self.model = model

    def run(self, x, y):
        print(f"\nRunning with model: {self.model.__class__.__name__}")
        self.model.fit(x, y)
        predictions = self.model.predict(x)
        print(f"Predicted values: {predictions}")

        # Output the final model and metrics for Linear Regression
        if isinstance(self.model, LinearRegression):
            self.model.print_final_model(x, y)


# Example Data for Linear Regression
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# Example Data for Logistic Regression
x_data = [1, 2, 3, 4, 5]
y_data = [0, 0, 0, 1, 1]

# Use Linear Regression
linear_model = LinearRegression(learning_rate=0.01, max_iterations=10000, tolerance=1e-6)
linear_regression = MainClass(linear_model)
linear_regression.run(x, y)

# Use Logistic Regression
logistic_model = LogisticRegression(learning_rate=0.1, epochs=1000)
logistic_regression = MainClass(logistic_model)
logistic_regression.run(x_data, y_data)

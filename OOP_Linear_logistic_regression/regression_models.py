from abc import ABC, abstractmethod

# Abstract base class
class RegressionModel(ABC):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self._learning_rate = learning_rate  # Encapsulated 
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    # Encapsulation: Getter and setter methods for learning rate
    def get_learning_rate(self):
        return self._learning_rate
    
    def set_learning_rate(self, value):
        if value > 0:
            self._learning_rate = value
        else:
            print("Learning rate must be positive. Keeping the previous value.")

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("Hello, World!")
    
    # Simple calculation
    a = 5
    b = 3
    sum_result = a + b
    print(f"The sum of {a} and {b} is {sum_result}")
    
    # List manipulation
    my_list = [1, 2, 3, 4, 5]
    print(f"Original list: {my_list}")
    my_list.append(6)
    print(f"List after appending 6: {my_list}")
    
    # Simple loop
    print("Counting from 1 to 5:")
    for i in range(1, 6):
        print(i, end=" ")
    print()  # New line

if __name__ == "__main__":
    main()

    def lightgbm_example():
        # Load sample data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Set parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbose': -1
        }
        
        # Train model
        model = lgb.train(params, train_data, num_boost_round=100)
        
        # Predict
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"LightGBM model accuracy: {accuracy:.4f}")

    if __name__ == "__main__":
        lightgbm_example()
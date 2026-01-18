# Step 1: Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data        # Features (sepal & petal measurements)
y = iris.target      # Labels (species)

# Step 3: Split the dataset into training and testing data
# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create the Machine Learning model
model = LogisticRegression(max_iter=200)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

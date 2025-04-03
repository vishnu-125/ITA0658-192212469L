# ITA0658-192212469L

**1.Find S-Algorithm**
import pandas as pd

def find_s_algorithm(training_data):
    # Initialize the most specific hypothesis
    hypothesis = ["ϕ"] * (len(training_data.columns) - 1)  # Exclude target column

    for index, row in training_data.iterrows():
        if row.iloc[-1] == "Yes":  # Fix: Use .iloc[-1] instead of row[-1]
            for i in range(len(hypothesis)):
                if hypothesis[i] == "ϕ":  
                    hypothesis[i] = row.iloc[i]  # Fix: Use .iloc[i] instead of row[i]
                elif hypothesis[i] != row.iloc[i]:  
                    hypothesis[i] = "?"  # Generalize when attributes differ

    return hypothesis

# Sample dataset (Outlook, Temperature, Humidity, Wind, PlayTennis)
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "No"],
    ["Sunny", "Warm", "High", "Weak", "Yes"]
]

# Convert data to DataFrame
columns = ["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"]
training_data = pd.DataFrame(data, columns=columns)

# Apply Find-S Algorithm
final_hypothesis = find_s_algorithm(training_data)

print("Most Specific Hypothesis Found:")
print(final_hypothesis)


**2.Candidate Elimination Algorithm**
import pandas as pd

def candidate_elimination(training_data):
    num_attributes = len(training_data.columns) - 1  # Exclude target column

    # Initialize S-hypothesis with the first positive example
    S = None
    for index, row in training_data.iterrows():
        if row.iloc[-1] == "Yes":  
            S = list(row.iloc[:-1])
            break

    # If no positive examples exist, return empty hypotheses
    if S is None:
        return None, None

    # Initialize G-hypothesis with the most general hypothesis
    G = [["?"] * num_attributes]

    # Process each example in the dataset
    for index, row in training_data.iterrows():
        instance = list(row.iloc[:-1])
        label = row.iloc[-1]

        if label == "Yes":  # Positive example
            for i in range(num_attributes):
                if S[i] != instance[i]:  
                    S[i] = "?"  # Generalize S to be consistent with this positive example
            
            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[i] == "?" or g[i] == instance[i] for i in range(num_attributes))]

        elif label == "No":  # Negative example
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == "?":
                        for value in set(training_data.iloc[:, i]):  # Consider all values for that attribute
                            if value != instance[i]:  
                                new_hypothesis = g[:]
                                new_hypothesis[i] = value
                                new_G.append(new_hypothesis)

            G = new_G
            G = [g for g in G if all(g[i] == "?" or g[i] != instance[i] for i in range(num_attributes))]

    return S, G


# Sample dataset (Outlook, Temperature, Humidity, Wind, PlayTennis)
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "No"],
    ["Sunny", "Warm", "High", "Weak", "Yes"]
]

# Convert data to DataFrame
columns = ["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"]
training_data = pd.DataFrame(data, columns=columns)

# Apply Candidate Elimination Algorithm
S_final, G_final = candidate_elimination(training_data)

print("\nMost Specific Hypothesis (S):", S_final)
print("Most General Hypothesis (G):", G_final)

**3.DT ID3 Algorithm**
import pandas as pd
import numpy as np
import math

# Function to calculate entropy
def entropy(data):
    target = data.keys()[-1]  # Get target column
    values = data[target].unique()
    entropy_value = 0

    for value in values:
        fraction = data[target].value_counts()[value] / len(data[target])
        entropy_value -= fraction * math.log2(fraction)

    return entropy_value

# Function to calculate information gain
def info_gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0

    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# Function to find the best attribute
def best_attribute(data):
    attributes = data.keys()[:-1]  # Exclude target column
    gains = {attr: info_gain(data, attr) for attr in attributes}
    return max(gains, key=gains.get)  # Attribute with highest information gain

# Function to build decision tree
def id3(data, tree=None):
    target = data.keys()[-1]  # Target column

    # If only one class remains in target, return that class
    if len(data[target].unique()) == 1:
        return data[target].unique()[0]

    # If no more attributes, return the most common class
    if len(data.columns) == 1:
        return data[target].mode()[0]

    # Find the best attribute to split on
    best = best_attribute(data)

    # Create tree with best attribute
    if tree is None:
        tree = {}
    tree[best] = {}

    # Split dataset and build subtrees
    for value in data[best].unique():
        subset = data[data[best] == value].drop(columns=[best])  # Remove split attribute
        subtree = id3(subset)
        tree[best][value] = subtree

    return tree

# Sample dataset (PlayTennis)
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Rainy", "Sunny", "Overcast", "Overcast", "Rainy"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Build decision tree using ID3
decision_tree = id3(df)

# Print the decision tree
import pprint
pprint.pprint(decision_tree)

**4.Artificial Neural Network **
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize dataset (X: inputs, y: outputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data (XOR problem)
y = np.array([[0], [1], [1], [0]])  # Expected output

# Initialize neural network parameters
input_neurons = 2  # Input layer neurons
hidden_neurons = 3  # Hidden layer neurons
output_neurons = 1  # Output layer neurons

# Randomly initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))

weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training parameters
epochs = 10000
learning_rate = 0.5

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute error
    error = y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_hidden_layer = d_predicted_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print error at intervals
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Final output after training
print("\nFinal Output After Training:")
print(predicted_output)


**K-NN Algorithm**
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (4 features)
y = iris.target  # Labels (3 classes)

# Select only the first two features for visualization
X = X[:, :2]  # Taking only the first two features

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize KNN classifier with K=5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model using only two features
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundaries
def plot_decision_boundary(X, y, model, k):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    plt.title(f"KNN Decision Boundary (K={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(X_train, y_train, knn, k)

**Logistic Regression Algorithm**
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only first two features for visualization
y = iris.target  # Labels (0, 1, or 2)

# Convert multi-class to binary classification (only classes 0 and 1 for visualization)
X = X[y < 2]
y = y[y < 2]

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, log_reg)

**Linear regression Algorithm**
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate a synthetic dataset (100 samples, 1 feature, with noise)
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = lin_reg.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization of the Regression Line
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.title("Linear Regression Model")
plt.legend()
plt.show()


**Linear and polynomial regression Algorithm**
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset (change to real dataset if needed)
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X**3 + 2 * X**2 + 7 * X + 5 + np.random.normal(0, 100, (100, 1))  # Polynomial relation with noise

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- LINEAR REGRESSION ---
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)

# --- POLYNOMIAL REGRESSION (Degree = 3) ---
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_pred_poly = poly_reg.predict(X_poly_test)

# --- MODEL PERFORMANCE ---
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Linear Regression - MSE: {mse_linear:.2f}, R² Score: {r2_linear:.2f}")
print(f"Polynomial Regression (Degree 3) - MSE: {mse_poly:.2f}, R² Score: {r2_poly:.2f}")

# --- VISUALIZATION ---
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred_linear, color='red', linewidth=2, label="Linear Regression")

# For smooth polynomial curve
X_curve = np.linspace(0, 10, 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)
y_curve_poly = poly_reg.predict(X_curve_poly)

plt.plot(X_curve, y_curve_poly, color='green', linewidth=2, label="Polynomial Regression (Degree 3)")

plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

**Estimation and Maximation Algorithm**
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic dataset
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model (GMM) using Expectation-Maximization
gmm = GaussianMixture(n_components=3, max_iter=100, random_state=42)
gmm.fit(X)

# Predict cluster assignments
labels = gmm.predict(X)

# Get cluster centers
centers = gmm.means_

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centers")
plt.title("Gaussian Mixture Model (EM Algorithm)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Print GMM parameters
print("Estimated Means:\n", gmm.means_)
print("Estimated Variances:\n", gmm.covariances_)
print("Estimated Weights:\n", gmm.weights_)

** program for the task of credit score classification**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load synthetic dataset (Replace with real credit score dataset)
data = pd.DataFrame({
    'Age': np.random.randint(20, 70, 1000),
    'Income': np.random.randint(20000, 150000, 1000),
    'Debt': np.random.randint(1000, 50000, 1000),
    'Credit_History': np.random.choice([0, 1], 1000),
    'Loan_Amount': np.random.randint(5000, 50000, 1000),
    'Credit_Score': np.random.choice([0, 1], 1000)  # 1 = Good, 0 = Bad
})

# 2️⃣ Preprocessing: No missing values, normalize numerical features
X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4️⃣ Train models: Logistic Regression & Random Forest
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)

# 6️⃣ Model Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 7️⃣ Confusion Matrix
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap="Greens")
plt.title("Random Forest - Confusion Matrix")

plt.show()

**Iris flower classification using KNN**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0 - Setosa, 1 - Versicolor, 2 - Virginica
labels = iris.target_names

# 2️⃣ Preprocessing: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4️⃣ Train KNN Classifier (Choose k=5)
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred = knn.predict(X_test)

# 6️⃣ Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=labels))

# 7️⃣ Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for KNN on Iris Dataset")
plt.show()

**Car price prediction using Python**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Load Dataset (Ensure file is present in the directory)
file_path = "car_data.csv"  # Change this to your correct file path

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ File not found: car_data.csv. Please check the file path and try again.")
    exit()

# 2️⃣ View Data
print(df.head())  # Display first few rows

# 3️⃣ Data Preprocessing: Handle Missing Values
df.dropna(inplace=True)

# 🔹 Select Features and Target Variable
X = df[['year', 'mileage', 'engine_size']]  # Choose relevant numerical features
y = df['price']  # Target variable

# 4️⃣ Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7️⃣ Make Predictions
y_pred = model.predict(X_test_scaled)

# 8️⃣ Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 9️⃣ Visualizing Predictions vs. Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Car Price Prediction: Actual vs Predicted")
plt.show()

**house price prediction**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ 🔹 Create a Synthetic Dataset (No External File Required)
np.random.seed(42)
num_samples = 500

# Generate random data for features
square_feet = np.random.randint(800, 4000, num_samples)
num_bedrooms = np.random.randint(1, 6, num_samples)
num_bathrooms = np.random.randint(1, 4, num_samples)
location_score = np.random.uniform(1, 10, num_samples)  # Location rating (1-10)

# Generate house prices with some randomness
price = (square_feet * 150) + (num_bedrooms * 50000) + (num_bathrooms * 30000) + (location_score * 10000) + np.random.randint(-50000, 50000, num_samples)

# Create DataFrame
df = pd.DataFrame({
    'square_feet': square_feet,
    'num_bedrooms': num_bedrooms,
    'num_bathrooms': num_bathrooms,
    'location_score': location_score,
    'price': price
})

print("✅ Dataset Created Successfully!")
print(df.head())

# 2️⃣ 🔹 Select Features and Target
X = df[['square_feet', 'num_bedrooms', 'num_bathrooms', 'location_score']]
y = df['price']

# 3️⃣ 🔹 Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 🔹 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ 🔹 Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6️⃣ 🔹 Make Predictions
y_pred = model.predict(X_test_scaled)

# 7️⃣ 🔹 Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8️⃣ 🔹 Visualizing Predictions vs. Actual Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction: Actual vs Predicted")
plt.show()

**Navie Bayes Algorithm**
# 1️⃣ Import Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# 2️⃣ Load the Iris Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add target labels
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
df['species'] = df['species'].map(species_map)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# 3️⃣ Define Features and Target
X = df.iloc[:, :-1]  # All columns except 'species'
y = df.iloc[:, -1]   # Target column 'species'

# 4️⃣ Split the Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Feature Scaling (Optional for Naïve Bayes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Train the Naïve Bayes Model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 7️⃣ Make Predictions
y_pred = model.predict(X_test_scaled)

# 8️⃣ Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# 9️⃣ Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=species_map.values(), yticklabels=species_map.values())
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Naïve Bayes")
plt.show()


**Mobile price classification**
# 1️⃣ Import Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2️⃣ Load the Dataset (Ensure the file is in the correct directory)
try:
    df = pd.read_csv("mobile_price.csv")  # Replace with actual dataset path
    print("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    print("❌ File not found: mobile_price.csv. Please check the file path and try again.")
    exit()

# 3️⃣ Display Dataset Information
print(df.head())  # View the first 5 rows
print(df.info())  # Dataset summary
print(df.describe())  # Statistical details

# 4️⃣ Define Features (X) and Target (y)
X = df.drop(columns=["price_range"])  # Features: Remove target column
y = df["price_range"]  # Target: Classifying price range (0,1,2,3)

# 5️⃣ Split Data into Training (80%) & Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Train the Naïve Bayes Model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 8️⃣ Make Predictions
y_pred = model.predict(X_test_scaled)

# 9️⃣ Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# 🔟 Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Mobile Price Classification")
plt.show()


**Navie bayes in python**
# 1️⃣ Import Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# 2️⃣ Load the Iris Dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# 3️⃣ Split Data into Training (80%) & Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train the Naïve Bayes Model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 6️⃣ Make Predictions
y_pred = model.predict(X_test_scaled)

# 7️⃣ Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# 8️⃣ Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Naïve Bayes")
plt.show()

**Future sales prediction**
# 1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Generate Sample Sales Data (Avoids FileNotFoundError)
data = {
    "Date": pd.date_range(start="2023-01-01", periods=100, freq='D'),
    "Sales": np.random.randint(500, 5000, 100)  # Random sales data
}
df = pd.DataFrame(data)

# 3️⃣ Convert 'Date' Column to Datetime Format
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# 4️⃣ Extract Year, Month, and Day as Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 5️⃣ Define Features (X) and Target (y)
X = df[['Year', 'Month', 'Day']]
y = df['Sales']

# 6️⃣ Split Data into Training (80%) and Testing (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 8️⃣ Make Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# 🔟 Future Sales Prediction
future_dates = pd.DataFrame({'Year': [2025], 'Month': [12], 'Day': [1]})  # Example Future Date
future_sales_pred = model.predict(future_dates)
print(f"\n📈 Predicted Sales for 2025-12-01: {future_sales_pred[0]:.2f}")

# 🔹 Plot Actual vs Predicted Sales
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, color='blue', label="Predicted Sales")
plt.plot(y_test, y_test, color='red', label="Perfect Fit")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()

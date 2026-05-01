import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Program started...")

# Load data
data = pd.read_csv("student_data.csv")

X = data[['Hours']]
y = data['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Ask input
hours = float(input("Enter study hours: "))

# Predict
predicted_marks = model.predict([[hours]])

print(f"Predicted Marks: {predicted_marks[0]:.2f}")

# Show graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.show()

input("Press Enter to exit...")
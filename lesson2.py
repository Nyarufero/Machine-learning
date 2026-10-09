from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Bigger dataset

Hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Scores =np.array([15, 30, 45, 60, 70, 80, 85, 95, 92, 98])

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Hours, Scores, test_size=0.2, random_state=42)

print(f"Training with {len(X_train)} Students")
print(f"Testing with {len(X_test)} Students")
print(f"Test hours: {X_test.flatten()}")

#train ONLY on train data
model = LinearRegression()
model.fit(X_train, y_train)

#Examine the model
y_pred = model.predict(X_test)

print("\n--- Exam Results ---")
for real_h, real_s, pred_s in zip(X_test.flatten(), y_test, y_pred):
    print(f"Hours: {real_h}, | Actual Score: {real_s}, | Predicted Score: {pred_s:.1f}")

#How good is it.
r2 = r2_score(y_test, y_pred)

print(f"\n ---MODEL EVALUATION---")
print(f"\nR2 Score: {r2:.3f}")

if r2 > 0.9:
    print("This is the BEST model!")
elif r2 > 0.7:
    print("This is a good model!")
else:
    print("This is a bad model!")
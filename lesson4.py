import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#Data Hours Studies, Attendance % -> Did they PASS? 1=Yes, 0=No
data = {
    "Hours": [1, 2, 3, 4, 5, 1, 2, 6, 7, 8, 9, 10, 3, 4, 8],
    "Attendance": [40, 55, 60, 65, 70, 30, 45, 80, 85, 90, 95, 88, 50, 75, 60],
    "Pass": [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
print(df.head())

#Split
X = df[["Hours", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train-Logistic Regression model

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\n--- EXAM ---")
for i in range(len(X_test)):
    h = X_test.iloc[i]["Hours"]
    a = X_test.iloc[i]["Attendance"]
    real = y_test.iloc[i]
    pred = y_pred[i]
    print(f"Hours: {h}, Attendance: {a}%, | Real: {'Pass' if real==1 else 'Fail'}, | Predicted: {'Pass' if pred==1 else 'Fail'}")

# Score
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.0f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("[[True Fail, False Pass],")
print(" [False Fail, True Pass]]")

# Predict NEW student
new_student = pd.DataFrame([[4, 70]], columns=["Hours", "Attendance"])
result = model.predict(new_student)[0]
prob = model.predict_proba(new_student)[0]

print(f"\nNew student: 4 Hours, 70% Attendance")
print(f"Prediction: {'Will Pass' if result==1 else 'Will Fail'}")
print(f"Probability: {prob[1]*100:.1f}% chance of passing")

#Your turn: try different student

my_hours = 2
my_att = 80
me = pd.DataFrame([[my_hours, my_att]], columns=["Hours", "Attendance"])
my_prob = model.predict_proba(me)[0][1]
print(f"\nYou: {my_hours} hrs, {my_att}% Attendance -> {my_prob*100:.0f}% chance of passing")
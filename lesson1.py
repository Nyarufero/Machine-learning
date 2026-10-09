import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#our data - this is all ML needs
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Scores": [15, 30, 45, 60, 70, 80, 85, 95]
}

df = pd.DataFrame(data)
print(df)

#pattern
plt.scatter(df["Hours"], df["Scores"])
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Does more Hours === more score?")
plt.show()

#Creating Model
model = LinearRegression()

#Training it
X = df[["Hours"]]
y = df["Scores"]
model.fit(X,y)
print("Model trained!")

#predict
prediction = model.predict([[9]])
print(f"Predicted score for 9 hours of study: {prediction[0]:.1f}")

#what learned
print(f"It learned: Score = {model.coef_[0]:.1f} * Hours + {model.intercept_:.1f}")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os 

#load CSV
csv_path = os.path.join(os.path.dirname(__file__), "houses.csv")
df = pd.read_csv(csv_path)
print("First 3 houses:")
print(df.head(3))

#Plot it
plt.scatter(df["Size"], df["Price"])
plt.xlabel("Size in sqm")
plt.ylabel("Price in $")
plt.title("Does bigger house = more expensive?")
plt.show()

#Train model with 1 feature
X = df[["Size"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score with only Size: {r2:.3f}")

#predict your dream house

my_size = 130
predicted_price = model.predict([[my_size]])
print(f"\nPredicted price for a house for {my_size} sqm house: ${predicted_price[0]:.0f}")

#Train model with 2 features
print("\n--- Training with 2 features ---")
X2 = df[["Size", "Bedrooms"]]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
r2_2 = r2_score(y2_test, model2.predict(X2_test))
print(f"R2 Score with Size+Bedrooms: {r2_2:.3f}")

#predict with both
price2 = model2.predict([[130,3]])
print(f"130 sqm, 3 bedrooms -> ${price2[0]:.0f}")
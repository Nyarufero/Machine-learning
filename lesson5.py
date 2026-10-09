import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Income": [300, 400, 500, 600, 800, 200, 250, 900, 1000, 1200, 350, 700, 150, 1100, 450],
    "Creditscore": [500, 550, 600, 650, 700, 450, 480, 720, 750, 800, 580, 680, 400, 770, 620],
    "Employed": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    "Approved": [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
X = df[["Income", "Creditscore", "Employed"]]
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Loan Model Accuracy: {acc*100:.0f}%\n")

#test a new applicant
def check_loan(income, credit_score, employed):
    applicant = pd.DataFrame([[income, credit_score, employed]], columns=["Income", "Creditscore", "Employed"])
    pred = model.predict(applicant)[0]
    prob = model.predict_proba(applicant)[0][1]
    status = "Approved" if pred == 1 else "Denied"
    print(f"Applicant: Income=${income}, Credit Score={credit_score}, Employed={employed} -> {status} (Probability of approval: {prob*100:.1f}%)")
    return status

#Try cases
check_loan(600, 650, 1)
check_loan(200, 450, 1)
check_loan(500, 600, 1)


#your turn

print("\n--- Your Applicant ---")
check_loan(750, 700, 1)
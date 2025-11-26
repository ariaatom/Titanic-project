import pandas as pd
from sklearn.linear_model import LogisticRegression

# --------------------------
# 1. Load cleaned data
# --------------------------
train_df = pd.read_csv(r"C:\Users\moamm\OneDrive\Desktop\Project\Titanic\adjusted_train.csv")
test_df = pd.read_csv(r"C:\Users\moamm\OneDrive\Desktop\Project\Titanic\adjusted_test.csv")

# --------------------------
# 2. Select features and target
# --------------------------
features = ['Age', 'Sex', 'Pclass']

X_train = train_df[features]
y_train = train_df['Survived']

X_test = test_df[features]

# --------------------------
# 3. Train Logistic Regression
# --------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------
# 4. Predict on test set
# --------------------------
predictions = model.predict(X_test)

# --------------------------
# 5. Create submission CSV
# --------------------------
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

submission.to_csv(r"C:\Users\moamm\OneDrive\Desktop\Project\Titanic\submission.csv", index=False)

print("Submission file created!")

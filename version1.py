import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------
# 1. Function to fill missing Age
# ------------------------------

def fill_age_with_regression(df):
    df_copy = df.copy()
    features = ['Sex', 'Pclass', 'Fare']
    
    # Split rows with known and unknown Age
    known_age = df_copy[df_copy['Age'].notnull()]
    unknown_age = df_copy[df_copy['Age'].isnull()]
    
    # Train regression model on known Age
    age_model = LinearRegression()
    age_model.fit(known_age[features], known_age['Age'])
    
    # Predict missing ages
    if not unknown_age.empty:
        df_copy.loc[df_copy['Age'].isnull(), 'Age'] = age_model.predict(
            unknown_age[features]
        )
    
    return df_copy, age_model, features


# ------------------------------
# 2. Load TRAIN data
# ------------------------------

df = pd.read_csv("Titanic-project/train.csv")

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# Fill missing Age in train
df, age_model, features = fill_age_with_regression(df)


# ------------------------------
# 3. Load TEST data
# ------------------------------

test_df = pd.read_csv("Titanic-project/test.csv")

test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})

# Predict missing Age using the SAME regression model
missing_age = test_df['Age'].isnull()

test_df.loc[missing_age, 'Age'] = age_model.predict(
    test_df.loc[missing_age, features]
)


# ------------------------------
# DONE — df and test_df are clean
# ------------------------------

df.to_csv('adjusted_train.csv', index=False)
test_df.to_csv('adjusted_test.csv', index=False)

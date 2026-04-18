# The dataset uses semicolons as seperators, so we need to specify that when loading the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Task 1: Load and Explore

df = pd.read_csv("student_performance_math.csv", sep=";")

print(df.shape)
print(df.head())
print(df.dtypes)

plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Grades")
plt.xlabel("G3")
plt.ylabel("Count")
plt.savefig("outputs/g3_distribution.png")
plt.close()

#Task 2: Preprocess the Data
print("Before filtering:",df.shape)

df_clean = df[df["G3"] !=0].copy()
print("After filtering:", df_clean.shape)

yes_no_columns = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_columns:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Correlation (original):", corr_original)
print("Correlation (filtered):", corr_filtered)

# Task 3: Exploratory Data Analysis

numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "freetime", "goout", "Walc"]

corrs = df_clean[numeric_cols + ["G3"]].corr()["G3"].sort_values()
print(corrs)

plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("Failures vs G3")
plt.xlabel("Failures")
plt.ylabel("G3")
plt.savefig("outputs/failures_vs_g3.png")
plt.close()

plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("Study Time vs G3")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("outputs/studytime_vs_g3.png")
plt.close()

# Task 4: Baseline Model

X = df_clean[["failures"]].values
Y = df_clean["G3"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
r2 = r2_score(Y_test, y_pred)

print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R2:", r2)

# Task 5: Build the Full Model

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Train R2:", model.score(X_train, y_train))
print("Test R2:", model.score(X_test, y_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Task 6: Evaluate and Summarize
plt.scatter(y_pred, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# Neglected Feature: The Power of G1
feature_cols.append("G1")

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

print("Test R2 with G1:", model.score(X_test, y_test))

# Does a high R² here mean G1 is causing G3? 
# No, it just means they are strongly correlated. G1 is a strong predictor of G3, but it doesn't cause G3. Both G1 and G3 could be influenced by other factors such as study habits or attendance.
# Is this a useful model for identifying students who might struggle?
# It could be useful for identifying students who might struggle, as a low G1 score could indicate a higher risk of a low G3 score. Relying solely on G1 might not capture all the factors though.  
# What might educators need to do if they wanted to intervene early, before G1 is even available?
# Other early indicators such as attendance, participation, homework, tests. Maybe also family background or social support. 
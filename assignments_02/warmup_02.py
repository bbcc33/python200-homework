# --- scikit-learn API ---

# Q1
from sklearn.linear_model import LinearRegression
import numpy as np

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 1. create model
model = LinearRegression()

# 2. fit model to data (learn)
model.fit(years, salary)

# 3. predict with new data
pred_4_years = model.predict([[4]])
pred_8_years = model.predict([[8]])


print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Predicted salary for 4 years of experience: ${pred_4_years[0]:.2f}")
print(f"Predicted salary for 8 years of experience: ${pred_8_years[0]:.2f}")

 # Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped shape:", x_2d.shape)

 # Q3
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(x_clusters)

labels = kmeans.predict(x_clusters)

print("Cluster centers:\n", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

plt.scatter(x_clusters[:, 0], x_clusters[:, 1], c=labels, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            marker="X", s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

#linear Regression
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

#LQ1
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Costs vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Costs")
plt.savefig("outputs/medical_costs.png")
plt.close()

#LQ2
X_age = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(X_age, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#LQ3
model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

#LQ4
X_full = np.column_stack([age, smoker])

Xf_train, X_test, yf_train, yf_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(Xf_train, yf_train)

print("R^2 with smoker:", model_full.score(X_test, yf_test))
print("age coefficient:", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])

#in practical terms the smoker coefficient means that being a smoker increases the predicted medical cost by approximately $15,000, smoking has a significant impact on medical costs.

y_pred_full = model_full.predict(X_test)

plt.scatter(y_pred_full, yf_test)
plt.plot([yf_test.min(), yf_test.max()], [yf_test.min(), yf_test.max()])
plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()
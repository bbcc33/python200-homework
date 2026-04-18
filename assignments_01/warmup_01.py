# --- Pandas ---
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# Pandas Q1
print("First 3 rows:")
print(df.head(3))

print(f"\nShape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

# Pandas Q2
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print("\nStudents who passed and have grade > 80:")
print(filtered_df[["name", "grade"]])

# Pandas Q3
df["grade_curved"] = df["grade"] + 5
print("\nDataFrame with curved grades:")
print(df)

# Pandas Q4
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5
mean_by_city = df.groupby("city")["grade"].mean()
print("Mean grade by city:")
print(mean_by_city)

# Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7
sorted_df = df.sort_values(by="grade", ascending=False)
print("Top 3 students by grade:")
print(sorted_df.head(3))

# --- NumPy Review ---
import numpy as np

# NumPy Q1
arr1 = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr1}")
print(f"Shape: {arr1.shape}")
print(f"Data Type: {arr1.dtype}")
print(f"Number of Dimensions: {arr1.ndim}")

# NumPy Q2
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Shape: {arr2.shape}")
print(f"Size: {arr2.size}")

# NumPy Q3
top_left_block = arr2[:2, :2]
print("Top-left 2x2 block:")
print(top_left_block)

# NumPy Q4
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 5))
print("3x4 array of zeros:")
print(zeros_array)
print("\n2x5 array of ones:")
print(ones_array)

# NumPy Q5
arr3 = np.arange(0, 50, 5)
print(f"Array: {arr3}")
print(f"Shape: {arr3.shape}")
print(f"Mean: {arr3.mean()}")
print(f"Sum: {arr3.sum()}")
print(f"Standard Deviation: {arr3.std()}")

# NumPy Q6
random_arr = np.random.normal(0, 1, 200)
print(f"Mean of random array: {random_arr.mean()}")
print(f"Standard Deviation of random array: {random_arr.std()}")

# --- Matplotlib Review ---

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.figure()
plt.plot(x,y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.figure()
plt.scatter(x1, y1, color='red', label='Dataset 1')
plt.scatter(x2, y2, color='blue', label='Dataset 2')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q4
fig, axes = plt.subplots(1, 2)

axes[0].plot(x, y)
axes[0].set_title("Squares")

axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")

plt.tight_layout()
plt.show()

# --- Descriptive Statistics Review ---

# Descriptive Stats Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

# Descriptive Stats Q2
scores = np.random.normal(65, 10, 500)
plt.figure()
plt.hist(scores, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure()
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distoribution Comparison")
plt.show()
# Exponential data is more skewed
# Median provides a more appropriate measure of central tendency for each distribution.

# Descriptive Stats Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Data1 Mean: {np.mean(data1)}, Median: {np.median(data1)}")
print(f"Data2 Mean: {np.mean(data2)}, Median: {np.median(data2)}")
# Data2 has an outlier, 150, which significantly increases the mean
# The median remains the same because the number in the middle remains unchanged

# --- Hypothesis Testing Review ---
from scipy import stats
from scipy.stats import pearsonr

# Hypothesis Q1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Hypothesis Q2
if p_value < 0.05:
    print("Result is statistically significant.")

# Hypothesis Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_value = stats.ttest_rel(before, after)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Hypothesis Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_value = stats.ttest_1samp(scores, 70)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Hypothesis Q5
t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative='less')
print(f"One-tailed P-value: {p_value}")

# Hypothesis Q6
print("Group A scores are significantly less than Group B scores. It is likely not due to chance.")

# --- Correlation Review ---

# Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
print(f"Correlation Coefficient: {corr_matrix[0, 1]}")
# I'm expecting 1, that they both increase together at the same rate.

# Correlation Q2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr, p_value = pearsonr(x, y)
print(f"Correlation: {corr}, P-value: {p_value}")

# Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())

# Correlation Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5
import seaborn as sns

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---

# Pipeline Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std()
    }

def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

result = data_pipeline(arr)
for k, v in result.items():
    print(f"{k}: {v}")


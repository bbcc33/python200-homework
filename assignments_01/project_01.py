from prefect inmport flow, task, get_run_logger
import pandas as pd
import numpy as np
import matplotilb.pyplot as plt
import seaborn as sns
import scipy.stats import ttest_ind, pearsonr
import pathlib import Path 
import re 

DATA_DIR = Path("assignments/resources/happiness_project")
OUTPUT_DIR = Path("assignments_01/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task 1: Load Multiple Years of Data

def standardize_column_names(df):
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    rename_map = {
        "country_or_region": "country",
        "country_name": "country",
        "happiness_score": "score",
        "ladder_score": "score",
        "regional_indicator": "region",
        "economy_(gdp_per_capita)": "gdp_per_capita",
    }
    df = df.rename(columns=rename_map)
    return df


@task(retries=3, retry_delay_seconds=2)
def load_and_clean_data(file_path):
    logger = get_run_logger()

    all_files = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    dfs = []

    for file in all_files:
        year = re.search(r"(\d{4})", file.name).group()

        logger.info(f"Loading {file.name} for year {year}")

        df = pd.read_csv(file, sep=",", decimal=",")
        df = standardize_columns(df)
        df["year"] = int(year)
        dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(OUTPUT_DIR / "cleaned_happiness_data.csv", index=False)

        logger.info("Meged dataset saved to outputs/merged_happiness_data.csv")
        return merged

# Task 2: Descriptive Statistics
@task
def descriptive_statistics(df):
    logger = get_run_logger()

    mean = df["score"].mean()
    median = df["score"].median()
    std = df["score"].std()

    logger.info(f"Overall Mean Score: {mean:.3f}")
    logger.info(f"Overall Median Score: {median:.3f}")
    logger.info(f"Overall Std Dev: {std:.3f}")

    by_year = df.groupby("year"["score"]).mean()
    by_region = df.grouby("region")["score"].mean()

    logger.info("Mean Score by Year:\n" + by_year.to_string())
    logger.info("Mean Score by Region:\n" + by_region.to_string())

    return by_region, by_year

# Task 3: Visual Exploration
@task
def create_visualizations(df):
    logger = get_run_logger()

    plt.figure()
    df["score"].hist()
    plt.title("Distribution of Happiness Scores")
    plt.savefig(OUTPUT_DIR / "happiness_score_distribution.png")
    logger.info("Saved histogram to outputs/happiness_score_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.savefig(OUTPUT_DIR / "happiness_scores_by_region.png")
    logger.info("Saved happiness_scores_by_region.png")
    plt.close()

# Task 4: Hypothesis Testing
@task
def hypothesis_testing(df):
    logger = get_run_logger()

    scores_2019 = df[df["year"] == 2019]["score"]
    scores_2020 = df[df["year"] == 2020]["score"]

    t_stat, p_value = ttest_ind(scores_2019, scores_2020)

    logger.info(f"2019 vs 2020 t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    logger.info(f"Mean 2019: {scores_2019.mean():.3f}")
    logger.info(f"Mean 2020: {scores_2020.mean():.3f}")

    if p_value < 0.05:
        logger.info("The difference in happiness scores between 2019 and 2020 significantly changed.")
    else:
        logger.info("There is no significant change in happiness scores between 2019 and 2020.")

    # Second Test
    region_means = df.groupby("region")["score"].mean().sort_values()
    top_region = region_means.idxmax()
    bottom_region = region_means.idxmin()

    top_scores = df[df["region"] == top_region]["score"]
    bottom_scores = df[df["region"] == bottom_region]["score"]

    t2, p2 = ttest_ind(top_scores, bottom_scores)

    logger.info(f"{top_region} vs {bottom_region} t-statistic: {t2:.4f}, p-value: {p2:.4f}")

#Task 5: Correlation and Multiple Comparisons
@task
def correlation_tests(df):
    logger = get_run_logger()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    tests = []

    for col in numeric_cols:
        if col != "score":
            corr, p = pearsonr(df["score"], df[col])
            tests.append((col, corr, p))

    n_tests = len(tests)
    adjusted_alpha = 0.05 / n_tests
    logger.info(f"Bonferroni adjusted alpha: {adjusted_alpha:.6f}")

    for col, r, p in tests:
        if p < 0.05:
            logger.info(f"{col} significant at 0.05")
        if p < adjusted_alpha:
            logger.info(f"{col} significant after Bonferroni correction")

# Task 6: Summary Report
@task
def summary_report(df, by_region):
    logger = get_run_logger()

    n_countries = df["country"].nunique()
    n_years = df["year"].nunique()

    logger.info(f"Dataset includes {n_countries} countries over {n_years} years.")

    top3 = by_region.sort_values(ascending=False).head(3)
    bottom3 = by_region.sort_values().head(3)

    logger.info("Top 3 Regions by Mean Happiness Score:\n" + top3.to_string())
    logger.info("Bottom 3 Regions by Mean Happiness Score:\n" + bottom3.to_string())

@flow
def happiness_pipeline():
    df = load_and_merge_data()
    by_region = descriptive_statistics(df)
    create_visualizations(df)
    hypothesis_testing(df)
    correlation_tests(df)
    summary_report(df, by_region)

if __name__ == "__main__":
    happiness_pipeline()
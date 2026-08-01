# Technical Documentation: Topic 1 - Exploratory Data Analysis with Pandas

## 1. Notebook Overview
This notebook is part of the mlcourse.ai open machine learning course, specifically focusing on Topic 1: Exploratory Data Analysis (EDA) with Pandas. It demonstrates the main methods of the Pandas library by analyzing a dataset on the churn rate of telecom operator clients (`telecom_churn.csv`). The objective is to introduce fundamental data manipulation, inspection, and visualization techniques, leading up to a preliminary attempt at predicting telecom customer churn without using complex machine learning algorithms. The notebook systematically guides the reader through reading data, inspecting data types and dimensions, generating descriptive statistics, sorting, indexing, applying functions, grouping, and creating pivot tables, culminating in a visual and logical baseline model for churn prediction.

## 2. Environment and Dependencies
The environment for this notebook requires Python and several core data science libraries.
- **numpy** (imported as `np`): Used for numerical operations and providing aggregation functions (e.g., `np.mean`, `np.max`).
- **pandas** (imported as `pd`): The primary library used for data structures (`Series`, `DataFrame`) and data analysis.
- **matplotlib.pyplot** (imported as `plt`): Used for setting up the plotting environment.
- **seaborn** (imported as `sns`): Used for statistical data visualization, specifically creating count plots.
Additionally, the notebook sets `pd.set_option('display.precision', 2)` to format floating-point output to two decimal places, and uses the `%config InlineBackend.figure_format = 'retina'` magic command to render high-resolution graphics.

## 3. Per-Stage Documentation

### Stage 1: Library Imports and Configuration
**[CODE]**
```python
import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)
```
**[PROCESS/CONTEXT]**
The notebook begins by importing the necessary libraries, numpy and pandas, which are foundational for numerical and tabular data manipulation in Python. The Pandas display precision is set to 2 to ensure that all floating-point numbers output by Pandas are rounded to two decimal places for better readability.
**[RESULT]**
Libraries are loaded and the pandas environment is configured.
**[INSIGHT]**
Setting display options early ensures consistent formatting throughout the notebook's output cells.

### Stage 2: Data Loading and Initial Inspection
**[CODE]**
```python
df = pd.read_csv("../input/telecom_churn.csv")
df.head()
```
**[PROCESS/CONTEXT]**
The telecom churn dataset is loaded from a CSV file into a Pandas DataFrame named `df`. The `head()` method is then used to display the first 5 rows of the DataFrame to verify successful loading and to provide an initial glance at the data structure.
**[RESULT]**
The output displays the first 5 rows (index 0 to 4) of the DataFrame, showing features like State, Account length, Area code, International plan, Voice mail plan, Number vmail messages, Total day minutes, Total day calls, Total day charge, Total eve minutes, Total eve calls, Total eve charge, Total night minutes, Total night calls, Total night charge, Total intl minutes, Total intl calls, Total intl charge, Customer service calls, and Churn (which is a boolean).
**[INSIGHT]**
The dataset contains a mix of categorical (e.g., State, International plan), integer (e.g., Account length, Total day calls), float (e.g., Total day charge), and boolean (Churn) features. Each row represents a single customer instance.

### Stage 3: Dimensionality and Feature Names
**[CODE]**
```python
print(df.shape)
print(df.columns)
```
**[PROCESS/CONTEXT]**
The shape of the DataFrame is printed to understand the overall size of the dataset (rows and columns). Then, the `columns` attribute is printed to get a list of all feature names available for analysis.
**[RESULT]**
The output of `print(df.shape)` is `(3333, 20)`. The output of `print(df.columns)` lists the 20 column names: 'State', 'Account length', 'Area code', 'International plan', 'Voice mail plan', 'Number vmail messages', 'Total day minutes', 'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls', 'Churn'.
**[INSIGHT]**
The dataset comprises exactly 3333 instances and 20 features, indicating a moderately sized dataset for initial exploration.

### Stage 4: Feature Types and Missing Values Check
**[CODE]**
```python
print(df.info())
```
**[PROCESS/CONTEXT]**
The `info()` method is utilized to print a concise summary of the DataFrame, including the number of non-null values and the data type of each column.
**[RESULT]**
The summary shows `3333 entries`. The data types are distributed as: `bool(1)`, `float64(8)`, `int64(8)`, `object(3)`. All 20 columns have exactly `3333 non-null` counts, meaning there are zero missing values in the dataset. Memory usage is approximately `498.1+ KB`.
**[INSIGHT]**
The dataset is perfectly clean in terms of missing values, requiring no imputation steps before proceeding with analysis. The data types are largely numerical, which is favorable for many machine learning models.

### Stage 5: Changing Column Types
**[CODE]**
```python
df["Churn"] = df["Churn"].astype("int64")
```
**[PROCESS/CONTEXT]**
The `Churn` feature, initially a boolean (`True`/`False`), is converted into an integer type (`1`/`0`) using the `astype()` method.
**[RESULT]**
The `Churn` column is successfully converted, altering its data type for future numerical or aggregation operations.
**[INSIGHT]**
Converting boolean targets to integers (0 and 1) is a standard practice because many machine learning algorithms and statistical functions expect numerical inputs.

### Stage 6: Descriptive Statistics for Numerical Features
**[CODE]**
```python
df.describe()
```
**[PROCESS/CONTEXT]**
The `describe()` method calculates basic statistical characteristics for all numerical columns (`int64` and `float64`), including count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum.
**[RESULT]**
A table is generated showing the statistics for 17 numerical columns (including the newly converted `Churn`). Key statistics include:
- `Account length`: mean 101.06, max 243.
- `Total day minutes`: mean 179.78, max 350.80.
- `Customer service calls`: mean 1.56, max 9.00.
- `Churn`: mean 0.14, indicating a 14% overall churn rate.
**[INSIGHT]**
The statistical summary reveals the scale and distribution of each feature. The mean of `Churn` (0.14) immediately highlights that the dataset is imbalanced, with a significant majority of customers being loyal.

### Stage 7: Descriptive Statistics for Categorical Features
**[CODE]**
```python
df.describe(include=["object", "bool"])
```
**[PROCESS/CONTEXT]**
To view statistics for non-numerical features, `describe()` is called with the `include` parameter specifying `object` and `bool` types.
**[RESULT]**
The output shows statistics for `State`, `International plan`, and `Voice mail plan`. For instance, `State` has 51 unique values, with `WV` being the most frequent (106 times). `International plan` has 2 unique values, with 'No' occurring 3010 times. `Voice mail plan` has 'No' occurring 2411 times.
**[INSIGHT]**
The vast majority of users do not have an International plan (3010 out of 3333). `State` has a uniform-like distribution across 51 unique regions (US states + DC).

### Stage 8: Target Variable Distribution
**[CODE]**
```python
df["Churn"].value_counts()
df["Churn"].value_counts(normalize=True)
```
**[PROCESS/CONTEXT]**
The `value_counts()` method is applied to the `Churn` variable to count the absolute occurrences of each class. Then, `normalize=True` is passed to calculate the relative frequency (fraction) of each class.
**[RESULT]**
The absolute counts show `0` (loyal): 2850, and `1` (churned): 483. The normalized counts show `0`: 0.86 (86%), and `1`: 0.14 (14%).
**[INSIGHT]**
Exactly 85.5% (rounded to 86% in display) of users are loyal, establishing a strong baseline accuracy for any predictive model: a naive model predicting '0' for everyone would be 85.5% accurate.

### Stage 9: Sorting Data
**[CODE]**
```python
df.sort_values(by="Total day charge", ascending=False).head()
df.sort_values(by=["Churn", "Total day charge"], ascending=[True, False]).head()
```
**[PROCESS/CONTEXT]**
The DataFrame is sorted by a single column (`Total day charge`) in descending order to find the highest chargers. Then, it is sorted by multiple columns (`Churn` ascending, then `Total day charge` descending) to find loyal customers with the highest day charges.
**[RESULT]**
The highest single day charge is 59.64 by a user from `CO`. When sorting by both columns, the loyal user with the highest day charge is from `MN` with a charge of 53.65.
**[INSIGHT]**
Sorting allows for the identification of extreme values and outliers within specific segments of the customer base.

### Stage 10: Indexing and Conditional Selection (Boolean Indexing)
**[CODE]**
```python
df["Churn"].mean()
df[df["Churn"] == 1].mean()
df[df["Churn"] == 1]["Total day minutes"].mean()
df[(df["Churn"] == 0) & (df["International plan"] == "No")]["Total intl minutes"].max()
```
**[PROCESS/CONTEXT]**
Various indexing techniques are demonstrated to answer specific analytical questions. The mean of churned users is calculated. Boolean indexing selects only rows where `Churn == 1` to find average feature values for churned users. Compound boolean conditions are used to find the maximum international minutes for loyal users without an international plan.
**[RESULT]**
- Proportion of churned users: `0.1449`
- Average `Total day minutes` for churned users: `206.91`
- Max `Total intl minutes` for loyal users without an international plan: `18.9`
**[INSIGHT]**
Churned users spend noticeably more time on the phone during the day (average 206.91 minutes) compared to the overall population mean (179.78 minutes). This indicates day minutes might be a strong predictor of churn.

### Stage 11: Loc and iLoc Indexing
**[CODE]**
```python
df.loc[0:5, "State":"Area code"]
df.iloc[0:5, 0:3]
df[-1:]
```
**[PROCESS/CONTEXT]**
The notebook demonstrates label-based indexing (`loc`) and integer-based indexing (`iloc`) to slice the DataFrame. Finally, negative indexing is used to retrieve the very last row.
**[RESULT]**
- `loc` retrieves rows 0 through 5 inclusive, and columns from 'State' to 'Area code' inclusive.
- `iloc` retrieves rows 0 through 4 (5 is exclusive) and columns 0 through 2 (3 is exclusive).
- `df[-1:]` retrieves only row index 3332.
**[INSIGHT]**
Pandas offers highly flexible data retrieval mechanisms, similar to Python list slicing but extended to two dimensions with both label and integer compatibility.

### Stage 12: Applying Functions to DataFrame Elements
**[CODE]**
```python
df.apply(np.max)
df[df["State"].apply(lambda state: state[0] == "W")].head()
```
**[PROCESS/CONTEXT]**
The `apply()` method is used to apply `np.max` across all columns to find the maximum value of each feature. A lambda function is then applied to the 'State' column to filter rows where the state abbreviation starts with 'W'.
**[RESULT]**
- `np.max` returns the maximums, e.g., max State is WY, max Total day minutes is 350.8.
- The lambda filter successfully retrieves rows for states like WV, WY, WI.
**[INSIGHT]**
The `apply` method is powerful for column-wise aggregations and row-wise element transformations or conditional evaluations that are not easily expressed with standard vectorized Pandas operations.

### Stage 13: Mapping and Replacing Values
**[CODE]**
```python
d = {"No": False, "Yes": True}
df["International plan"] = df["International plan"].map(d)
df = df.replace({"Voice mail plan": d})
```
**[PROCESS/CONTEXT]**
The `map()` and `replace()` methods are used to translate 'Yes'/'No' strings in the 'International plan' and 'Voice mail plan' columns into boolean `True`/`False` values.
**[RESULT]**
The columns are successfully transformed, replacing the string categories with boolean types, which is visible in the subsequent `head()` output.
**[INSIGHT]**
Standardizing categorical binary variables into boolean or integer formats is essential for preparing data for mathematical modeling and reducing memory footprint.

### Stage 14: Grouping Data
**[CODE]**
```python
columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]
df.groupby(["Churn"])[columns_to_show].describe(percentiles=[])
df.groupby(["Churn"])[columns_to_show].agg([np.mean, np.std, np.min, np.max])
```
**[PROCESS/CONTEXT]**
The data is grouped by the `Churn` variable, and statistics are calculated for specific columns (day, evening, and night minutes) using both the `describe()` method (with empty percentiles) and the `agg()` method with a custom list of numpy functions.
**[RESULT]**
The output shows side-by-side comparisons of churned (1) vs loyal (0) users. For instance, `Total day minutes` mean is 175.18 for Churn=0 and 206.91 for Churn=1.
**[INSIGHT]**
Churned customers consistently have higher average minute usage across day, evening, and night times, reinforcing the hypothesis that higher usage (and consequently higher billing) correlates with churn.

### Stage 15: Contingency and Pivot Tables
**[CODE]**
```python
pd.crosstab(df["Churn"], df["International plan"])
pd.crosstab(df["Churn"], df["Voice mail plan"], normalize=True)
df.pivot_table(["Total day calls", "Total eve calls", "Total night calls"], ["Area code"], aggfunc="mean")
```
**[PROCESS/CONTEXT]**
`crosstab` is used to create contingency tables examining the relationship between Churn and International plan, and Churn and Voice mail plan. `pivot_table` calculates the mean number of calls across different times of day grouped by 'Area code'.
**[RESULT]**
- The crosstab shows 137 users with an International plan churned, out of 323 total users with the plan.
- The pivot table reveals that the average calls are remarkably consistent across the three area codes (around 100 calls for day, evening, and night).
**[INSIGHT]**
The churn rate among users with an International Plan is disproportionately high (137 out of 323, roughly 42%) compared to the overall churn rate (14.5%). Area code does not appear to have a strong relationship with call volume.

### Stage 16: DataFrame Transformations (Adding/Removing Columns)
**[CODE]**
```python
total_calls = df["Total day calls"] + df["Total eve calls"] + df["Total night calls"] + df["Total intl calls"]
df.insert(loc=len(df.columns), column="Total calls", value=total_calls)
df["Total charge"] = df["Total day charge"] + df["Total eve charge"] + df["Total night charge"] + df["Total intl charge"]
df.drop(["Total charge", "Total calls"], axis=1, inplace=True)
```
**[PROCESS/CONTEXT]**
New columns 'Total calls' and 'Total charge' are created by summing existing columns. The `insert` method is used for the former, and direct assignment is used for the latter. Afterward, both newly created columns are removed using the `drop` method with `inplace=True`.
**[RESULT]**
The columns are temporarily added and then permanently removed from the DataFrame `df`.
**[INSIGHT]**
Feature engineering (creating aggregates like Total charge) is straightforward in Pandas through vectorized arithmetic operations.

### Stage 17: Visualizing Churn vs International Plan
**[CODE]**
```python
import matplotlib.pyplot as plt
import seaborn as sns
%config InlineBackend.figure_format = 'retina'
pd.crosstab(df["Churn"], df["International plan"], margins=True)
sns.countplot(x="International plan", hue="Churn", data=df);
```
**[PROCESS/CONTEXT]**
A final contingency table with margins is printed, and seaborn is imported to create a countplot visualizing the distribution of Churn across the International plan feature.
**[RESULT]**
The table confirms 137/323 users with the plan churned. The countplot visually emphasizes that the bar representing churned users is significantly taller relative to loyal users within the "True" International plan category compared to the "False" category.
**[INSIGHT]**
The visual strongly confirms the earlier statistical finding: possessing an International plan is heavily correlated with customer churn.

### Stage 18: Visualizing Churn vs Customer Service Calls
**[CODE]**
```python
pd.crosstab(df["Churn"], df["Customer service calls"], margins=True)
sns.countplot(x="Customer service calls", hue="Churn", data=df);
```
**[PROCESS/CONTEXT]**
The relationship between the number of customer service calls and churn is analyzed using a crosstab and visualized with a seaborn countplot.
**[RESULT]**
The crosstab shows churn spikes at 4 or more calls (e.g., at 4 calls, 76 out of 166 users churned). The countplot clearly displays the proportion of the orange (Churn=1) bar overtaking the blue (Churn=0) bar starting at 4 calls.
**[INSIGHT]**
There is a distinct threshold at 4 customer service calls where the probability of a customer churning becomes exceptionally high.

### Stage 19: Creating a Baseline Predictive Model
**[CODE]**
```python
df["Many_service_calls"] = (df["Customer service calls"] > 3).astype("int")
pd.crosstab(df["Many_service_calls"] & df["International plan"], df["Churn"])
```
**[PROCESS/CONTEXT]**
A new binary feature `Many_service_calls` is engineered based on the insight from the previous step. A crosstab is then generated to combine the two strongest predictive features: `Many_service_calls` AND `International plan` being True against the `Churn` variable.
**[RESULT]**
The final table shows:
- Predict 0 (Loyal): 2841 correctly predicted, 464 incorrectly predicted (False negative).
- Predict 1 (Churn): 19 correctly predicted, 9 incorrectly predicted (False positive).
Total errors = 464 + 9 = 473. Accuracy = (3333 - 473) / 3333 = 85.8%.
**[INSIGHT]**
A simple logical rule ("If customer has >3 service calls AND an International Plan, they will churn") yields an accuracy of 85.8%. This serves as a non-machine-learning baseline model.

## 4. Cross-Cell Dependency Analysis
- **Data Initialization Pipeline:** Cell 5 (`read_csv`) is a hard dependency for all subsequent cells.
- **Type Casting Dependencies:** Cells converting data types (like `Churn` to integer, and mapping 'Yes'/'No' to boolean) structurally modify the dataframe, affecting the behavior of downstream aggregation functions (like `describe` or `mean`).
- **Feature Engineering Dependencies:** The creation of `Many_service_calls` in cell 73 is strictly required to execute the final predictive crosstab in cell 76.
- **Visual Library Imports:** The `seaborn` and `matplotlib` imports in cell 67 dictate the success of the plotting functions in cells 68, 71, and 74.

## 5. Summary of Analytical Findings
Without deploying any formal machine learning models, the exploratory data analysis yielded a strong baseline heuristic. The initial dataset has a baseline loyalty rate of 85.5%. By strictly analyzing statistical distributions and visualizations, two primary drivers of customer churn were identified:
1.  **International Plan:** Customers enrolled in the international plan have a drastically higher churn rate.
2.  **Customer Service Interactions:** A distinct threshold exists at 4 or more customer service calls, beyond which churn likelihood increases sharply.
Combining these two insights into a simple heuristic ("If Many_service_calls and International_plan") provides an accuracy of 85.8%, narrowly beating the naive baseline and providing a target for future machine learning algorithms to surpass.

## 6. Conclusions and Recommendations
The notebook successfully demonstrates that comprehensive Exploratory Data Analysis (EDA) using Pandas is capable of uncovering profound business insights and establishing predictive baselines without complex modeling. 
Based strictly on the data, the telecom operator should:
- Investigate the pricing or service quality of the International Plan, as it is a major friction point leading to churn.
- Implement an intervention protocol when a customer reaches 3 customer service calls, as reaching a 4th call strongly predicts imminent churn.
- Use the established 85.8% accuracy as a minimum threshold when evaluating future Decision Tree or Random Forest models on this dataset.

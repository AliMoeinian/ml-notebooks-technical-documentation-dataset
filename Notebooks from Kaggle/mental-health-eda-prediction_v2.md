# Technical Documentation: Mental Health EDA and Prediction

## 1. Notebook Overview
This notebook presents a comprehensive exploratory data analysis (EDA) and predictive modeling workflow focusing on the global prevalence of various mental health disorders. The author utilizes four distinct but interconnected datasets, exploring the statistical properties, visual correlations, and predictive relationships among different mental illnesses. The analysis relies heavily on the Plotly library to produce highly interactive, multi-dimensional visualizations, specifically mapping the prevalence of major depression and bipolar disorder across 22 global regions. In the second phase, the notebook transitions into a machine learning task, where a Linear Regression model is implemented to predict the prevalence of "Eating disorders" based on the prevalence rates of other mental health conditions, namely Schizophrenia, Depressive disorders, Anxiety disorders, and Bipolar disorders. 

The predictive modeling pipeline involves normalization, basic linear regression, the automated addition of polynomial features (squared terms) through a custom function, and the manual creation of interaction terms ("Dot Models"). The notebook systematically evaluates the model's performance using standard regression metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) scores, achieving a maximum R2 score of approximately 67.8%. The author concludes with an assessment of the linear model's limitations and suggests alternative techniques such as clustering or Principal Component Analysis (PCA) for future work.

## 2. Environment and Dependencies
The execution environment relies on a standard data science stack in Python, incorporating libraries for data manipulation, visualization, and machine learning. The environment produces a warning related to the SciPy and NumPy versions ("A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5"), indicating a minor version mismatch that does not halt execution.

The specific libraries and modules imported in Cell 7 are:
- **numpy (as np)**: Used for numerical operations, specifically calculating the square root of the mean squared error (`np.sqrt`).
- **pandas (as pd)**: The primary tool for data loading, data frame construction, subsetting, renaming columns, and applying string/regex replacements.
- **matplotlib.pyplot (as plt)**: Used for static visualizations, including setting figure sizes, dpi, subplots, and creating the final actual vs. predicted scatter plots.
- **seaborn (as sns)**: Used for statistical data visualization, specifically rendering the correlation heatmap, boxplots, and pairwise scatterplots.
- **sklearn.linear_model.LinearRegression**: The sole machine learning algorithm instantiated and trained in the notebook.
- **sklearn.model_selection**:
  - `train_test_split`: Used to partition the data into training and testing sets.
  - `KFold`: Used to define the cross-validation splitting strategy.
  - `cross_val_score`: Used to evaluate the linear regression model across the KFold splits.
- **sklearn.preprocessing**: Provides the `MinMaxScaler` used to normalize the feature space.
- **sklearn.metrics**: Supplies the evaluation functions: `mean_absolute_error`, `mean_squared_error`, and `r2_score`.
- **plotly.graph_objects (as go)**: Used to construct complex, multi-trace visualizations, such as the dynamic subplot and the line charts.
- **plotly.express (as px)**: Used for rapid creation of horizontal bar charts.
- **plotly.subplots.make_subplots**: Employed to construct the dual-axis subplot layout.
- **plotly.offline.init_notebook_mode**: Initialized with `connected=True` to enable Plotly rendering within the Jupyter Notebook environment.
- **warnings**: Used to filter out and ignore standard warning messages to keep the output clean.

## 3. Per-Stage Documentation

### Introduction and Concept Definition
**[CODE]** Cells 1 through 5 are entirely markdown cells containing stylized HTML and CSS. They utilize a custom design system with properties like `color:white`, `border-radius:100px`, `background-color:#5642C5`, `font-family:Verdana`, and specific padding and letter-spacing values to present introductory text.
**[PROCESS/CONTEXT]** The author sets the stage by defining mental health as encompassing emotional, psychological, and social well-being, and explains its importance in relation to physical health (e.g., depression increasing the risk for diabetes, heart disease, and stroke). The intent of the notebook is explicitly stated: to perform comprehensive analysis using Plotly and apply a regression algorithm to model one variable as a target, while noting that clustering might be a better approach for future iterations.
**[RESULT]** The notebook begins with a visually distinct, purple-themed, rounded-corner introduction that outlines the conceptual framework and the technical roadmap.
**[INSIGHT]** The extensive use of raw HTML and CSS in markdown cells indicates a strong emphasis on presentation and user experience, aiming for a highly customized report format rather than a standard plain-text notebook.

### Data Loading and DataFrame Instantiation
**[CODE]** In Cell 9, the author uses `pd.read_csv` to load four datasets from a Kaggle input directory (`/kaggle/input/mental-health/`):
1. `1- mental-illnesses-prevalence.csv` (Assigned to `Data1`)
2. `4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv` (Assigned to `Data2`)
3. `6- depressive-symptoms-across-us-population.csv` (Assigned to `Data3`)
4. `7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv` (Assigned to `Data4`)
In Cell 10, these raw variables are explicitly wrapped in `pd.DataFrame()` calls and assigned to `df1`, `df2`, `df3`, and `df4`.
**[PROCESS/CONTEXT]** The data loading step brings the necessary CSV files into memory. The explicit `pd.DataFrame()` wrapping in Cell 10 is technically redundant since `pd.read_csv` already returns a DataFrame object, but it formally establishes the variable names used throughout the analysis.
**[RESULT]** Four distinct DataFrames are initialized and ready for structural inspection.
**[INSIGHT]** The datasets cover different aspects of mental health: overall prevalence (`df1`), specific adult population coverage (`df2`), depressive symptoms frequency (`df3`), and data availability by country (`df4`). 

### Custom Functional Description
**[CODE]** In Cell 12, the author defines a custom function `describe(df)`. This function initializes empty lists for `variables`, `dtypes`, `count`, `unique`, and `missing`. It iterates over `df.columns`, appending the column name, data type (`df[item].dtype`), length (`len(df[item])`), number of unique values (`len(df[item].unique())`), and sum of null values (`df[item].isna().sum()`). It returns a new DataFrame constructed from these lists. Cell 13 defines a `color` class with ANSI escape codes (`BLUE`, `BOLD`, `UNDERLINE`, `END`). In Cell 15, the `describe` function is called on all four DataFrames, wrapped in the custom ANSI color formatting.
**[PROCESS/CONTEXT]** Instead of using the built-in `pd.DataFrame.info()` or `pd.DataFrame.describe()`, the author engineers a custom diagnostic tool. This provides a clean, tabular view of the dataset structures, unifying type checking, unique value counting, and missing value detection into a single output.
**[RESULT]** The output reveals the exact shapes and metadata:
- `df1` (Mental illness): 6420 rows, 8 columns. 'Entity' has 214 unique values. The disease columns are float64. There are 270 missing values in the 'Code' column.
- `df2` (Adult population): 22 rows, 9 columns. Contains columns like 'Major depression', 'Bipolar disorder', 'Eating disorders', etc. 'Schizophrenia' is imported as an object (string) rather than a float. The 'Code' column has 21 missing values out of 22.
- `df3` (Depressive): 10 rows, 7 columns. Features columns for symptom frequency: 'Nearly every day', 'More than half the days', 'Several days', 'Not at all'. 'Code' is entirely missing (10 nulls).
- `df4` (Number of countries): 15 rows, 4 columns. Features 15 unique entities and 11 unique values for the target metric. 'Code' is entirely missing (15 nulls).
**[INSIGHT]** The custom function successfully identifies data quality issues early on. The discovery that `df2['Schizophrenia']` is an object type (string) rather than a numerical float immediately signals that data cleaning will be necessary before plotting or mathematical operations can be performed on that specific feature.

### Univariate Plotly Bar Charts and Data Cleaning
**[CODE]** In Cell 17, `df2` is sorted by "Major depression" and a horizontal bar chart is plotted using `px.bar` (x="Major depression", y="Entity", color='Bipolar disorder'). In Cell 18, the process is repeated, sorting by "Eating disorders" and plotting against "Entity", colored by "Dysthymia". In Cell 19, the author applies a regex replacement on `df2`: `df2.replace(to_replace="<0.1", value=0.1, regex=True, inplace=True)`. In Cell 20, `df2['Schizophrenia']` is successfully cast to float using `.astype(float)`. Finally, in Cell 21, the cleaned Schizophrenia column is sorted and plotted as a horizontal bar chart, colored by "Anxiety disorders".
**[PROCESS/CONTEXT]** The author conducts exploratory visual analysis on the adult population dataset (`df2`). The bar charts map disorder prevalence across the 22 entities. Before the Schizophrenia column can be plotted, the author must resolve the issue identified in the describe phase (it being an object). The regex replacement reveals that the string "<0.1" was present in the data, preventing numerical parsing. By replacing "<0.1" with the float 0.1, the column is repaired and converted.
**[RESULT]** Three interactive horizontal bar charts are generated. The data cleaning operation successfully transforms the string column into a usable numeric format. 
**[INSIGHT]** The data cleaning steps in Cells 19 and 20 are a direct response to the structural anomalies surfaced by the custom describe function. Replacing "<0.1" with exactly 0.1 introduces a slight upward bias for those specific data points (as the true value could be 0.05 or 0.01), but it allows the analysis to proceed without dropping rows.

### Complex Dynamic Plotly Subplots
**[CODE]** In Cell 23, the author constructs an "Amazing Dynamik Subplot" using `make_subplots`. The layout features 1 row and 2 columns, with a shared x-axis disabled. A hardcoded list `x1` containing 22 regional strings (e.g., "Andean Latin America", "Western Europe", "World") is defined. 
A horizontal bar trace (`go.Bar`) is appended to position (1,1), plotting `df2["Bipolar disorder"]` against `x1`, styled with a semi-transparent green color. 
A line/scatter trace (`go.Scatter`) is appended to position (1,2), plotting `df2["Major depression"]` against `x1`, styled with a dark purple line.
Extensive layout updates configure specific domains for the axes, gridline visibility, and background colors (`paper_bgcolor='rgb(248, 248, 255)'`). A `for` loop iterates over the depression values, bipolar values, and the `x1` regions, utilizing `zip()` to dynamically generate and append text annotations (`{:,}%` format for depression, `str()%` for bipolar) directly onto the plot area.
**[PROCESS/CONTEXT]** This is the most complex visualization in the notebook, designed to compare two different disorders across 22 global regions side-by-side. The manual definition of the layout domains and the precise loop-based annotation system demonstrate advanced Plotly capabilities.
**[RESULT]** The output is a highly customized, dual-axis interactive chart. The left side displays a bar chart of Bipolar prevalence, and the right side displays a scatter/line plot of Major Depression, with exact percentage values hovering next to every data point.
**[INSIGHT]** While the visual output is impressive, the methodology contains a significant hardcoding flaw. The 22 regions in the list `x1` are manually typed out in a specific order. If `df2` were re-sorted (which it actively was in Cells 17, 18, and 21), the values in `df2["Bipolar disorder"]` would no longer correspond to the manually defined string labels in `x1`. The data points mapped to "Western Europe" might actually belong to "Australasia" due to the hidden sorting state of the DataFrame. Extracting the labels directly from the DataFrame (e.g., `df2['Entity']`) would have prevented this fragility.

### Multiple Trace Line Charts
**[CODE]** In Cell 25, a hardcoded list `x` of 11 string values (e.g., "Appetite change", "Depressed mood", "Suicidal ideation") is defined. A `go.Figure()` is instantiated, and three scatter traces are added to plot columns from `df3`: "Nearly every day" (firebrick line), "More than half the days" (royalblue line), and "Several days" (black dashdot line). The chart is titled 'Depressive symptoms across us population'.
In Cell 26, another hardcoded list `x` of 15 illness names (e.g., "Alcohol use disorders", "Anorexia nervosa") is defined. A single scatter trace plots `df4["Number of countries with primary data on prevalence of mental disorders"]` against this list.
**[PROCESS/CONTEXT]** The author visualizes `df3` and `df4` using multi-trace line charts to show the relationship between specific symptoms/illnesses and their frequencies/country counts. 
**[RESULT]** Two interactive line charts are produced. 
**[INSIGHT]** Similar to the previous visualization, the author relies heavily on manual list typing (`x` lists) rather than dynamically extracting the categories from the 'Entity' columns of `df3` and `df4`. This approach is error-prone and disconnects the visual labels from the actual underlying data structure.

### Feature Selection and Renaming for Machine Learning
**[CODE]** In Cell 28, the column names of `df1` are extracted to a list. In Cell 29, the author utilizes `df1.rename()` with a dictionary to heavily truncate the verbose column headers. For example, `'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized'` is shortened to simply `'Schizophrenia disorders'`. This is repeated for Depressive, Anxiety, Bipolar, and Eating disorders. In Cell 31, a subset DataFrame named `df1_variables` is created, containing only these five newly renamed numerical columns.
**[PROCESS/CONTEXT]** The EDA phase concludes, and the notebook pivots toward modeling. The verbose column names from the Global Burden of Disease study are too long for practical programmatic access. The subsetting isolates the numerical features required for statistical analysis and regression.
**[RESULT]** The DataFrame `df1` is updated with clean column names, and a dedicated feature matrix `df1_variables` (6420 rows, 5 columns) is established. The printed output in Cell 32 shows values ranging typically from 0.09 to 4.99 across the different disorders.
**[INSIGHT]** This isolation step is crucial. By dropping the categorical 'Entity', 'Code', and 'Year' columns, the dataset is purely continuous and numerical, ready for correlation mapping and model fitting.

### Correlation and Bivariate Analysis
**[CODE]** In Cell 34, `df1_variables.corr()` calculates the Pearson correlation coefficient matrix. In Cell 35, this matrix is visualized using `sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)`. 
In Cell 37, a 2x2 subplot grid of seaborn scatterplots is created, plotting "Eating disorders" on the y-axis against each of the other four disorders on the x-axes.
In Cell 38, a `while` loop iterates through a list of the five numerical column names, generating a highly elongated boxplot (`figsize = [30,3]`) for each variable to visualize distributions and outliers.
**[PROCESS/CONTEXT]** Before running a linear regression, it is necessary to understand the linear relationships between the predictors and the target ("Eating disorders"). The heatmap quantifies these relationships, while the scatterplots visually depict the shape and variance. The boxplots serve to identify the presence and scale of outliers within each feature's distribution.
**[RESULT]** A heatmap, a grid of four scatterplots, and five horizontal boxplots are rendered. The exact correlation values are embedded in the heatmap.
**[INSIGHT]** The target variable ("Eating disorders") is explicitly established by its placement on the y-axis of all four bivariate scatterplots. The visualizations prepare the logical foundation for predicting Eating disorders based on the prevalence of the other four conditions.

### Feature Scaling (Normalization)
**[CODE]** In Cell 41, the features `X_model` are defined as `df1[['Schizophrenia disorders', 'Depressive disorders','Anxiety disorders','Bipolar disorders']]`, and the target `y_model` is defined as `df1["Eating disorders"]`. In Cell 42, `preprocessing.MinMaxScaler()` is instantiated as `scaler`, and `X_model_norm = scaler.fit_transform(X_model)` is executed. Cell 43 outputs the resulting numpy array.
**[PROCESS/CONTEXT]** Scaling transforms the features to a common boundary (between 0 and 1) so that variables with naturally larger scales do not disproportionately influence the regression coefficients or optimization algorithms.
**[RESULT]** A transformed numpy array `X_model_norm` is created, where all values are bounded between 0.0 and 1.0. For example, the first row transforms to `[0.12714204, 0.56728135, 0.42008448, 0.39345779]`.
**[INSIGHT]** The `MinMaxScaler` is applied to the entire `X_model` dataset simultaneously, *prior* to splitting the data into training and testing sets. This introduces data leakage. The minimum and maximum values of the test set influence the scaling logic applied to the training set, meaning the training process has indirect access to information from the test environment.

### Baseline Linear Regression Model
**[CODE]** In Cell 45, `train_test_split` is called on `X_model_norm` and `y_model` with `test_size=20` and `random_state=42`. In Cell 46, the shapes are printed: `X_train` is (6400, 4) and `X_test` is (20, 4). In Cell 47, `Model = LinearRegression()` is instantiated and fitted using `Model.fit(X_train, y_train)`. In Cell 48, predictions are generated via `y_pred = Model.predict(X_test)`. In Cell 49, evaluation metrics are printed using `sklearn.metrics`. In Cell 50, a 10-fold cross-validation is executed using `KFold(10)` and `cross_val_score(Model, X_model_norm, y_model.ravel(), cv=k_fold, n_jobs=1)`.
**[PROCESS/CONTEXT]** A standard linear hyperplane is fitted to the training data. The model is then evaluated on the hold-out test set to determine its predictive accuracy and error margins. K-Fold cross-validation provides a secondary, more robust check of the model's stability across different subsets of the data.
**[RESULT]** The baseline model produces the following metrics on the test set:
- Mean Absolute Error: 0.08003
- Mean Squared Error: 0.02178
- Root Mean Squared Error: 0.14760
- R2 Score: 0.62899
The 10-fold cross-validation yields an array of R2 scores ranging from a low of 0.3022 to a high of 0.8401.
**[INSIGHT]** The use of `test_size=20` (an absolute integer) rather than a float (like 0.20) means the test set contains exactly 20 rows out of the 6420 available. This is an extremely small, highly volatile test sample (0.3% of the data). The extreme variance in the K-Fold cross-validation scores (from 0.30 to 0.84) highlights the instability of evaluating on such tiny subsets, demonstrating that the model's performance is highly dependent on which specific data points happen to fall into the test split.

### Higher Dimension Feature Engineering
**[CODE]** In Cell 52, a custom function `check(Dimension, testsize)` is defined. It iterates through each column in `X_model` (the unscaled DataFrame, not the normalized array). It creates a new polynomial feature by raising the column to the power of `Dimension`. It inserts this new column into `X_model`, executes a `train_test_split(test_size=testsize, random_state=0)`, fits a new `LinearRegression`, and generates predictions on the test set. It then calculates `r2_new`. If `r2_new` is less than the baseline `r2` (0.6289), it drops the newly inserted column. If it is greater, it updates the baseline `r2` and keeps the column. The function is executed as `check(2, 0.2)`. 
**[PROCESS/CONTEXT]** This is an automated feature selection algorithm designed to test whether adding squared, non-linear representations of the features improves the model's predictive power. 
**[RESULT]** The function prints a new maximum R2 score: 0.65542. Cell 53 displays the updated `X_model` DataFrame, revealing that four new columns have been permanently added to the dataset: `Bipolar disorders2`, `Anxiety disorders2`, `Depressive disorders2`, and `Schizophrenia disorders2`. All polynomial additions successfully improved the evaluation metric.
**[INSIGHT]** This method exhibits severe methodological flaws. First, it abandons the normalized `X_model_norm` data, reverting to the unscaled raw data. Second, the decision to keep or drop a feature is made entirely based on its performance on the *test* set. The model hyperparameters (the feature space structure) are actively learning from the test data, completely invalidating the test set as an unseen, unbiased evaluation metric. 

### Dot Model (Interaction Terms)
**[CODE]** In Cell 55, the author manually creates interaction terms by multiplying distinct features together:
- `Bipolar_Anx = X_model["Bipolar disorders"] * X_model["Anxiety disorders"]`
- `Bipolar_Anx2 = X_model["Bipolar disorders2"] * X_model["Anxiety disorders2"]`
- `Dep_Schi = X_model["Depressive disorders"] * X_model["Schizophrenia disorders"]`
- `Dep_Schi2 = X_model["Depressive disorders2"] * X_model["Schizophrenia disorders2"]`
In Cell 56, these four new series are inserted into `X_model` using `X_model.insert()`. In Cell 57, a final `train_test_split(test_size=0.2, random_state=0)` is executed, a new Linear Regression is fitted, and predictions are made. The final R2 score is calculated and outputted.
**[PROCESS/CONTEXT]** Interaction terms allow a linear model to account for situations where the effect of one predictor variable on the target depends on the value of another predictor variable. By multiplying the prevalence of Bipolar and Anxiety together, the model can capture compound effects.
**[RESULT]** The dataset expands horizontally again, and the final linear regression model achieves its highest R2 score: 0.67791 (approximately 67.8%).
**[INSIGHT]** The manual multiplication of variables is an ad-hoc implementation of polynomial feature expansion. The R2 score successfully increases from 0.6289 to 0.6554 (via squares) to 0.6779 (via interaction terms).

### Final Visualizations of Model Output
**[CODE]** In Cells 59-67, the author isolates the x-axis features (Bipolar, Schizophrenia, Anxiety, and Depressive) from `X_test`, along with the true labels (`y_test`) and the predictions (`y_pred`). Custom font dictionaries (`font1`, `font2`, `font3`) are defined. Four sequential `plt.figure(figsize=(20,10))` plots are generated. Each plot scatters the `Real Values` as blue circles and the `Predicted Values` as large hexagonal markers (`marker="H", s=80`) in distinct colors (maroon, orange, indigo, green).
**[PROCESS/CONTEXT]** Visualizing the actual targets against the model's predictions along the axes of the primary features helps in qualitatively assessing the model's fit. By plotting predicted and actual values side-by-side, the user can visually detect patterns of variance, heteroscedasticity, or systemic bias.
**[RESULT]** Four massive scatter plots are rendered, comparing real and predicted values for Eating disorders across the domains of the other four illnesses. 
**[INSIGHT]** The visualizations confirm that the model (with an R2 of ~67.8%) captures the general trend but exhibits significant variance, particularly at the extreme ends of the distributions. The model's predictions (hexagons) roughly track the central mass of the real values but struggle to capture outliers.

### Conclusion and Recommendations
**[CODE]** Cell 69 contains another stylized HTML block summarizing the findings. The author states the maximum regression accuracy was almost 70%. 
**[PROCESS/CONTEXT]** The author reflects on the project's outcome. They correctly identify that the linear correlation required for a highly accurate regression model was likely insufficient in this dataset. They hypothesize that alternative techniques like clustering, Principal Component Analysis (PCA), or Self-Organizing Maps (SOM) might yield better analytical results.
**[RESULT]** A conclusive summary wraps up the notebook, explicitly noting the limitations of the chosen linear methodology.
**[INSIGHT]** The author demonstrates strong analytical self-awareness. Recognizing when a linear model is inadequate for a specific dataset is a key data science competency. The suggestions to use dimensionality reduction (PCA) and unsupervised learning (clustering) are highly appropriate next steps for a dataset containing complex, overlapping psychological conditions.

## 4. Cross-Cell Dependency Analysis
The notebook exhibits several critical, brittle dependencies across different execution stages:
- **Hardcoded Plotly Arrays (Cell 23 & Cell 25):** The annotations and traces in the Plotly visualizations rely on manually constructed lists of strings (e.g., `x1 = ["Andean Latin America", ...]`). Because the underlying DataFrames (`df2`) are sorted in previous cells (Cells 17, 18, 21), these manual lists are entirely dependent on the specific sorting order applied prior to execution. If a user runs the cells out of order, or if the underlying CSV data changes, the visual labels will incorrectly map to the wrong data points without throwing a Python error.
- **Dynamic Feature Injection (Cell 52 -> Cell 55):** The interaction terms in Cell 55 (e.g., `X_model["Bipolar disorders2"]`) explicitly rely on the automated `check` function in Cell 52 having accepted and injected those squared columns into the DataFrame. The `check` function decides based on a random `train_test_split`. If a different `random_state` caused the test-set R2 score to drop, the `check` function would drop the squared column, and Cell 55 would violently crash with a `KeyError` when attempting to reference a column that doesn't exist.
- **Normalization Abandonment (Cell 42 -> Cell 52):** The normalized data array `X_model_norm` is used exclusively for the baseline model in Cell 47. However, the advanced feature engineering in Cells 52-57 completely abandons the normalized array and reverts to using the raw, unscaled `X_model` pandas DataFrame. The subsequent "Higher Dimension" models are trained on completely different scaling paradigms than the baseline model.

## 5. Model Performance Summary
The target variable is the prevalence of "Eating disorders". The input features are the prevalence rates of Schizophrenia, Depressive disorders, Anxiety disorders, and Bipolar disorders.

**Baseline Linear Regression (Normalized Data):**
- Test Size: 20 data points (0.3% of data)
- Mean Absolute Error: 0.08003
- Mean Squared Error: 0.02178
- Root Mean Squared Error: 0.14760
- R-Squared (R2): 0.62899
- 10-Fold CV Range: 0.3022 to 0.8401

**Polynomial Expansion (Raw Data):**
- Test Size: 20% of data (approx. 1284 data points)
- Included squared terms for all 4 features.
- R-Squared (R2): 0.65542

**Polynomial + Interaction Terms (Raw Data):**
- Test Size: 20% of data
- Included explicit dot products (Bipolar*Anxiety, Depressive*Schizophrenia, and their squared counterparts).
- R-Squared (R2): 0.67791

The model's predictive capability improves from ~62.9% to ~67.8% variance explained as complexity increases. However, the extreme volatility observed in the cross-validation scores indicates that the baseline evaluation was heavily skewed by a minuscule test size. The final performance confirms that a simple linear combination of these four mental health disorders can only account for roughly two-thirds of the variance in eating disorder prevalence.

## 6. Conclusions and Recommendations
The notebook demonstrates excellent capabilities in interactive data visualization utilizing Plotly and a thorough understanding of the iterative nature of feature engineering in machine learning. However, the predictive modeling pipeline suffers from systemic methodological flaws regarding data leakage and test-set contamination.

**Recommendations for Improvement:**
1. **Prevent Data Leakage during Scaling:** The `MinMaxScaler` must be applied *after* the dataset is split. The scaler should be fitted exclusively on `X_train`, and then both `X_train` and `X_test` should be transformed independently.
2. **Prevent Test-Set Contamination:** The `check` function actively learns hyperparameters (which features to include) by evaluating against the test set. The dataset should be divided into Train, Validation, and Test sets. The `check` function should evaluate against the Validation set, leaving the Test set completely unseen until the final 67.8% model is ready for evaluation.
3. **Utilize Scikit-Learn Pipelines:** The manual calculation of polynomial squares and interaction terms via multiplication is error-prone and brittle. Implementing `sklearn.preprocessing.PolynomialFeatures(degree=2)` within a `sklearn.pipeline.Pipeline` will handle all dimensional expansions mathematically safely and prevent the data drift observed between normalized and raw states.
4. **Remove Hardcoded Plotly Labels:** Replace all manually typed lists of regions and diseases with dynamic extractions from the DataFrames (e.g., `x1 = df2['Entity'].tolist()`). This ensures the visualizations remain accurate regardless of sorting logic or upstream data changes.
5. **Standardize Test Sizes:** The shift from an absolute test size of 20 points (0.3%) in the baseline model to a relative 20% in the advanced models makes baseline comparison impossible. A consistent `test_size=0.2` should be maintained throughout the entire notebook.

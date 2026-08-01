# Technical Documentation: Car Price Prediction v2

## 1. Notebook Overview
**Problem statement:** The notebook describes a process of Reading and Understanding the Dataset, Data Preprocessing, Exploratory Data Analysis, Data Preparation, and Model Creation/Evaluation on a vehicle dataset from cardekho (`car data.csv`). Its implicit primary goal is to predict the selling price of used cars given a variety of physical and historical attributes.
**Overall analytical approach:** The notebook begins by importing essential data manipulation and plotting libraries (pandas, numpy, matplotlib.pyplot, and seaborn). It then loads the dataset, performs exploratory data analysis including checking dimensions, observing data types, reviewing descriptive statistics, and running missing value checks. Data preprocessing physically transforms the integer 'Year' column into an 'Age' column. Extensive Univariate and Bivariate Exploratory Data Analysis follows, intentionally checking for data outliers and mapping feature correlations to the target variable. Categorical features are converted using one-hot encoding (dummy variables). Finally, the workflow applies an 80/20 train-test split and evaluates five independent regression algorithms: Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting. The notebook heavily utilizes cross-validation and `RandomizedSearchCV` for hyperparameter tuning. A custom evaluation function tracks R-squared metrics and automatically plots residuals and prediction scatter plots.
**Notable characteristics:** The notebook relies heavily on inline plotting, exhaustive random search hyperparameter tuning grids for both tree models and penalized linear models, and a centralized custom evaluation function that unfortunately mutates global variables.

## 2. Environment and Dependencies
- **pandas (pd)**: Utilized heavily for reading the CSV dataset (`pd.read_csv`), displaying dataframes (`df_main.head()`), computing shapes, checking info and descriptive statistics, running missing value checks (`isna().sum()`), column renaming, dropping columns, creating dummy variables for categorical encoding (`pd.get_dummies`), and creating pivot tables for bivariate analysis (`df_main.pivot_table()`).
- **numpy (np)**: Used primarily to generate mathematical arrays for hyperparameter tuning, specifically `np.logspace(-3,3,num=14)` to generate the alpha parameter grids for Ridge and Lasso regression.
- **matplotlib.pyplot (plt)**: Used for managing plot figures, constructing subplots (`plt.subplot`), and actively displaying the plots (`plt.show()`). Specific global configuration includes `%matplotlib inline` to render charts in the notebook and `plt.style.use('seaborn')` for aesthetics.
- **seaborn (sns)**: Used extensively for data visualization including `sns.countplot` for categorical univariate analysis, `sns.boxplot` for numeric univariate analysis, `sns.heatmap` for generating the correlation matrix visualization, `sns.distplot` for generating probability density residual plots, and basic plotting infrastructure.
- **os**: Imported at the top of the notebook but not actively utilized anywhere in the subsequent code.
- **warnings**: Used to suppress verbose console warnings using `warnings.simplefilter(action='ignore')`.
- **scikit-learn (sklearn)**:
  - `sklearn.model_selection`: `train_test_split` (for dividing the data 80/20), `cross_val_score` (for computing 5-fold cross-validation means), and `RandomizedSearchCV` (for hyperparameter tuning across parameter grids).
  - `sklearn.linear_model`: `LinearRegression`, `Ridge`, and `Lasso` regression implementations.
  - `sklearn.ensemble`: `RandomForestRegressor` and `GradientBoostingRegressor` tree-based implementations.
  - `sklearn.metrics`: `r2_score` (the sole metric utilized for regression model evaluation).

## 3. Per-Stage Documentation

### Reading and Understanding the Dataset
**[CODE]** 
Cell 1: Imports the necessary libraries (`pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `os`, `warnings`), sets display options to show maximum rows/columns, ignores general warnings, and sets the plot style to 'seaborn'.
Cell 2: Reads the source dataset from the relative directory `../input/vehicle-dataset-from-cardekho/car data.csv` into a pandas DataFrame named `df_main`.
Cell 3: Executes `df_main.head()` to display the first 5 rows of the ingested data.
Cell 4: Executes `df_main.shape` to view dataset matrix dimensions.
Cell 5: Executes `df_main.info()` to view column data types and structural non-null counts.
Cell 6: Executes `df_main.describe()` to view mathematical summary statistics of the numeric columns.
Cell 7: Executes `df_main.isna().sum()` to sum missing values per column.
**[PROCESS/CONTEXT]** The notebook explicitly states the section title "Reading and Understanding the Dataset" without further markdown explanation, but the code operations are standard steps for initial data inspection to thoroughly understand the schema, scale, ranges, and data cleanliness prior to manipulation.
**[RESULT]** 
- `df_main.head()` output: Shows columns `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`. The first row is a 2014 ritz, Selling_Price 3.35, Present_Price 5.59, Kms_Driven 27000, Fuel_Type Petrol, Seller_Type Dealer, Transmission Manual, Owner 0.
- `df_main.shape` output: The DataFrame has exactly `(301, 9)` rows and columns.
- `df_main.info()` output: Shows 301 entries, 9 columns. Dtypes include `float64(2)`, `int64(3)`, `object(4)`. Memory usage is 21.3+ KB.
- `df_main.describe()` output: `Year` ranges from 2003 to 2018 (mean 2013.63). `Selling_Price` ranges from 0.10 to 35.00 (mean 4.66). `Present_Price` ranges from 0.32 to 92.60 (mean 7.63). `Kms_Driven` ranges from 500 to 500000 (mean 36947.21). `Owner` ranges from 0 to 3.
- `df_main.isna().sum()` output: All columns show exactly `0` missing values.
**[INSIGHT]** The initial exploration definitively proves the dataset is fully complete (0 missing values across the board), but the `Kms_Driven` maximum value is 500,000, which is extremely high compared to the 75th percentile (48,767). The `Present_Price` max is 92.60 compared to a 75th percentile of just 9.90. These extreme maximum boundary values clearly indicate the presence of significant right-skewed outliers in the numerical variables that will likely impact downstream linear modeling.

### Data Preprocessing
**[CODE]**
Cell 9: Creates a new continuous column `Age` by calculating `2020 - df_main['Year']`, then immediately drops the original categorical-like `Year` column in-place.
Cell 10: Renames specific columns using `df_main.rename(columns = {'Selling_Price':'Selling_Price(lacs)','Present_Price':'Present_Price(lacs)','Owner':'Past_Owners'},inplace = True)`.
**[PROCESS/CONTEXT]** The notebook title implies this stage prepares the data format for future processing. Specifically, converting a chronological static year into an elapsed relative time (`Age`) and clarifying column unit definitions. No markdown explanation is given to justify these specific steps.
**[RESULT]** The dataset now contains an `Age` integer column representing the number of years since vehicle manufacture (relative to 2020) and no longer contains the original `Year` feature. The columns `Selling_Price`, `Present_Price`, and `Owner` are specifically renamed to reflect units and business context (`(lacs)` and `Past_Owners`).
**[INSIGHT]** The code hardcodes the integer `2020` as the fixed baseline year for the relative `Age` calculation. This mathematically freezes the dataset pipeline in the year 2020. If a user attempts to run this notebook in 2025 and feeds it brand new cars from 2024, the static subtraction logic `2020 - 2024` will result in negative ages (`-4`), completely breaking the model's logic. Using `datetime.now().year` is a required fix for longevity.

### Exploratory Data Analysis (EDA) - Univariate Analysis
**[CODE]**
Cell 13: Prints the new column schema via `df_main.columns`.
Cell 14: Defines categorical columns `cat_cols = ['Fuel_Type','Seller_Type','Transmission','Past_Owners']`. Uses a `while` loop (from i=0 to 3) to generate 4 side-by-side seaborn `countplot`s. The loop intentionally creates `subplot(1,2,1)` and `subplot(1,2,2)` and increments the variable `i` twice per figure iteration.
Cell 15: Defines numeric columns `num_cols = ['Selling_Price(lacs)','Present_Price(lacs)','Kms_Driven','Age']`. Uses a similar `while` loop structure to generate 4 side-by-side seaborn `boxplot`s.
Cell 16: Filters and displays dataframe rows where `Present_Price(lacs)` is strictly greater than its 99th statistical percentile (`df_main['Present_Price(lacs)'].quantile(0.99)`).
Cell 17: Filters and displays dataframe rows where `Selling_Price(lacs)` is strictly greater than its 99th statistical percentile.
Cell 18: Filters and displays dataframe rows where `Kms_Driven` is strictly greater than its 99th statistical percentile.
**[PROCESS/CONTEXT]** The section title states "Univariate Analysis". The countplots evaluate the mass distribution of categories, the boxplots visually highlight numeric distributions and extreme values, and the quantile filtering directly exposes the specific data rows containing the most extreme 1% of values in the dataset.
**[RESULT]**
- Cell 13 output: `Index(['Car_Name', 'Selling_Price(lacs)', 'Present_Price(lacs)', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Past_Owners', 'Age'], dtype='object')`.
- Cell 14 output: Generates two horizontal figures (720x288), each containing two bar countplots showcasing the imbalance in fuel types and transmissions.
- Cell 15 output: Generates two horizontal figures (936x216), each containing two boxplots. The boxplots for `Kms_Driven`, `Selling_Price`, and `Present_Price` show significant single-point outliers far past the upper statistical whiskers.
- Cell 16 output: Reveals exactly 2 outlier rows for `Present_Price(lacs)`: a 2017 'fortuner' (36.23 lacs) and a 2010 'land cruiser' (92.60 lacs).
- Cell 17 output: Reveals exactly 3 outlier rows for `Selling_Price(lacs)`: two 'fortuner' models (23.5 and 33.0 lacs) and the 'land cruiser' (35.0 lacs).
- Cell 18 output: Reveals exactly 3 outlier rows for `Kms_Driven`: an 'innova' (197,176 kms), a 'Honda Karizma' (213,000 kms), and an 'Activa 3g' (500,000 kms).
**[INSIGHT]** The author writes very specific querying logic to successfully locate the top 1% of outliers across three major numerical features, explicitly revealing highly skewed and abnormal data points (like an Activa scooter reportedly driven 500,000 kilometers). However, the author inexplicably performs no actual statistical mitigation—there are no `.drop()` operations or mathematical capping functions applied to these corrupted rows. They are identified through code and then entirely ignored during the modeling phase, which will warp the linear regression weights.

### Exploratory Data Analysis (EDA) - Bivariate/Multi-Variate Analysis
**[CODE]**
Cell 20: Computes the full numerical correlation matrix `df_main.corr()` and visualizes it directly using `sns.heatmap(..., annot=True, cmap="RdBu")`.
Cell 21: Extracts and displays correlations specifically targeted at the dependent variable: `df_main.corr()['Selling_Price(lacs)']`.
Cell 22: Creates a pandas pivot table summarizing the mean `Selling_Price(lacs)` grouping by `Seller_Type` (rows) and `Fuel_Type` (columns).
Cell 23: Creates a pandas pivot table summarizing the mean `Selling_Price(lacs)` grouping by `Seller_Type` (rows) and `Transmission` (columns).
**[PROCESS/CONTEXT]** Bivariate analysis explores multidimensional relationships between variables. The correlation matrix checks for direct linear relationships among numerical features, and the pivot tables summarize average historical selling prices across intersecting categorical consumer segments.
**[RESULT]**
- Cell 20 output: Displays a 576x396 pixel color-coded heatmap figure with annotated correlation coefficients.
- Cell 21 output: `Selling_Price(lacs)` numerical correlations are: `Present_Price(lacs)` (0.878983), `Kms_Driven` (0.029187), `Past_Owners` (-0.088344), and `Age` (-0.236141).
- Cell 22 output: Dealer + Diesel = 10.18, Dealer + Petrol = 5.25. Individual + Diesel = 16.00, Individual + Petrol = 0.73 (CNG data is sparse/NaN for Individuals).
- Cell 23 output: Dealer + Automatic = 12.25, Dealer + Manual = 5.76. Individual + Automatic = 1.97, Individual + Manual = 0.74.
**[INSIGHT]** The correlation array explicitly proves that the `Present_Price(lacs)` independent variable holds a very strong positive linear relationship (~0.88) with the target variable `Selling_Price(lacs)`. Shockingly, the `Kms_Driven` feature has an extremely weak linear correlation (0.029), which fundamentally contradicts general automotive industry intuition (where higher mileage usually drastically reduces a vehicle's asking price); this exceptionally weak linear metric is highly likely to be heavily distorted by the unhandled 500,000-km outlier identified but ignored in the previous analysis step.

### Data Preparation
**[CODE]**
Cell 26: `df_main.drop(labels='Car_Name',axis= 1, inplace = True)` removes the high-cardinality vehicle name string feature from the dataset.
Cell 27: Executes `df_main.head()` to visually confirm the deletion.
Cell 28: Applies one-hot encoding across the entire dataframe: `df_main = pd.get_dummies(data = df_main,drop_first=True)`.
Cell 29: Executes `df_main.head()` to view the newly encoded horizontal dataframe.
**[PROCESS/CONTEXT]** "Creating Dummies for Categorical Features" converts human-readable text labels into numerical boolean columns required by machine learning estimators. `drop_first=True` drops the first alphabetical level of the categorical variables to deliberately prevent perfect multicollinearity (the dummy variable trap). `Car_Name` is dropped entirely before the encoding process begins.
**[RESULT]**
- Cell 27 output: The dataframe dimensions shrink down to 8 active columns; the `Car_Name` column is completely gone.
- Cell 29 output: The text-based categorical columns `Fuel_Type`, `Seller_Type`, and `Transmission` are removed and instantly replaced by newly generated binary dummy columns: `Fuel_Type_Diesel`, `Fuel_Type_Petrol`, `Seller_Type_Individual`, and `Transmission_Manual`.
**[INSIGHT]** Dropping the `Car_Name` column outright discards the exact brand and specific model identity of the car, which is undoubtedly a massive determinant of any vehicle's resale value in the real world. For instance, a Mercedes and a Maruti Suzuki might have identical feature dimensions (age, fuel type, transmission) in this truncated dataset but vastly different real-world depreciation curves. This total loss of brand signal guarantees the model is predicting strictly based on generic mechanical and age states rather than accumulated brand equity.

### Train-Test Split
**[CODE]**
Cell 31: Splits the horizontal features from the vertical target: `y = df_main['Selling_Price(lacs)']` and `X = df_main.drop('Selling_Price(lacs)',axis=1)`.
Cell 33: Uses sklearn's `train_test_split(X, y, test_size=0.2, random_state=1)` to create a hard 80/20 train/test partition. It explicitly prints the matrix shapes of all resulting arrays.
**[PROCESS/CONTEXT]** Isolates the dependent target variable vector and randomly partitions the dataset arrays to evaluate final model performance on entirely unseen data.
**[RESULT]**
- Cell 33 output: `x train: (240, 8)`, `x test: (61, 8)`, `y train: (240,)`, `y test: (61,)`.
**[INSIGHT]** Because the starting dataset is incredibly small (only 301 total rows), the randomly allocated holdout test set is reduced to an anemic 61 data samples. Evaluating a machine learning model's final performance on just 61 data points makes the reported R-squared metric highly volatile and overly sensitive to which exact samples were randomly partitioned into the test array.

### Model Creation/Evaluation
**[CODE]**
Cell 37: Defines empty global Python lists: `CV`, `R2_train`, `R2_test`. Defines a large custom evaluation function `car_pred_model(model, model_name)`. The function fits the model on `X_train, y_train`, computes R2 scores for both the train and test sets using `r2_score`, runs a 5-fold `cross_val_score` on the training set block, appends these three metrics directly to the global lists, prints the numeric results to the console, and displays two matplotlib figures: a residual distribution plot (`sns.distplot`) of the error `(y_train - y_pred_train)` and a standard scatter plot of actual `y_test` vs predicted `y_pred_test`.
Cell 39: Standard Linear Regression (`LinearRegression()`) passed to the custom evaluation function.
Cell 41: Ridge Regression. Creates a `RandomizedSearchCV` object searching over a generated alpha parameter list `np.logspace(-3,3,num=14)` and evaluates the optimal estimator.
Cell 43 & 44: Lasso Regression. Uses the identical `RandomizedSearchCV` alpha space and evaluates the optimal estimator.
Cell 46 & 47 & 48: Random Forest. Creates a large hyperparameter tuning dictionary spanning `n_estimators` (500 to 900), `max_depth` (4, 8), `min_samples_split` (4, 6, 8), `min_samples_leaf` (1, 2, 5, 7), and `max_features` ('auto', 'sqrt'). Wraps this configuration inside `RandomizedSearchCV` and evaluates. Cell 48 explicitly prints the final `best_estimator_`.
Cell 50 & 51: Gradient Boosting. Uses a highly similar dictionary adding a `learning_rate` parameter (`[0.001, 0.01, 0.1, 0.2]`), wraps the configuration in `RandomizedSearchCV`, and evaluates.
Cell 52: Creates a Pandas DataFrame grouping the hardcoded string `Technique` names with the global lists `R2_train`, `R2_test`, and `CV` to summarize all model runs in a single view.
**[PROCESS/CONTEXT]** This block rigorously compares five different algorithmic regression estimators. It attempts to ensure fair comparison by mathematically optimizing the hyperparameters of the penalized and tree-based models via randomized search cross-validation before evaluating their final R-squared prediction performance.
**[RESULT]**
- Linear Regression outputs: Train R2: 0.88, Test R2: 0.86, Train CV mean: 0.84.
- Ridge Regression outputs: Train R2: 0.88, Test R2: 0.86, Train CV mean: 0.84.
- Lasso Regression outputs: Train R2: 0.88, Test R2: 0.86, Train CV mean: 0.84.
- Random Forest outputs: Train R2: 0.95, Test R2: 0.82, Train CV mean: 0.87. `best_estimator_` shows `max_depth=8`, `min_samples_leaf=2`, `min_samples_split=8`, `n_estimators=900`.
- Gradient Boosting outputs: Train R2: 1.0, Test R2: 0.94, Train CV mean: 0.87.
- Final Summary DataFrame (Cell 52): Displays a clean 5x4 table of all models and their associated `R Squared(Train)`, `R Squared(Test)`, and `CV score mean(Train)` metrics correctly rounded to two decimal places.
**[INSIGHT]** The Ridge and Lasso regularized models return completely identical Train (0.88) and Test (0.86) scores when compared directly to the basic unpenalized Linear Regression model. This statistically indicates the L1 and L2 regularization penalties had virtually no impact on the loss function, heavily likely because the final feature space (only 8 total columns) is far too small for regularization to matter. Additionally, the Gradient Boosting model achieves a Train R2 of 1.0 (a flawless mathematical fit), signaling massive overfitting to the training data, yet paradoxically posts the highest unseen Test R2 (0.94) across the board.

## 4. Cross-Cell Dependency Analysis
- **Dangerous Global List Append Mutation:** The global variables `CV`, `R2_train`, and `R2_test` defined in Cell 37 are explicitly mutated (appended to) every single time the `car_pred_model()` function is executed anywhere in the notebook. This creates a severe and fragile execution-order dependency. If the user arbitrarily re-runs the Random Forest cell, for instance, a second Random Forest output entry will be immediately appended to all three lists. By the time Cell 52 executes to create the summary dataframe, the data lists will possess 6 numerical elements, while the hardcoded string `Technique` list possesses only 5. This mismatch will trigger a fatal `ValueError: All arrays must be of the same length`, crashing the notebook entirely.
- **Categorical and Numeric Variable Name Matching:** The array lists `cat_cols` (Cell 14) and `num_cols` (Cell 15) deliberately hardcode specific column names exactly as they are typed, like `'Selling_Price(lacs)'`. If Cell 10 (the rename operation) is skipped by the user, or run out of order, the `while` loops attempting to generate the EDA plotting figures will instantly fail with a Python `KeyError`.

## 5. Model Performance Summary
- **Linear Models (OLS, Ridge, Lasso):** All three linear mathematical variations yielded completely indistinguishable evaluation metrics: a Training R2 of 0.88, a Test R2 of 0.86, and a Cross-Validation Training mean of precisely 0.84.
- **Tree-Based Models (Random Forest, Gradient Boosting):** The optimized Random Forest estimator posted an exceptionally high Training R2 of 0.95 but generalized extremely poorly to unseen data, dropping significantly to a Test R2 of just 0.82. The Gradient Boosting model successfully achieved a perfect Train R2 of 1.0, and the highest Test R2 of the entire group at 0.94, even though its 5-fold CV mean stalled at only 0.87.
- **Direct Comparisons:** Based strictly and exclusively on the isolated holdout test set of 61 rows, the Gradient Boosting algorithm vastly outperformed all other approaches, while the Random Forest paradoxically performed the worst in terms of pure generalization (test score). The linear models proved highly consistent and robust across all three calculation metrics without tuning.

## 6. Conclusions and Recommendations
- **Untreated Outliers Directly Impact Linear Models:** The notebook successfully wrote logic to identify the extreme top 1% outliers in both vehicle price and vehicle mileage using quantile calculations (Cells 16-18) but completely abandoned mitigating them. The visual residual plots produced by `car_pred_model()` for the linear regression models visually confirm heavy tails on the left side of the prediction error distribution. Capping (Winsorizing) or log-transforming highly skewed variables like `Kms_Driven` and `Present_Price(lacs)` would almost certainly improve linear model accuracy and correct the heavily warped correlations.
- **Hardcoded Temporal Features:** Calculating elapsed vehicle time via `Age = 2020 - Year` permanently anchors the model's fundamental math to the calendar year 2020. This fragile code should be rewritten using dynamic system time variables (such as Python's `datetime.now().year`) to ensure the pipeline survives execution in future years.
- **Dangerous Global State Mutations:** The central evaluation function `car_pred_model` is functionally fragile. Modifying global lists inside a local function scope causes notebook state corruption if any single evaluation cell is re-executed by a user. The function should instead cleanly `return` the calculated scalar values, which can then be aggregated safely in an independent dictionary, array, or dataframe at the conclusion of the notebook.
- **Severe Overfitting Indicators:** The Gradient Boosting model memorized the training data entirely (scoring an R2 of exactly 1.0) and although its Test R2 was surprisingly high, its 5-fold CV mean was only 0.87. This massive mathematical discrepancy between a 1.0 Train score and a 0.87 CV score implies the complex hyperparameters found by `RandomizedSearchCV` forced the model to overfit heavily on the specific `X_train` data block rather than discovering true generalizable patterns.
- **Missing Absolute Error Metrics:** While the R-squared value is a useful statistical percentage of variance metric, accurately predicting physical vehicle prices requires a deep understanding of absolute monetary error. The complete absence of standard regression metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) makes it entirely impossible to know if the model is typically wrong by 0.5 lacs or 5.0 lacs. Incorporating MAE into the custom evaluation function would provide urgently necessary real-world business context to the generated predictions.

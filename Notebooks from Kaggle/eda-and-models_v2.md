# Technical Documentation: EDA and Models (IEEE Fraud Detection)

## 1. Notebook Overview

### Problem Statement
The notebook aims to tackle the IEEE Fraud Detection competition hosted on Kaggle. According to the notebook's introductory markdown, the competition involves the IEEE Computational Intelligence Society (IEEE-CIS) partnering with Vesta Corporation, the world’s leading payment service company, to seek the best solutions for the fraud prevention industry. The core challenge presented is a binary classification problem. The author explicitly notes that the dataset exhibits a "heavy imbalance which is an inherent property of such problems," meaning that the vast majority of transactions are non-fraudulent, while only a small fraction represent actual fraud.

### Overall Analytical Approach
The author's stated analytical plan is a linear data science workflow:
1.  **Exploration:** The author states, "At first I'll explore the data and try to find valuable insights." This is carried out by examining distributions, missing values, and time-based characteristics of the data.
2.  **Feature Engineering:** The author mentions, "maybe I'll do some feature engineering." This is executed by creating aggregation-based features on the data.
3.  **Modeling:** The final step is to "build models." The notebook concludes with the training of a LightGBM model.

The observable workflow is highly structured, beginning with library imports and the definition of a large set of helper functions. These helper functions handle tasks ranging from rendering Altair visualizations in a Jupyter environment, optimizing memory usage, calculating metrics like ROC-AUC using Numba for speed, to executing large-scale cross-validation loops for various model types (LightGBM, XGBoost, CatBoost, Scikit-Learn). After setup, the data is loaded, explored visually, engineered to create relative numerical features, cleaned of highly sparse columns, label encoded, and finally fed into a 5-fold cross-validation LightGBM training process.

### Notable Characteristics
Several distinct characteristics define the methodology and structure of this notebook:
-   **Extensive Boilerplate Functions:** The notebook contains massive, multi-purpose functions for training regression (`train_model_regression`) and classification (`train_model_classification`) models. These functions are capable of handling multiple algorithms and evaluation metrics, abstracting the training loop away from the main execution cells.
-   **Alternative Visualization Library:** Rather than relying solely on the ubiquitous Matplotlib or Seaborn libraries, the author heavily utilizes Altair for Exploratory Data Analysis. This requires custom JavaScript wrapper functions (`prepare_altair` and `render`) to ensure the charts display correctly in the notebook environment.
-   **Discrepancy Between Insight and Implementation:** A notable characteristic of the notebook is the disconnect between a critical insight generated during the EDA phase (the need for a time-based validation split) and the actual code executed during the modeling phase (where standard KFold is ultimately used).
-   **Unused Code Segments:** The notebook contains defined functions (like `reduce_mem_usage` and `train_model_regression`) and commented-out code blocks (for XGBoost and model blending) that are never actually executed in the final run, indicating a "work in progress" state.

## 2. Environment and Dependencies

The notebook relies on a wide array of Python libraries, installed via `pip` or imported directly from the environment. Below is a detailed breakdown of all libraries imported and their specific, observable roles within this particular notebook.

### Installed Packages
In Cell 2, the author uses `!pip install -U vega_datasets notebook vega` to install or upgrade specific packages.
-   **`vega_datasets`**, **`notebook`**, **`vega`**: These packages are installed to support the Altair visualization library. The output stream shows the collection and installation of these packages, alongside the uninstallation of an older `notebook-5.5.0` version and the successful installation of `notebook-6.0.1`, `vega-2.6.0`, and `vega_datasets-0.7.0`. Note that an error is thrown regarding `cufflinks 0.16` having an incompatible `plotly` requirement, though this does not stop the execution.

### Imported Libraries
-   **`numpy` (as `np`)**: Heavily used for numerical operations. It is used in the custom `fast_auc` function (`np.asarray`, `np.argsort`), memory reduction function (`np.iinfo`, `np.finfo`, `np.int8`, `np.float32`, etc.), for defining bounds in `group_mean_log_mae` (`np.log`), initializing arrays for Out-Of-Fold predictions (`np.zeros`), and replacing infinite values (`np.inf`, `-np.inf`, `np.nan`).
-   **`pandas` (as `pd`)**: The primary data manipulation library. Used for loading CSVs (`pd.read_csv`), merging dataframes (`pd.merge`), concatenating DataFrames (`pd.concat`), performing groupby aggregations (`df.groupby().transform()`), mapping values, replacing values, managing columns, dropping columns (`df.drop()`), and writing the final submission file (`pd.to_csv`). It is also used in custom metric calculations (`pd.Series`).
-   **`os`**: Imported but visibly unused in the execution paths, although it's included in the boilerplate cell.
-   **`matplotlib.pyplot` (as `plt`)**: Used for basic visualizations. It plots histograms of `id_01` (Cell 19), `id_07` (Cell 25), and the `TransactionDT` for train and test sets (Cell 31). It is also used to initialize figures and set titles for the LightGBM feature importance bar plots.
-   **`seaborn` (as `sns`)**: Used exclusively to plot the feature importance bar chart (`sns.barplot`) generated after the LightGBM cross-validation loop completes.
-   **`tqdm.tqdm_notebook`**: Imported but not actively used in any of the executed loops.
-   **`sklearn.preprocessing.StandardScaler`**: Imported but not used.
-   **`sklearn.svm.NuSVR`, `SVR`**: Imported but not used.
-   **`sklearn.metrics.mean_absolute_error`**: Imported but not used directly in the active classification pipeline.
-   **`lightgbm` (as `lgb`)**: The core modeling library utilized. Specifically, `lgb.LGBMRegressor` is referenced in the unused regression function, and `lgb.LGBMClassifier` is executed to generate the final model and predictions.
-   **`xgboost` (as `xgb`)**: Imported and fully integrated into the custom training functions (`xgb.DMatrix`, `xgb.train`), and its execution is present in commented-out cells at the very end of the notebook intended for blending. It is not executed in the active run.
-   **`catboost.CatBoostRegressor`, `CatBoostClassifier`**: Imported and integrated into the custom training functions, but never invoked during the execution of the notebook.
-   **`time`**, **`datetime`**: Used to print the start times of the cross-validation folds (e.g., `time.ctime()`).
-   **`sklearn.preprocessing.LabelEncoder`**: Used to encode all categorical string variables into integers, which is a requirement for LightGBM's internal handling of categories.
-   **`sklearn.model_selection`**:
    -   `StratifiedKFold`, `RepeatedKFold`, `GroupKFold`, `GridSearchCV`, `train_test_split`: Imported but not used.
    -   `TimeSeriesSplit`: Instantiated but immediately overwritten.
    -   `KFold`: Used to define the 5-fold cross-validation splitting strategy.
-   **`sklearn.metrics`**: Used to map evaluation metrics in the custom functions (`metrics.mean_absolute_error`, `metrics.mean_squared_error`, `metrics.roc_auc_score`).
-   **`sklearn.linear_model`**: Imported but not used.
-   **`gc`**: Garbage collection, used explicitly via `gc.collect()` before modeling to free up RAM.
-   **`warnings`**: Used to suppress warnings via `warnings.filterwarnings("ignore")`.
-   **`eli5`**, **`shap`**: Imported for model explainability but never used.
-   **`IPython.display.HTML`**: Used in the custom Altair rendering function to output raw HTML to the Jupyter interface.
-   **`json`**: Used in the custom Altair rendering function (`json.dumps`).
-   **`altair` (as `alt`)**: The primary visualization library used for EDA. Used to generate bar charts (`alt.Chart().mark_bar().encode()`).
-   **`altair.vega.v5`**: Used in the Altair setup workaround.
-   **`networkx` (as `nx`)**: Imported but not used.
-   **`numba.jit`**: A Just-In-Time compiler decorator used to drastically speed up the custom `fast_auc` calculation function by compiling Python code to machine code.
-   **`itertools.product`**: Imported in the hidden cell but not used.

## 3. Per-Stage Documentation

### Stage 1: Setup and Boilerplate Functions
**[CODE]**
The author configures Altair and defines six functions: `prepare_altair`, `add_autoincrement`, `render`, `reduce_mem_usage`, `fast_auc`, `eval_auc`, `group_mean_log_mae`, `train_model_regression`, and `train_model_classification`.
**[PROCESS/CONTEXT]**
- `prepare_altair`, `add_autoincrement`, and `render` are explicitly created as workaround functions to allow Altair visualizations to render correctly inside the specific Kaggle/Jupyter environment.
- `reduce_mem_usage` iterates over DataFrame columns and downcasts numerical types to their lowest possible memory footprint (e.g., float64 to float32 or int8) to prevent Out-Of-Memory errors.
- `fast_auc` uses Numba to rapidly calculate the ROC-AUC score, referencing a specific Kaggle discussion post for the optimized logic.
- `train_model_regression` and `train_model_classification` are massive wrappers intended to abstract cross-validation splitting, model initialization (LGB, XGB, CatBoost, Sklearn), model training, prediction generation (OOF and test), and metric evaluation.
**[RESULT]**
The functions are compiled into memory. The environment variables are set (e.g., `JOBLIB_TEMP_FOLDER=/tmp`). The Altair workaround outputs an `<IPython.core.display.HTML object>`.
**[INSIGHT]**
This stage sets up a highly robust, generic pipeline. However, much of the defined code (e.g., `train_model_regression`, `group_mean_log_mae`, `reduce_mem_usage`, and all XGBoost/CatBoost logic) is never actually used in the notebook, resulting in significant "dead code" bloat.

### Stage 2: Data Loading and Overview
**[CODE]**
The author defines the data folder path and reads `train_identity.csv`, `train_transaction.csv`, `test_identity.csv`, `test_transaction.csv`, and `sample_submission.csv`. Using `pd.merge` with a left join on `TransactionID`, the transaction and identity tables are combined into `train` and `test` dataframes. The shape of the combined data is printed. The individual dataframes are then deleted (`del`). Missing values and columns with single unique values are counted.
**[PROCESS/CONTEXT]**
The author notes: "Data is separated into two datasets... let's combine the data and work with the whole dataset". The identity data is merged onto the transaction data because, as the author states, "Not all transactions belong to identities". The overview is to check the shape and missingness of the resulting tables.
**[RESULT]**
The `pd.merge` creates a massive dataset. The print statement outputs:
`Train dataset has 590540 rows and 434 columns.`
`Test dataset has 506691 rows and 433 columns.`
The printed output for missing values reveals:
`There are 414 columns in train dataset with missing values.`
The printed output for unique values reveals:
`There are 0 columns in train dataset with one unique value.`
`There are 1 columns in test dataset with one unique value.`
**[INSIGHT]**
The dataset is extremely wide (434 columns) and heavily sparse, with 414 out of 434 columns containing missing data. The memory reduction function `reduce_mem_usage` defined earlier is notably *not* called here. Merging such massive tables without downcasting wastes significant RAM.

### Stage 3: Exploratory Data Analysis (EDA)
**[CODE]**
The author uses `plt.hist` to plot the distributions of continuous variables `id_01` and `id_07`. The `value_counts()` function is used to inspect the exact distributions of `id_03` and `id_11`. The custom Altair `render()` wrapper is then used in a loop to plot bar charts showing the counts of numerous categorical variables, such as "found" status identifiers (`id_12` to `id_38`), `DeviceType`, `DeviceInfo`, `ProductCD`, card variables (`card1` to `card6`), matching variables (`M1` to `M9`), and email domains. Finally, `plt.hist` is used to plot `TransactionDT` (Transaction Date-Time) for both the train and test sets on the same axis.
**[PROCESS/CONTEXT]**
The purpose of this section is visual inspection. The author explicitly notes insights derived from the plots:
- For `id_01`: "`id_01` has an interesting distribution: it has 77 unique non-positive values with skeweness to 0."
- For `id_03`: "`id_03` has 88% of missing values and 98% of values are either missing or equal to 0."
- For `id_11`: "22% of values in `id_11` are equal to 100 and 76% are missing. Quite strange."
- For `id_07`: "Some of features seem to be normalized."
- For the `TransactionDT` plot, the author states: "A very important idea: it seems that train and test transaction dates don't overlap, so it would be prudent to use time-based split for validation."
**[RESULT]**
Various matplotlib histograms and interactive Altair HTML bar charts are output to the notebook. The value counts for `id_03` show `NaN` at 88.76% and `0.0` at 10.82%. The value counts for `id_11` show `NaN` at 76.12% and `100.0` at 22.54%.
**[INSIGHT]**
The author correctly identifies a critical characteristic of the dataset through the `TransactionDT` histogram: the test set is chronologically strictly after the train set. This means random cross-validation splitting would result in future data predicting past data, which is mathematically unsound for time-series forecasting. The author explicitly recognizes the need for a "time-based split."

### Stage 4: Feature Engineering
**[CODE]**
The author executes a series of groupby aggregations. Specifically, the continuous variables `TransactionAmt`, `id_02`, and `D15` are grouped by categorical identifiers `card1`, `card4`, `addr1`, and `addr2`. For each combination, the `.transform('mean')` and `.transform('std')` are calculated. The original continuous value is then divided by these grouped statistical values, creating features like `TransactionAmt_to_mean_card1` and `TransactionAmt_to_std_card1`. These operations are performed identically on both the train and test sets. Additionally, `P_emaildomain` and `R_emaildomain` are split into three new columns using `.str.split('.', expand=True)`.
**[PROCESS/CONTEXT]**
The author describes this simply as: "Let's create some aggregations. There is no logic in them - simply aggregations on top features." Splitting the email domains separates the core domain (e.g., 'gmail') from its Top Level Domain (e.g., 'com'). 
**[RESULT]**
Dozens of new relative numerical columns are added to the train and test dataframes. The email domains are split into components (e.g., `P_emaildomain_1`, `P_emaildomain_2`, `P_emaildomain_3`).
**[INSIGHT]**
While the author claims there is "no logic," this is a powerful relative magnitude scaling technique. It allows the model to assess how unusual a transaction amount is *relative to a specific card's average*.
However, using `.transform('std')` introduces a critical mathematical flaw. If a particular group (e.g., a specific `card1` ID) only appears exactly one time in the dataset, its standard deviation is mathematically undefined (`NaN`). If a group appears multiple times but always has the exact same `TransactionAmt`, the standard deviation is `0`. Dividing the `TransactionAmt` by a standard deviation of `0` results in `inf` (infinity). Thus, this step silently introduces massive amounts of `NaN` and `inf` values into the dataset.

### Stage 5: Data Preparation
**[CODE]**
The author creates lists of columns to drop based on structural rules:
1. `many_null_cols`: Columns with over 90% missing values (`isnull().sum() / df.shape[0] > 0.9`).
2. `big_top_value_cols`: Columns where the most frequent value accounts for over 90% of the distribution.
3. `one_value_cols`: Columns with exactly 1 or 0 unique values.
These lists are combined, the `isFraud` target is removed from the drop list, and `df.drop()` is executed.
Next, a massive list of `cat_cols` (categorical columns) is defined. A `LabelEncoder()` loop iterates over them. Crucially, the `le.fit()` is performed on the concatenated values of both the train and test sets for that column. The datasets are then separated into `X` (features), `y` (target), and `X_test`. `TransactionDT` and `TransactionID` are dropped. 
Finally, a function `clean_inf_nan` is defined to replace `np.inf` and `-np.inf` with `np.nan`, and is applied to `X` and `X_test`. `gc.collect()` is run to clear memory.
**[PROCESS/CONTEXT]**
The author labels this section "Prepare data for modelling." Dropping high-sparsity and zero-variance columns eliminates useless noise. Label Encoding is strictly necessary for tree-based models like LightGBM to process categorical string data; it maps strings to integers. Cleaning the infinite values is a direct response to the mathematical errors introduced during the feature engineering stage.
**[RESULT]**
The code outputs `84` as the length of `cols_to_drop`. The training feature set `X` is completely purged of infinite values and strings, resulting in a dense, entirely numeric matrix ready for ingestion by LightGBM. The garbage collector outputs `47`.
**[INSIGHT]**
Fitting the `LabelEncoder` on the combined list of train and test values is an excellent practice. It ensures that if a categorical string exists in the test set but not the train set, it is still assigned a valid integer mapping, preventing runtime errors during prediction. Furthermore, the `clean_inf_nan` function confirms the insight from Stage 4 that the feature engineering step indeed generated infinite values.

### Stage 6: Model Creation and Evaluation
**[CODE]**
Under the heading "LGBM", the author defines the cross-validation strategy:
```python
n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)
folds = KFold(n_splits=5)
```
A dictionary of LightGBM hyperparameters (`params`) is defined, including `num_leaves: 256`, `max_depth: 13`, `learning_rate: 0.03`, and `metric: auc`. The massive custom `train_model_classification` function is then invoked, passing the data, params, folds, and requesting `lgb` as the model type with feature importance plotting enabled. Finally, the generated predictions are written to `submission.csv` and `lgb_oof.csv`.
**[PROCESS/CONTEXT]**
The goal is to train the LightGBM model using 5-fold cross validation. The custom function abstracts the iteration over folds, training the model on 4 folds, validating on the 5th, generating predictions on the test set for each fold, and averaging the test predictions.
**[RESULT]**
The custom training loop prints the progress of the 5 folds. The metric being optimized and printed is the ROC-AUC score.
- Fold 1: best iteration is 513, validation AUC is `0.919435`
- Fold 2: best iteration is 546, validation AUC is `0.931982`
- Fold 3: best iteration is 372, validation AUC is `0.926607`
- Fold 4: best iteration is 464, validation AUC is `0.946655`
- Fold 5: best iteration is 405, validation AUC is `0.923309`
The final printed output is: `CV mean score: 0.9296, std: 0.0095.`
A large Seaborn feature importance bar plot is generated in the output. The predictions are saved to CSV. The output of `sub.head()` shows prediction probabilities like `0.001571974543870`.
**[INSIGHT]**
The overwrite of `TimeSeriesSplit` with `KFold` is a catastrophic error. Despite the author correctly identifying in the EDA stage that "train and test transaction dates don't overlap, so it would be prudent to use time-based split", they accidentally override the `TimeSeriesSplit` variable immediately after declaring it. As a result, standard `KFold` is used, causing future transaction data to be used to train models predicting past transactions during the local cross-validation loop. This introduces data leakage and severely inflates the reported CV mean score of `0.9296`.

## 4. Cross-Cell Dependency Analysis

Based on a strict observation of the notebook's execution flow, several critical dependencies and state mutations are evident:
1.  **Dead Function Definitions:** The `reduce_mem_usage` function defined in Cell 6 is completely unreferenced by any subsequent cell. The massive `train_model_regression` function is similarly never called.
2.  **Feature Engineering to Cleaning Dependency:** In Cell 37, the execution of `.transform('std')` in the denominator mathematically guarantees the creation of `NaN`s (from groups of size 1) and `inf`s (from groups with 0 variance). This globally mutates the `train` and `test` dataframes. Because of this, Cell 46, which applies `clean_inf_nan` to convert `inf` to `NaN`, is strictly required. If Cell 46 were removed or run out of order, the LightGBM model in Cell 50 would crash, as LightGBM cannot natively handle `inf` values.
3.  **The Overwrite of the Splitting Strategy:** In Cell 49, the code strictly executes linearly:
    `folds = TimeSeriesSplit(n_splits=n_fold)`
    `folds = KFold(n_splits=5)`
    The variable `folds` is globally mutated on the second line, completely erasing the `TimeSeriesSplit` object. When `folds` is passed into `train_model_classification` in Cell 50, it passes the `KFold` object. 
4.  **Target and Feature Separation Dependency:** Cell 45 drops the `isFraud` target from `train` to create `X`, and assigns it to `y`. Running Cell 45 multiple times would fail because `isFraud` no longer exists in `train` after the first execution. 

## 5. Model Performance Summary

The notebook evaluates a single model architecture: LightGBM classification. The metric used to evaluate performance is the Area Under the Receiver Operating Characteristic Curve (ROC-AUC), which is appropriate given the heavily imbalanced nature of the fraud data.

The actual, observable output metrics from the 5-fold cross-validation training loop are:
-   **Fold 1:** Validation AUC = `0.919435`
-   **Fold 2:** Validation AUC = `0.931982`
-   **Fold 3:** Validation AUC = `0.926607`
-   **Fold 4:** Validation AUC = `0.946655`
-   **Fold 5:** Validation AUC = `0.923309`

-   **Cross-Validation Mean AUC:** `0.9296`
-   **Cross-Validation Standard Deviation:** `0.0095`

No comparison models (like XGBoost or CatBoost) are executed, despite their presence in the boilerplate helper functions and the commented-out cells at the end of the notebook.

## 6. Conclusions and Recommendations

The notebook represents a structurally complex but logically flawed implementation. It utilizes advanced, highly predictive feature engineering techniques and a robust, abstracted cross-validation loop. However, observable oversights severely limit the reliability of the model.

Based strictly on the observable code and text within the notebook, the following concrete recommendations are made:

1.  **Correct the Validation Split:** The most critical observable flaw is the data leakage introduced in Cell 49. The author explicitly noted in the EDA stage that a "time-based split for validation" was necessary because the transaction dates do not overlap. However, the code `folds = TimeSeriesSplit(n_splits=n_fold)` is immediately overwritten by `folds = KFold(n_splits=5)`. The `KFold` line must be deleted to honor the EDA insight, prevent data leakage, and produce a CV score that is realistically correlated with the unseen test set.
2.  **Execute Memory Reduction:** The dataset expands to 590,540 rows and over 434 columns. The author wrote a complex `reduce_mem_usage` function in Cell 6 specifically to downcast numerical data types and save RAM, but forgot to call it. The recommendation is to call `train = reduce_mem_usage(train)` and `test = reduce_mem_usage(test)` immediately after merging the identity tables in Cell 8 to prevent potential Out-Of-Memory (OOM) crashes on Kaggle servers.
3.  **Address Dead Code:** The notebook contains massive amounts of boilerplate code that is completely unused. The entire `train_model_regression` function, the XGBoost dictionary parameters, and the `group_mean_log_mae` metric function should be deleted to improve readability and code cleanliness.
4.  **Re-evaluate Division by Zero in Feature Engineering:** The feature engineering step divides continuous variables by their group's standard deviation. The author later cleans the resulting `inf` values by converting them to `NaN`. This means both missing data and mathematical infinity are being grouped together under the same `NaN` label for LightGBM to handle. The recommendation is to add logic to handle standard deviations of zero explicitly (e.g., adding a small epsilon value to the denominator like `+ 1e-9`) rather than creating infinite values that destroy the mathematical integrity of the feature.

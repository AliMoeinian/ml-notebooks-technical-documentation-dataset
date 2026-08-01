# Technical Documentation: End-to-End Machine Learning Project (California Housing)

## 1. Notebook Overview

This notebook is Chapter 2 of the book *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (1st edition) by Aurélien Géron. The notebook itself explicitly states this in its opening cell: *"This notebook contains all the sample code and solutions to the exercises in chapter 2."* The declared task is predicting median house values in Californian districts for a fictional organization called "Machine Learning Housing Corp."

The notebook is explicitly designed as a pedagogical textbook chapter, not a research paper. A second opening warning cell explicitly flags: *"Warning: this is the code for the 1st edition of the book. Please visit https://github.com/ageron/handson-ml2 for the 2nd edition code."* This signals that parts of the code may be outdated by later Scikit-Learn API changes, and numerous cells contain compatibility shim patterns (try/except for API differences between Scikit-Learn versions).

The overall analytical philosophy is methodical and layered: the author defines a problem, acquires data programmatically, explores it visually, engineers features manually via custom transformers, encapsulates the full transformation logic in scikit-learn `Pipeline` objects for reproducibility, trains and compares three model families, tunes the best one using both `GridSearchCV` and `RandomizedSearchCV`, and finally evaluates on a held-out stratified test set. The notebook concludes with five exercise solutions demonstrating advanced extensions such as SVR tuning, feature selection inside a pipeline, and joint hyperparameter search over both preprocessing and model parameters.

A defining characteristic is the consistent use of `np.random.seed(42)` at the start and at key re-initialization points (Cell 15) to enforce reproducibility. The notebook explicitly acknowledges why this matters: *"to make this notebook's output stable across runs."*

---

## 2. Environment and Dependencies

- **`__future__` (division, print_function, unicode_literals)**: Explicit Python 2/3 compatibility shims imported at the top. The notebook is designed to run on both Python 2 and Python 3, which constrains certain API choices.
- **`numpy`**: Used for random seed control, array operations, and mathematical utilities (`np.sqrt`, `np.ceil`, `np.linspace`, `np.sort`, `np.argpartition`).
- **`os` / `tarfile` / `urllib.request`**: Used for programmatic data download from a remote URL, extraction of a `.tgz` archive, and local path management.
- **`pandas`**: Used for loading the CSV data, displaying it (`head()`, `info()`, `describe()`, `value_counts()`), plotting histograms (`hist()`), computing correlations (`corr()`), scatter matrices (`pandas.plotting.scatter_matrix`), and constructing comparison DataFrames.
- **`matplotlib` (`matplotlib.pyplot` / `matplotlib.image`)**: Used for scatter plots of latitude/longitude, histograms of income categories, an overlay plot combining scatter data with a downloaded California map image, and visualizations of scipy probability distributions.
- **`zlib` / `hashlib`**: Used in the custom stable hash-based train/test split functions, demonstrating a CRC32-based identifier hashing approach for stable splits.
- **`scipy.stats`**: Used at two points — for computing 95% confidence intervals on the final RMSE using t-distribution and z-distribution (`stats.t.interval`, `stats.t.ppf`, `stats.norm.ppf`, `stats.sem`), and in Exercise 2 for defining sampling distributions for `RandomizedSearchCV` (`expon`, `reciprocal`, `geom`).
- **scikit-learn**:
  - `model_selection`: `train_test_split`, `StratifiedShuffleSplit`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`.
  - `impute` / `preprocessing`: `SimpleImputer` (with fallback to `Imputer`), `OrdinalEncoder` (with fallback from `future_encoders.py`), `OneHotEncoder`, `StandardScaler`, `FunctionTransformer`.
  - `base`: `BaseEstimator`, `TransformerMixin` — used to build two custom transformer classes: `CombinedAttributesAdder` and `TopFeatureSelector`.
  - `pipeline` / `compose`: `Pipeline`, `FeatureUnion` (legacy), `ColumnTransformer` (with fallback from `future_encoders.py`).
  - `linear_model`: `LinearRegression`.
  - `tree`: `DecisionTreeRegressor`.
  - `ensemble`: `RandomForestRegressor`.
  - `svm`: `SVR`.
  - `metrics`: `mean_squared_error`, `mean_absolute_error`.
- **`joblib`**: Used in the Extra Material section to serialize the trained pipeline to disk (`joblib.dump`) and reload it (`joblib.load`). The notebook explicitly comments out the deprecated `sklearn.externals.joblib` import path.

---

## 3. Per-Stage Documentation

### Stage 1: Setup and Figure Saving Infrastructure (Cells 0–5)

**[CODE]** Cell 5 sets a global `np.random.seed(42)`, imports matplotlib with inline backend, configures global font sizes for all plots (`mpl.rc('axes', labelsize=14)`, `mpl.rc('xtick', labelsize=12)`, `mpl.rc('ytick', labelsize=12)`), defines a path for saving figures (`./images/end_to_end_project/`), and defines a `save_fig()` utility function that calls `plt.tight_layout()` before `plt.savefig()` at 300 DPI resolution.

**[PROCESS/CONTEXT]** The notebook explicitly states the reason: *"to make this notebook's output stable across runs."* A consistent random seed is a prerequisite for reproducible training and splitting results across executions. The `save_fig()` function is a textbook utility that enforces a standard output format for all chapter figures.

**[RESULT]** No output is printed. The `images/end_to_end_project/` directory is created if absent. All subsequent visualization cells call `save_fig()` to simultaneously display inline and save a high-resolution PNG.

**[INSIGHT]** Setting `np.random.seed(42)` globally at the start does not guarantee reproducibility for all subsequent operations — it only controls NumPy's random state at that moment. Cell 15 explicitly re-calls `np.random.seed(42)` immediately before the custom `split_train_test()` function is defined and used, acknowledging that the global seed may have been consumed by earlier operations. This is correct practice: reset the seed immediately before the operation you want to be deterministic.

---

### Stage 2: Data Acquisition (Cells 6–10)

**[CODE]** Cell 7 defines a `fetch_housing_data()` function that downloads `housing.tgz` from a raw GitHub URL (`https://raw.githubusercontent.com/ageron/handson-ml/master/`), saves it to `datasets/housing/`, and extracts it using `tarfile.open().extractall()`. Cell 8 calls this function. Cell 9 defines `load_housing_data()`, which reads `datasets/housing/housing.csv` into a pandas DataFrame. Cell 10 calls `housing = load_housing_data()` and displays `housing.head()`.

**[PROCESS/CONTEXT]** Programmatic data download directly from a version-controlled source (a specific GitHub commit) makes the data acquisition fully reproducible. The notebook explicitly avoids assuming the data is already present on disk.

**[RESULT]** `housing.head()` displays 5 rows with 10 columns: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, `ocean_proximity`. The `ocean_proximity` column contains string values (e.g., `NEAR BAY`).

**[INSIGHT]** The URL points to the `master` branch of the repository, not a pinned commit. If the file is ever removed or modified upstream, this cell will fail silently or produce wrong data. A more robust approach would pin to a specific commit hash or include a local fallback.

---

### Stage 3: Data Exploration and Quality Assessment (Cells 11–14)

**[CODE]** `housing.info()` (Cell 11) reveals that `total_bedrooms` has only 20,433 non-null values out of 20,640 rows, confirming 207 missing values — the only column with missing data. `housing["ocean_proximity"].value_counts()` (Cell 12) shows the 5 distinct categories and their counts: `<1H OCEAN` is the most common. `housing.describe()` (Cell 13) shows min/max/mean/std for all 9 numeric features. Cell 14 calls `housing.hist(bins=50, figsize=(20,15))` to plot histograms of all 9 numeric features.

**[PROCESS/CONTEXT]** This standard EDA pass reveals key dataset characteristics before any transformations are applied.

**[RESULT]** From `info()`: `total_bedrooms` has 207 missing values; all other columns are complete. From `value_counts()`: `<1H OCEAN`: 9136, `INLAND`: 6551, `NEAR OCEAN`: 2658, `NEAR BAY`: 2290, `ISLAND`: 5. From `describe()`: `housing_median_age` and `median_house_value` are visibly capped (max values of 52 and 500,001 respectively — the 500,001 cap is explicitly shown). The histograms reveal heavily right-skewed distributions for `total_rooms`, `total_bedrooms`, `population`, `households`, and `median_income`. `housing_median_age` and `median_house_value` are both hard-capped: `housing_median_age` peaks at 52 and `median_house_value` has an abnormal spike at $500,000.

**[INSIGHT]** The hard cap on `median_house_value` at $500,001 is a critical data quality issue visible in the histogram and `describe()` output. The notebook never explicitly addresses this cap. Any model trained on this data will learn a hard ceiling of ~$500k, meaning districts that truly have higher median values (if such exist) would be systematically mispredicted. This is a direct observable problem in the data that the notebook's analysis silently ignores.

---

### Stage 4: Train/Test Splitting — Custom Implementations and Stable Hash Split (Cells 15–27)

**[CODE]** Cells 15–17 define and demonstrate a naive `split_train_test()` using `np.random.permutation()`. Cells 18–22 demonstrate two versions of `test_set_check()` using CRC32 (`zlib.crc32`) and MD5 (`hashlib.md5`) hash functions — these produce stable, reproducible splits by hashing a unique row identifier rather than relying on random state. Cell 23 applies the hash-based split using the DataFrame index as ID. Cell 24 constructs a composite ID `longitude * 1000 + latitude` to create a stable geographical identifier. Cell 26 shows the final approach: `sklearn.model_selection.train_test_split(housing, test_size=0.2, random_state=42)`.

**[PROCESS/CONTEXT]** The notebook explicitly narrates the progression: the naive permutation approach fails because re-running reshuffles the entire dataset, gradually exposing the whole dataset to the model. The hash-based approach ensures that the same rows always end up in the test set regardless of how many times the notebook is re-run or how many rows are added. This is an important concept for deployed systems where data grows over time.

**[RESULT]** Cell 17 output: `"16512 train + 4128 test"`. The hash-based splits produce similar proportions. `test_set.head()` (Cells 25, 27) shows a few sample rows from the test set.

**[INSIGHT]** The composite ID in Cell 24 (`longitude * 1000 + latitude`) is a clever but potentially brittle approach. If two districts share coordinates rounded to 3 decimal places, they would share the same ID and both be assigned to the same split — causing a small but non-zero collision risk. The notebook does not mention this edge case.

---

### Stage 5: Stratified Sampling on Income Categories (Cells 28–38)

**[CODE]** Cell 28 plots a histogram of `median_income`. Cell 30 creates `income_cat` using `pd.cut()` to bin `median_income` into 5 categories: `[0, 1.5)→1`, `[1.5, 3.0)→2`, `[3.0, 4.5)→3`, `[4.5, 6.0)→4`, `[6.0, ∞)→5`. Cell 31 shows the value counts of `income_cat`. Cells 32–33 use `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` to create `strat_train_set` and `strat_test_set`. Cells 34–36 compare the income category proportions in the overall dataset, the stratified test set, and a purely random test set. Cell 38 drops the `income_cat` column from both sets.

**[PROCESS/CONTEXT]** The notebook's markdown cells explain: since median income is likely the most important attribute for predicting housing prices, ensuring that the test set represents the income distribution of the full dataset prevents a scenario where, by chance, the test set over-represents very wealthy or very poor districts, which would make evaluation metrics unrepresentative.

**[RESULT]** From Cell 31, `income_cat` value counts: category 3 is most common (~7236 rows), followed by category 2, 4, 1, and 5. From Cell 36, the `compare_props` DataFrame shows that the stratified test set achieves near-zero percentage error on all 5 income categories (errors < 0.1%), while the random test set has errors up to ~3.6% on the least-common category.

**[INSIGHT]** A warning cell (Cell 29) documents a known bug from the book: the original code used `np.ceil()` with `.where()` to create income categories, which produced slightly different boundaries than `pd.cut()`. The corrected version uses `pd.cut()`. This is significant because it means the documentation (the book) and the code diverge — readers comparing outputs against the published book will see different values. The notebook explicitly flags this as a bug fix.

---

### Stage 6: Geographical Visualization (Cells 40–46)

**[CODE]** Cell 40 creates a working copy: `housing = strat_train_set.copy()`. Cell 41 plots a basic scatter of latitude vs. longitude. Cell 42 adds `alpha=0.1` to reduce overplotting. Cell 44 creates a full visualization with point size proportional to `population/100`, point color mapped to `median_house_value` using the `jet` colormap and a colorbar. Cell 45 programmatically downloads `california.png` from GitHub. Cell 46 overlays the scatter plot on the downloaded California map image using `plt.imshow()` with coordinate extents `[-124.55, -113.80, 32.45, 42.05]` and formats the colorbar to display dollar values (`"$%dk"`).

**[PROCESS/CONTEXT]** Cell 43 explicitly notes: *"The argument `sharex=False` fixes a display bug (the x-axis values and legend were not displayed). This is a temporary fix."* The progression from opaque dots (Cell 41) → semi-transparent (Cell 42) → population-sized and value-colored (Cell 44) → map overlay (Cell 46) is a deliberate pedagogical demonstration of iterative visualization improvement.

**[RESULT]** The final map overlay (Cell 46) shows a scatter plot where high-value districts (hot colors: red/orange) cluster along the California coast — particularly around the Bay Area (San Francisco peninsula) and the Los Angeles basin. Low-value districts (cool colors: blue) are concentrated in the Central Valley (inland). Point sizes show population concentration in coastal urban areas. The colorbar is formatted as `$0k` to `$500k`.

**[INSIGHT]** A second download from the GitHub `master` branch occurs in Cell 45 (for `california.png`). If this URL ever fails, Cell 46 will throw a `FileNotFoundError` when attempting to read the image. No local fallback is provided.

---

### Stage 7: Correlation Analysis and Feature Engineering (Cells 47–55)

**[CODE]** Cell 47 computes `corr_matrix = housing.corr()`. Cell 48 sorts correlations with `median_house_value` descending. Cell 49 generates a scatter matrix for `["median_house_value", "median_income", "total_rooms", "housing_median_age"]`. Cell 50 plots `median_income` vs. `median_house_value` with `alpha=0.1` and axis limits `[0, 16, 0, 550000]`. Cells 51–53 engineer three new ratio features directly on the `housing` DataFrame: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`, then recompute the correlation matrix. Cell 54 plots `rooms_per_household` vs. `median_house_value`.

**[PROCESS/CONTEXT]** Cell 52 contains an important note: *"there was a bug in the previous cell, in the definition of the rooms_per_household attribute. This explains why the correlation value below differs slightly from the value in the book."* The updated correlation matrix (Cell 53) reflects the corrected formula.

**[RESULT]** From Cell 48, the top correlations with `median_house_value`: `median_income` (0.688), `rooms_per_household` (0.151), `housing_median_age` (0.105), then slightly negative for total counts. From Cell 53 (after engineering): `bedrooms_per_room` shows a meaningful negative correlation with `median_house_value` (approximately -0.26), confirming that districts where a higher proportion of rooms are bedrooms tend to have lower house values. The scatter plot in Cell 50 shows a clear positive relationship between income and house value, but also reveals the $500,000 cap as a horizontal band of data points at the top of the plot.

**[INSIGHT]** The scatter plot in Cell 50 with `plt.axis([0, 16, 0, 550000])` explicitly sets the y-axis limit to 550,000 — above the $500,001 cap — making the horizontal band of capped values clearly visible as a visual artifact. The notebook comments on the cap indirectly through the axis choice but never explicitly recommends handling it.

---

### Stage 8: Prepare Data for ML — Missing Values and Encoding (Cells 56–88)

**[CODE]** Cell 57 separates labels from features. Cell 58 shows a sample of rows with missing values (only `total_bedrooms` is affected). Cells 59–61 demonstrate three strategies for handling missing values: drop rows, drop the feature, or fill with the median. Cell 63 instantiates `SimpleImputer(strategy="median")` with a try/except fallback to `sklearn.preprocessing.Imputer` for Scikit-Learn < 0.20. Cells 64–75 remove `ocean_proximity` (text), fit the imputer on numeric columns, verify `imputer.statistics_` matches manually computed medians, transform, and reconstruct a DataFrame. Cells 76–88 handle `ocean_proximity`: demonstrate `OrdinalEncoder` (with version fallback) and then `OneHotEncoder` in both sparse and dense modes. Cell 88 shows `cat_encoder.categories_` revealing the 5 category strings in the order used for encoding.

**[PROCESS/CONTEXT]** The notebook explicitly explains the rational for using `OrdinalEncoder` vs. `OneHotEncoder`: ordinal encoding implies a rank ordering between categories, which is not appropriate for `ocean_proximity`. `OneHotEncoder` creates a separate binary column for each category, which is the correct approach for nominal categories.

**[RESULT]** From Cell 67: `imputer.statistics_` contains 8 median values, one per numeric column. Cell 69 confirms these match `housing_num.median().values` exactly. Cell 80: `housing_cat_encoded[:10]` shows integers 0–4. Cell 81: `ordinal_encoder.categories_` = `[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])]`. Cell 87: the one-hot encoded array has shape `(16512, 5)` — one binary column per category.

**[INSIGHT]** Cells 79 and 83 both use try/except blocks to import from `future_encoders.py` if Scikit-Learn < 0.20. The `future_encoders.py` file is a companion file that must exist in the same directory as the notebook for this fallback to work. This dependency is never mentioned in the notebook itself — if the file is missing and the Scikit-Learn version is older than 0.20, the notebook will raise an `ImportError` with a misleading message.

---

### Stage 9: Custom Transformer — CombinedAttributesAdder (Cells 89–94)

**[CODE]** Cell 90 shows `housing.columns` to identify the column index positions. Cell 91 defines `CombinedAttributesAdder(BaseEstimator, TransformerMixin)`. It uses dynamic column index lookup: `rooms_ix, bedrooms_ix, population_ix, household_ix = [list(housing.columns).index(col) for col in (...)]`. The `transform()` method computes `rooms_per_household = X[:, rooms_ix] / X[:, household_ix]`, `population_per_household = X[:, population_ix] / X[:, household_ix]`, and optionally `bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]` (controlled by `self.add_bedrooms_per_room`). Cell 93 provides an alternative using `FunctionTransformer` with a standalone function `add_extra_features()`. Cell 94 wraps the output in a labeled DataFrame.

**[PROCESS/CONTEXT]** The notebook notes: *"Note: this feature selector assumes that you have already computed the feature importances somehow."* By implementing the transformer as a scikit-learn `BaseEstimator`/`TransformerMixin`, it gains `get_params()` and `set_params()` automatically (from `BaseEstimator`), making it compatible with `GridSearchCV` hyperparameter tuning — including the `add_bedrooms_per_room` boolean flag.

**[RESULT]** Cell 94 shows the first few rows of `housing_extra_attribs` with column names `rooms_per_household` and `population_per_household` appended.

**[INSIGHT]** The column index lookup in Cell 91 (`list(housing.columns).index(col)`) is computed at class definition time at module scope, not inside the `fit()` method. This means the indices `rooms_ix`, `bedrooms_ix`, etc. are computed once and hardcoded as module-level variables. If a different DataFrame is passed to `transform()` with a different column order, the indices would be silently wrong. The notebook is aware of this fragility: the comment in Cell 91 explicitly says *"safer than hard-coding indices 3, 4, 5, 6"* but the solution is only safer relative to hardcoding numeric literals — it is still fragile to column reordering.

---

### Stage 10: Full Preprocessing Pipelines (Cells 95–110)

**[CODE]** Cell 96 builds `num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)), ('std_scaler', StandardScaler())])`. Cell 100 builds `full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)])` and applies it: `housing_prepared = full_pipeline.fit_transform(housing)`. Cell 102 checks `housing_prepared.shape`. Cells 103–110 demonstrate the legacy `FeatureUnion` approach using `OldDataFrameSelector` and verify `np.allclose(housing_prepared, old_housing_prepared)` returns `True`.

**[PROCESS/CONTEXT]** The notebook explains the evolution from the older `FeatureUnion`+`DataFrameSelector` approach to the newer `ColumnTransformer`: *"It is now preferable to use the ColumnTransformer class that was introduced in Scikit-Learn 0.20."* Cell 110 explicitly verifies the two approaches produce identical outputs.

**[RESULT]** Cell 102: `housing_prepared.shape = (16512, 16)`. The 16 columns are: 8 original numeric features + 3 engineered ratios (`rooms_per_household`, `population_per_household`, `bedrooms_per_room`) — wait, `add_bedrooms_per_room=True` by default in `FunctionTransformer` call — actually with `add_bedrooms_per_room=False` in Cell 93, so 8 + 2 = 10 numeric + 5 one-hot ocean proximity + 1 (rooms_per_household) = revisiting: `add_extra_features` is called with `add_bedrooms_per_room=False` via kw_args, so 2 extra features → 8+2=10 numeric → 10+5 one-hot = 15. But `housing_prepared.shape` = 16 implies `add_bedrooms_per_room=True` for the actual pipeline in Cell 96 (no kw_args, so default `True`) → 8+3=11 numeric + 5 one-hot = 16. Cell 110 returns `True`.

**[INSIGHT]** The `add_extra_features` function in Cell 93 is called with `kw_args={"add_bedrooms_per_room": False}` — but the `num_pipeline` in Cell 96 calls it without `kw_args`, meaning it uses the function's default `add_bedrooms_per_room=True`. This is an inconsistency between Cell 93 and Cell 96 that could confuse readers: Cell 93 explicitly demonstrates excluding `bedrooms_per_room`, but the production pipeline in Cell 96 includes it. The notebook does not highlight this discrepancy.

---

### Stage 11: Model Training — Baseline Models (Cells 112–120)

**[CODE]** Cell 112 trains `LinearRegression()` on `housing_prepared`. Cell 113 generates predictions for 5 training instances and Cell 115 prints the corresponding labels. Cell 117 computes training RMSE using `mean_squared_error`. Cell 119 trains `DecisionTreeRegressor(random_state=42)`. Cell 120 computes its training RMSE.

**[PROCESS/CONTEXT]** Per the surrounding markdown, the linear model serves as a baseline. The Decision Tree is tested as a non-linear alternative.

**[RESULT]** Cell 113 predictions (5 samples): approximately `[286600, 340600, 196900, 46300, 254500]`. Cell 115 actual labels: `[286500, 340600, 196900, 46300, 254500]` (very close for these 5). Cell 117 `lin_rmse` ≈ 68,628. Cell 120 `tree_rmse` = 0.0 — the Decision Tree perfectly memorizes all training data.

**[INSIGHT]** `tree_rmse = 0.0` is the canonical demonstration of overfitting. The Decision Tree with no depth constraint and default hyperparameters creates one leaf node per training instance, achieving zero training error. This is used deliberately as an instructive example of why training RMSE alone is meaningless — the model has memorized the data rather than learned a generalizable pattern. The subsequent cross-validation section is set up specifically to reveal this.

---

### Stage 12: Cross-Validation — Model Comparison (Cells 121–130)

**[CODE]** Cell 122 runs `cross_val_score(tree_reg, ..., scoring="neg_mean_squared_error", cv=10)` and stores 10 fold scores. Cell 123 defines `display_scores()` which prints the array of scores, their mean, and standard deviation. Cell 124 runs the same 10-fold CV for `lin_reg`. Cell 126 trains `RandomForestRegressor(n_estimators=10, random_state=42)`. Cell 127 computes its training RMSE. Cell 128 runs 10-fold CV for the Random Forest. Cell 129 summarizes Linear Regression CV scores with `pd.Series.describe()`. Cell 130 trains `SVR(kernel="linear")` and computes its training RMSE.

**[PROCESS/CONTEXT]** Cell 125 (markdown) explicitly states: *"Note: we specify n_estimators=10 to avoid a warning about the fact that the default value is going to change to 100 in Scikit-Learn 0.22."* This confirms the notebook was written for Scikit-Learn versions before 0.22, where the default was 10 estimators.

**[RESULT]** Decision Tree CV (Cell 123): mean RMSE ≈ 71,407, std ≈ 2,439. Linear Regression CV (Cell 124): mean RMSE ≈ 69,052, std ≈ 2,731. Random Forest training RMSE (Cell 127): ≈ 21,933. Random Forest CV (Cell 128): mean RMSE ≈ 53,357, std ≈ 1,928. SVR (linear) training RMSE (Cell 130): ≈ 111,812.

**[INSIGHT]** The Decision Tree's CV RMSE (71,407) is worse than Linear Regression's (69,052) despite the Decision Tree achieving 0 training error — perfectly demonstrating the bias-variance tradeoff. The Random Forest's CV RMSE (53,357) substantially outperforms both, but the gap between its training RMSE (21,933) and CV RMSE (53,357) reveals significant overfitting even for the ensemble model with only 10 estimators.

---

### Stage 13: Hyperparameter Tuning — GridSearchCV and RandomizedSearchCV (Cells 131–141)

**[CODE]** Cell 131 defines a `param_grid` for `RandomForestRegressor` with two groups: 12 combinations of `{n_estimators: [3, 10, 30], max_features: [2, 4, 6, 8]}` and 6 combinations with `{bootstrap: False, n_estimators: [3, 10], max_features: [2, 3, 4]}`. A total of 18 combinations × 5 CV folds = 90 training runs. Cell 133 prints `grid_search.best_params_`. Cell 136 loops through `cv_results_` to print RMSE for all 18 combinations. Cell 138 runs `RandomizedSearchCV` with `n_iter=10` using `randint` distributions for `n_estimators` and `max_features`. Cell 140 extracts `feature_importances_` from `grid_search.best_estimator_`. Cell 141 maps importances to feature names and sorts descending.

**[PROCESS/CONTEXT]** The notebook comments: *"train across 5 folds, that's a total of (12+6)*5=90 rounds of training"* — transparently documenting the computational cost of grid search.

**[RESULT]** Cell 133: `grid_search.best_params_` = `{'max_features': 8, 'n_estimators': 30}`. Cell 136 shows all 18 RMSE scores; the best combination achieves approximately 49,682. Cell 141 (feature importances sorted): `median_income` has the highest importance, followed by `INLAND` (ocean_proximity one-hot), `population_per_household`, `bedrooms_per_room`, `rooms_per_household`. The `ISLAND` category has near-zero importance.

**[INSIGHT]** The `n_estimators` grid only goes up to 30. The fact that the best combination is `n_estimators=30` (the maximum tested) is a signal that even more estimators might improve performance — the search grid hit its upper boundary. The same issue occurs for `max_features=8` (the maximum tested). The notebook does not explicitly note this boundary effect, though it's observable from the grid search results.

---

### Stage 14: Final Evaluation on Test Set (Cells 142–150)

**[CODE]** Cell 142 selects the `final_model = grid_search.best_estimator_`, splits the test set into `X_test` and `y_test`, transforms `X_test` using `.transform()` (not `.fit_transform()`), and computes `final_predictions`. Cells 145–150 compute 95% confidence intervals for the final RMSE using scipy's t-distribution, and provide both manual and z-score-based implementations for comparison.

**[PROCESS/CONTEXT]** Cell 144 explains: *"We can compute a 95% confidence interval for the test RMSE."* The confidence interval provides a range for the true expected RMSE, accounting for uncertainty in the test sample estimate. Both t-score and z-score approaches yield nearly identical intervals because of the large test set size (4,128 samples).

**[RESULT]** Cell 143: `final_rmse` ≈ 47,766. The 95% confidence interval from Cell 146: approximately `[45,685, 49,913]`. The z-score interval from Cell 150 is nearly identical.

**[INSIGHT]** The final RMSE (≈47,766) is notably lower than the cross-validated RMSE estimate from GridSearch (≈49,682), which is the expected direction — the final model was fit on the full training set (not CV subsets), giving it more data. This consistency confirms the pipeline is methodologically sound with no data leakage between training and test sets.

---

### Stage 15: Extra Material — Full Pipeline Persistence (Cells 151–156)

**[CODE]** Cell 153 creates a `full_pipeline_with_predictor = Pipeline([("preparation", full_pipeline), ("linear", LinearRegression())])`. Cell 156 uses `joblib.dump(my_model, "my_model.pkl")` and then `joblib.load("my_model.pkl")` to demonstrate model serialization. The commented-out line `from sklearn.externals import joblib` explicitly documents the deprecated import path.

**[PROCESS/CONTEXT]** This section demonstrates production-readiness: the entire preprocessing and prediction pipeline can be serialized as a single object and reloaded in a deployment environment.

**[RESULT]** Cell 155 predictions for 4 training samples — approximate values. The pipeline successfully roundtrips through joblib.

**[INSIGHT]** The serialized pipeline (`my_model`) uses `LinearRegression` as the predictor, not the best `RandomForestRegressor` found during grid search. This is a deliberate simplification for demonstration purposes, but the notebook does not explicitly point out this choice, which could confuse readers who expect the best model to be serialized.

---

### Stage 16: Exercise Solutions (Cells 159–212)

**[CODE]**
- **Exercise 1**: Runs a `GridSearchCV` on `SVR` with two grids: linear kernel with `C` in `[10, 30, 100, 300, 1000, 3000, 10000, 30000]` and RBF kernel with `C` in `[1, 3, 10, 30, 100, 300, 1000]` and `gamma` in `[0.01, 0.03, 0.1, 0.3, 1, 3]` — 56+42=98 combinations × 5 folds = 490 training runs.
- **Exercise 2**: Replaces GridSearchCV with `RandomizedSearchCV(n_iter=50)` using `reciprocal(20, 200000)` for `C` and `expon(scale=1.0)` for `gamma`.
- **Exercise 3**: Defines `TopFeatureSelector(BaseEstimator, TransformerMixin)` with `indices_of_top_k()` using `np.argpartition`. Sets `k=5` and builds a pipeline adding feature selection after the preparation pipeline.
- **Exercise 4**: Creates `prepare_select_and_predict_pipeline` combining full preprocessing + `TopFeatureSelector` + `SVR(**rnd_search.best_params_)`.
- **Exercise 5**: Runs a `GridSearchCV` over the full pipeline, searching over `preparation__num__imputer__strategy` (`['mean', 'median', 'most_frequent']`) and `feature_selection__k` (all values from 1 to 16).

**[RESULT]**
- Exercise 1: Best SVR with linear kernel, best RMSE ≈ 70,232. Cell 167 notes: *"That's much worse than the RandomForestRegressor."* `best_params_` shows `C` hits the maximum tested value of 30,000.
- Exercise 2: Best SVR with RBF kernel, RMSE ≈ 54,767. Cell 175 notes this is *"much closer to the RandomForestRegressor."* `rnd_search.best_params_` shows kernel='rbf', specific C and gamma values.
- Exercise 3: `top_k_feature_indices` for k=5 returns the 5 most important feature column indices. `np.array(attributes)[top_k_feature_indices]` lists the feature names.
- Exercise 5: `grid_search_prep.best_params_` → `{'feature_selection__k': 15, 'preparation__num__imputer__strategy': 'most_frequent'}`. Cell 211 notes: *"The best imputer strategy is most_frequent and apparently almost all features are useful (15 out of 16). The last one (ISLAND) seems to just add some noise."*

**[INSIGHT]** Exercise 1 demonstrates the classic case where the best `C` value hits the upper boundary of the search grid (30,000), clearly indicating that the true optimal `C` lies above 30,000. The notebook itself acknowledges this in Cell 167: *"Notice that the value of C is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for C... because it is likely that higher values of C will be better."* This is excellent pedagogical practice — explicitly teaching the reader to recognize when grid search boundaries are binding.

---

## 4. Cross-Cell Dependency Analysis

| Producer Cell | Variable | Consumer Cells | Risk if Missing |
|---|---|---|---|
| Cell 5 | `save_fig()`, `IMAGES_PATH` | Cells 41, 42, 44, 45, 46, 49, 50, 54 | `NameError` in all visualization cells |
| Cell 7 | `fetch_housing_data()` | Cell 8 | `NameError` |
| Cell 9 | `load_housing_data()` | Cell 10 | `NameError` |
| Cell 10 | `housing` (full DataFrame) | Cells 11, 12, 13, 14, 24, 28, 30, 31, 32, 33, 34, 35, 36 | `NameError` |
| Cell 30 | `housing["income_cat"]` | Cells 31, 32, 33 | `KeyError` |
| Cell 33 | `strat_train_set`, `strat_test_set` | Cells 34, 35, 38, 40, 57, 142 | `NameError` — breaks entire downstream flow |
| Cell 40 | `housing` (reassigned to training copy) | Cells 41–55, 57 | Prior cells referring to full `housing` would silently operate on training set only |
| Cell 51 | `rooms_per_household`, `bedrooms_per_room`, `population_per_household` on `housing` | Cell 53, 54 | `KeyError` |
| Cell 57 | `housing` (features only), `housing_labels` | All model training cells | `NameError` |
| Cell 65 | `housing_num` | Cells 66–75, 96, 100 | `NameError` |
| Cell 91 | `rooms_ix, bedrooms_ix, population_ix, household_ix` | Cell 91 transform method, Cell 93 `add_extra_features()` | `NameError` — silently wrong indices if column order changes |
| Cell 96 | `num_pipeline` | Cell 100 | `NameError` |
| Cell 100 | `full_pipeline`, `housing_prepared` | Cells 101, 102, 108, 110, 112–130, 142, 186–210 | Breaks entire ML section |
| Cell 112 | `lin_reg` | Cells 113, 117, 124, 129, 153 | `NameError` |
| Cell 119 | `tree_reg` | Cells 120, 122 | `NameError` |
| Cell 126 | `forest_reg` | Cells 127, 128 | `NameError` |
| Cell 131 | `grid_search` (GridSearchCV on RandomForest) | Cells 133, 134, 136, 137, 140, 141, 142, 186–193 | `NameError` — breaks feature importance and final evaluation |
| Cell 140 | `feature_importances` | Cells 141, 183–193, 202, 209 | `NameError` — breaks all exercise 3/4/5 solutions |
| Cell 138 | `rnd_search` (on RandomForest) | Cell 139 | `NameError` |
| Cell 170 | `rnd_search` (on SVR, Exercise 2) | Cells 172, 174, 202, 203, 205 | `NameError` — the final exercise 4 pipeline uses `rnd_search.best_params_` |

**Critical execution order dependency**: Cell 40 reassigns `housing` from the full loaded DataFrame to `strat_train_set.copy()`. Any cell after Cell 40 that references `housing` operates on the training set only. This silent reassignment is the most critical global state mutation in the notebook.

---

## 5. Model Performance Summary

| Model | Training RMSE | 10-Fold CV RMSE (Mean ± Std) |
|---|---|---|
| Linear Regression | ~68,628 | ~69,052 ± 2,731 |
| Decision Tree (no constraints) | 0.0 | ~71,407 ± 2,439 |
| Random Forest (n_estimators=10) | ~21,933 | ~53,357 ± 1,928 |
| SVR (linear kernel) | ~111,812 | not computed |
| Random Forest (GridSearchCV best: n_estimators=30, max_features=8) | — | ~49,682 (best CV score) |
| **Final model on test set** | — | **RMSE ≈ 47,766** (95% CI: [45,685, 49,913]) |
| SVR GridSearch (Exercise 1, linear) | — | ~70,232 |
| SVR RandomizedSearch (Exercise 2, rbf) | — | ~54,767 |

The Decision Tree's CV RMSE (71,407) being higher than its training RMSE (0.0) is the most extreme overfitting example in the notebook. Linear Regression and the Decision Tree perform similarly on unseen data despite very different training behaviors. The Random Forest substantially outperforms both, and the tuned version on the full test set achieves the best performance.

---

## 6. Conclusions and Recommendations

**Observable problems in the notebook:**

1. **`median_house_value` cap at $500,001**: The histogram in Cell 14 and scatter plot in Cell 50 both show a hard cap. The `describe()` output confirms a max of 500,001. The notebook never addresses this, meaning the model learns to never predict above ~$500k. This is a significant and directly observable data quality issue that compromises the usefulness of any model trained on this data.

2. **GridSearch boundary effects**: In both the main tuning (Cell 131) and Exercise 1, the best parameters (`n_estimators=30`, `max_features=8`, SVR `C=30000`) hit the upper boundary of the search grid. The notebook explicitly warns about this only for the SVR case (Cell 167) but not for the Random Forest case.

3. **`housing` variable silent reassignment**: The notebook reassigns `housing` in Cell 40 from the full dataset to the training set copy without any marking or comment. This is a fragile pattern: any cell run out of order after Cell 40 that expects the full dataset will silently operate on training data only.

4. **`future_encoders.py` undeclared dependency**: Cells 79 and 83 import from `future_encoders.py` as a fallback, but this file's existence is never explained in the notebook.

5. **`bedrooms_per_room` inconsistency**: The `FunctionTransformer` in Cell 93 is called with `add_bedrooms_per_room=False`, but the `num_pipeline` in Cell 96 uses the same function with its default (`True`). The resulting `housing_prepared` has 16 columns (not 15), which is only consistent with `add_bedrooms_per_room=True`. This discrepancy is undocumented.

**Concrete recommendations (based only on what is in the notebook):**

1. **Cap handling**: The notebook shows the $500k cap explicitly in Cell 14. The logical extension would be to filter out `housing[housing["median_house_value"] < 500001]` before training, as the notebook itself implicitly acknowledges these rows are unreliable.
2. **GridSearch boundary extension**: When `best_params_` shows the edge of the search space (e.g., `n_estimators=30` is the maximum), extend the grid to include larger values. Cell 167 gives this advice for SVR — the same reasoning applies to the Random Forest grid.
3. **Joblib serialization alignment**: Cell 156 serializes a pipeline with `LinearRegression`, not the best `RandomForestRegressor`. For a production deployment, `my_model` should use `grid_search.best_estimator_` wrapped in the `full_pipeline`.

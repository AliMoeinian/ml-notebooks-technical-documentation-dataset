import sys

content = """# Technical Documentation: Modeling House Price with Regularized Linear Model & Xgboost

## 1. Notebook Overview
This technical documentation provides an in-depth, cell-by-cell analysis of a Jupyter notebook designed to predict house prices using a combination of data preprocessing techniques, a regularized linear model (ElasticNet), and gradient boosting (XGBoost). The notebook demonstrates a complete end-to-end machine learning pipeline starting from exploratory data analysis and handling missing values, moving through statistical transformations for normality, and concluding with training and evaluating multiple predictive models. The workflow explicitly relies on the data itself, extracting insights through distribution plots, correlation matrices, and model-derived feature importance metrics. The primary objective is to accurately model the `SalePrice` variable, minimizing the Root Mean Squared Error (RMSE) on a holdout test set by iteratively refining the feature set and algorithm hyperparameters.

## 2. Environment and Dependencies
The notebook relies on the following libraries and modules for its execution:
- **warnings**: Utilized to suppress warning messages during execution (`warnings.warn = ignore_warn`).
- **numpy (np)**: Essential for numerical operations and applying logarithmic transformations (`np.log1p`).
- **pandas (pd)**: Used for data manipulation, reading CSV files, handling missing values, generating dummy variables, and computing descriptive statistics.
- **matplotlib.pyplot (plt)**: Employed for creating static visualizations such as distribution plots, bar charts, and feature importance graphs. `%matplotlib inline` is used to render plots within the notebook.
- **seaborn (sns)**: Used for high-level statistical data visualization, specifically `distplot`, `heatmap`, and `pairplot`.
- **scipy.stats**: The `norm` and `skew` functions are used to compute and visualize the normal distribution fit and skewness of the data. `stats.probplot` is used to generate QQ-plots.
- **sklearn.preprocessing**: Imported but not explicitly used in the visible code cells.
- **sklearn.metrics**: `r2_score` and `mean_squared_error` are used to evaluate model performance on both training and testing datasets.
- **sklearn.model_selection**: `train_test_split` is used to partition the data into training and validation sets. `RandomizedSearchCV` and `StratifiedKFold` are imported but not visibly utilized in the provided cells.
- **sklearn.linear_model**: `ElasticNetCV` and `ElasticNet` are imported for regularized linear regression modeling.
- **xgboost**: `XGBRegressor` and `plot_importance` are imported for training gradient boosted decision trees and extracting feature significance.
- **collections**: `OrderedDict` is used to sort and display the feature importance scores extracted from the XGBoost model.

## 3. Per-Stage Documentation

### Stage 1: Data Loading and Initial Inspection (Cells 1-5)
**[CODE]**
The notebook begins by reading the dataset `house_train.csv` into a pandas DataFrame named `df`. The shape of the dataset is verified using `df.shape`, returning `(1460, 81)`, indicating 1460 rows and 81 columns. The proportion of missing values per column is calculated using `(df.isnull().sum() / len(df)).sort_values(ascending=False)[:20]`. Based on this output, the columns `PoolQC` (0.995), `MiscFeature` (0.963), `Alley` (0.938), `Fence` (0.808), and `Id` are explicitly dropped using `df.drop(..., axis=1, inplace=True)`. Finally, descriptive statistics for the target variable `SalePrice` are generated using `df['SalePrice'].describe()`.

**[PROCESS/CONTEXT]**
The initial data inspection is crucial for understanding the dimensionality and completeness of the dataset. Identifying columns with exceptionally high missing value rates informs the decision to remove them, as imputing data where over 80-99% is missing could introduce significant noise or bias. The descriptive statistics for `SalePrice` provide a baseline understanding of the target variable's central tendency and spread.

**[RESULT]**
The dataset is successfully loaded and reduced from 81 to 76 columns after dropping the specified features. The output of the missing value check shows that `PoolQC`, `MiscFeature`, `Alley`, and `Fence` have extreme missingness. The descriptive statistics for `SalePrice` reveal a mean of 180,921.196, a standard deviation of 79,442.503, a minimum of 34,900.000, and a maximum of 755,000.000.

**[INSIGHT]**
The descriptive statistics for `SalePrice` indicate a substantial difference between the 75th percentile (214,000) and the maximum value (755,000). This suggests the presence of outliers or a long right tail in the distribution, which warrants further visual and statistical analysis to determine if transformations are necessary for optimal model performance.

### Stage 2: Target Variable Distribution Analysis and Transformation (Cells 6-7)
**[CODE]**
A distribution plot of `SalePrice` is generated using `sns.distplot` fitted with a normal distribution (`norm.fit`). The mean (`mu = 180921.20`) and standard deviation (`sigma = 79415.29`) are printed. A QQ-plot is also generated using `stats.probplot`. Subsequently, a logarithmic transformation (`np.log1p`) is applied to `SalePrice`, and the distribution and QQ-plots are regenerated. The new fitted parameters are `mu = 12.02` and `sigma = 0.40`.

**[PROCESS/CONTEXT]**
Many statistical models, including linear regression, perform better when the target variable and features follow a normal distribution. The initial plots are used to visually assess the skewness of `SalePrice`. The logarithmic transformation (`np.log1p`, which applies `log(1+x)`) is a standard mathematical technique used to reduce rightward skewness, pulling extreme values closer to the mean and stabilizing the variance.

**[RESULT]**
The initial distribution plot and QQ-plot clearly show that `SalePrice` is right-skewed and deviates from a theoretical normal distribution, particularly at the higher quantiles. After applying `np.log1p`, the new distribution plot and QQ-plot demonstrate a shape that much more closely resembles a normal distribution, with the points on the QQ-plot aligning tightly with the diagonal reference line.

**[INSIGHT]**
The successful normalization of the `SalePrice` variable via the `np.log1p` transformation confirms that the data was originally heavily skewed by expensive outlier properties. By transforming the target, the models subsequently trained will be optimizing for the log-scaled price, which inherently penalizes relative errors rather than absolute errors, potentially leading to more robust predictions across the entire price spectrum.

### Stage 3: Correlation Analysis and Feature Interactions (Cells 8-14)
**[CODE]**
A Pearson correlation heatmap is created for all numeric features excluding `SalePrice`. A bar chart is then plotted to show the correlation of each numeric feature specifically with `SalePrice`, sorted in descending order. Finally, a pairplot is generated for the target variable and the top three correlated features: `OverallQual`, `GrLivArea`, and `GarageCars`.

**[PROCESS/CONTEXT]**
Correlation analysis serves two main purposes: identifying multicollinearity among predictors and finding the predictors most strongly associated with the target variable. The heatmap visually highlights pairs of features that are highly correlated with each other. The bar chart isolates the relationship between features and the target. The pairplot allows for a granular, pairwise scatter plot examination of the most influential variables to detect linear or non-linear trends and potential outliers.

**[RESULT]**
The heatmap reveals strong correlations between certain predictor pairs (e.g., `GarageYrBlt` and `YearBuilt`, `TotRmsAbvGrd` and `GrLivArea`, `GarageArea` and `GarageCars`). The bar chart shows that `OverallQual` has the highest correlation with `SalePrice` (around 0.8), followed by `GrLivArea` (>0.7) and `GarageCars` (>0.6). The pairplot visualizes these relationships, showing positive linear trends between these three features and `SalePrice`.

**[INSIGHT]**
The strong correlations between predictor pairs indicate the presence of multicollinearity, meaning that some features express redundant information. The notebook explicitly states the intention to use ElasticNetCV to handle this redundancy, implying an understanding that regularized models can mitigate the instability caused by multicollinear features. The pairplot confirms that higher overall quality, larger living area, and greater garage capacity directly correlate with higher house prices.

### Stage 4: Feature Skewness Correction and Encoding (Cells 15-18)
**[CODE]**
The `np.log1p` transformation is permanently applied to `df["SalePrice"]`. Numeric features are identified, and their skewness is calculated using `skew()`. Features with a skewness strictly greater than 0.75 are isolated, and the `np.log1p` transformation is applied to them. Categorical variables are then one-hot encoded using `pd.get_dummies(df)`. Remaining missing values are filled with the mean of their respective columns using `df.fillna(df.mean())`. The dataset is then split into `X` (features) and `y` (target), and further partitioned into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets using an 80/20 split (`test_size=0.2`).

**[PROCESS/CONTEXT]**
Just as the target variable was normalized, highly skewed predictor variables are also log-transformed to satisfy the assumptions of linear modeling and improve algorithmic stability. Categorical variables must be converted to numeric formats (dummy encoding) for compatibility with scikit-learn and XGBoost models. Mean imputation is a simple technique to ensure no NaN values remain, which would otherwise cause errors during model fitting. The train-test split is a standard validation strategy to evaluate model generalization on unseen data.

**[RESULT]**
The target and highly skewed numeric features are log-transformed. The DataFrame expands horizontally to accommodate the dummy variables created for categorical features. All missing values are replaced by column means. The data is successfully divided into `X_train`, `X_test`, `y_train`, and `y_test`. A `RuntimeWarning` is output during the logarithmic transformation, indicating that an invalid value (likely a negative number or zero in a context where it caused an issue, though `log1p` handles zeros) was encountered.

**[INSIGHT]**
The decision to apply mean imputation across the entire dataset after dummy encoding treats all missing numerical data uniformly. This approach guarantees that the models can be trained without execution errors. Furthermore, the log transformation of predictors ensures that the subsequent ElasticNet model will receive data that more closely approximates a normal distribution, optimizing the behavior of its penalty terms.

### Stage 5: ElasticNetCV Modeling and Feature Selection (Cells 19-24)
**[CODE]**
An `ElasticNetCV` model is instantiated with a grid of `l1_ratio` values `[.1, .5, .7, .9, .95, .99, 1]`, `n_alphas=100`, and 6-fold cross-validation (`cv=6`). The model is fitted on `X_train` and `y_train`. The optimal parameters found are `alpha: 0.00013634` and `l1_ratio: 0.700`, converging in 84 iterations. Predictions are made, and evaluation metrics are calculated: Train R2 `0.9352`, Test R2 `0.8300`, Train RMSE `0.0963`, and Test RMSE `0.1604`. Finally, feature importances (absolute coefficients) are extracted. Features with coefficients strictly greater than zero are counted, showing 113 features retained, which represents a reduction of 58.91%. A bar chart of the top 30 features is plotted.

**[PROCESS/CONTEXT]**
ElasticNet is a regularized regression method that linearly combines the L1 (Lasso) and L2 (Ridge) penalties. By using the cross-validation variant (`ElasticNetCV`), the model automatically searches for the optimal combination of these penalties (`l1_ratio` and `alpha`) to minimize error. Because the L1 penalty can shrink coefficients exactly to zero, ElasticNet inherently performs feature selection, dropping irrelevant or highly collinear variables.

**[RESULT]**
The model identifies a predominantly L1-weighted penalty (`l1_ratio=0.7`) as optimal, indicating that a sparser model is preferred. The model achieves a Test RMSE of 0.1604. Crucially, the model drives 58.91% of the feature coefficients to exactly zero, utilizing only 113 of the available features.

**[INSIGHT]**
The significant reduction in features (nearly 60%) confirms the earlier hypothesis that the dataset contained substantial redundancy and multicollinearity, especially after the expansion caused by dummy encoding. The ElasticNet model effectively navigates this high-dimensional space, successfully acting as an automated feature selector while establishing a strong baseline performance metric.

### Stage 6: XGBoost Baseline and Hyperparameter Tuning (Cells 25-33)
**[CODE]**
A series of `XGBRegressor` models are trained and evaluated:
- **Model 1**: Default parameters. Train R2: 0.9546, Test R2: 0.8856, Train RMSE: 0.0820, Test RMSE: 0.1316.
- **Model 2**: `n_estimators=1000`, `early_stopping_rounds=5`, evaluated on `[(X_test, y_test)]`. Train R2: 0.9678, Test R2: 0.8931, Train RMSE: 0.0699, Test RMSE: 0.1272.
- **Model 3**: `n_estimators=1000`, `learning_rate=0.05`, `early_stopping_rounds=5`. Train RMSE: 0.0869, Test RMSE: 0.1325.
- **Model 4**: `n_estimators=100`, `learning_rate=0.08`, `gamma=0`, `subsample=0.75`, `colsample_bytree=1`, `max_depth=7`. Train RMSE: 0.0455, Test RMSE: 0.1290.
The notebook concludes that `xgb_model2` is the best model.

**[PROCESS/CONTEXT]**
XGBoost (Extreme Gradient Boosting) is a powerful, non-linear ensemble algorithm. The author iteratively tests different configurations to improve predictive accuracy. `early_stopping_rounds` is used in Models 2 and 3 to halt training when the performance on the evaluation set (`X_test`, `y_test`) stops improving, preventing overfitting over a large number of trees (1000 estimators). Manual tuning of parameters like `learning_rate` and `max_depth` is attempted to further optimize the model.

**[RESULT]**
Even the default XGBoost model (Model 1) significantly outperforms the ElasticNet model on the test set (RMSE 0.1316 vs 0.1604). Incorporating a larger number of estimators with early stopping (Model 2) yields the lowest Test RMSE thus far (0.1272). Attempts to adjust the learning rate (Model 3) or manually specify a suite of tree parameters (Model 4) fail to improve upon the Test RMSE achieved by Model 2. Model 4 exhibits the lowest Train RMSE (0.0455), indicating it fits the training data very closely.

**[INSIGHT]**
The fact that Model 4 achieves a highly superior training error (0.0455) but a worse test error (0.1290) compared to Model 2 perfectly illustrates the concept of overfitting. Model 2's use of `early_stopping_rounds` provides a highly effective, dynamic defense against overfitting, allowing the algorithm to find the optimal number of boosting iterations natively, proving superior to the manual parameter guessing attempted in Model 4.

### Stage 7: XGBoost Feature Selection and Final Modeling (Cells 34-38)
**[CODE]**
The feature F-scores from the best model (`xgb_model2`) are extracted and sorted in descending order using an `OrderedDict`. The top features are `LotArea` (68), `GrLivArea` (64), and `OverallQual` (56). A list named `most_relevant_features` is created by filtering the dataset to include only features with an F-score strictly greater than or equal to 4. A new training set `train_x` is created containing only these filtered features. A new train-test split is executed using `train_x`. Finally, `xgb_model5` is trained using `n_estimators=1000`, `early_stopping_rounds=5`, and `eval_set=[(X_test, y_test)]`. The evaluation metrics are Train R2: 0.9585, Test R2: 0.8950, Train RMSE: 0.0787, Test RMSE: 0.1261.

**[PROCESS/CONTEXT]**
Gradient boosting trees can inherently calculate feature importance based on how frequently a feature is used to split the data (F-score) and how much those splits improve the model. By extracting these scores, the author performs a wrapper-based feature selection technique. By filtering out low-importance features (scores < 4) and retraining the model on the reduced dataset, the author aims to eliminate noise and force the algorithm to focus entirely on the most robust predictive signals.

**[RESULT]**
The F-scores reveal that `LotArea`, `GrLivArea`, and `OverallQual` are the most critical features for the XGBoost model's decision-making process. The filtering process successfully subsets the data. The final retrained model (`xgb_model5`) achieves the lowest Test RMSE in the entire notebook (0.1261) and the highest Test R2 (0.8950).

**[INSIGHT]**
The improvement seen in Model 5 confirms that XGBoost, despite its robustness, can be slightly negatively impacted by the presence of low-signal, noisy features. By aggressively pruning the feature space based on the algorithm's own internal logic, the model's variance is reduced, leading to a small but measurable improvement in generalization on the test set.

## 4. Cross-Cell Dependency Analysis
- **Logarithmic Transformation Dependency:** The application of `np.log1p` to `df["SalePrice"]` in Cell 16 fundamentally alters the scale and meaning of the target variable for the remainder of the notebook. Consequently, every evaluation metric reported subsequently (such as the Train RMSE of 0.0963 for ElasticNet or the Test RMSE of 0.1261 for XGBoost) is calculated in the log-transformed space, not in absolute dollars. This is a critical dependency; predictions generated by these models must be inverse-transformed (`np.expm1`) before they can be interpreted as actual currency values.
- **Feature Selection Pipeline Dependency:** The final model (`xgb_model5`) is completely dependent on the exact state and learned parameters of `xgb_model2`. The dataset `train_x` constructed in Cell 36 is derived dynamically by querying `xgb_model2.get_booster().get_fscore()`. If `xgb_model2` had been instantiated with a different random state or had stopped at a different iteration, the F-scores would fluctuate, altering the list of features exceeding the threshold of 4, which would in turn change the structure of the dataset fed into the final model.
- **Data Splitting Dependency:** The notebook performs two separate `train_test_split` operations with the exact same `random_state=0` and `test_size=0.2`. The first split (Cell 18) acts on the full dummy-encoded dataset. The second split (Cell 36) acts on the filtered `train_x` dataset. Because the `random_state` is identical, the row indices for the training and testing sets are guaranteed to be identical across both operations, ensuring that the Test RMSE comparisons between Model 2 and Model 5 are valid, apples-to-apples evaluations on the exact same holdout observations.

## 5. Model Performance Summary
**[CODE]**
Model performance is universally evaluated using `r2_score` and `mean_squared_error` (which is then square-rooted to produce RMSE) from `sklearn.metrics`. These metrics are calculated for both the training and testing datasets for every trained model. 

**[PROCESS/CONTEXT]**
Tracking both R-squared and RMSE on train and test sets simultaneously provides a comprehensive view of model accuracy and generalization capability. R-squared indicates the proportion of the variance in the dependent variable that is predictable from the independent variables, providing a relative measure of fit. RMSE provides an absolute measure of the average magnitude of the errors, in the same units as the (log-transformed) target variable. Comparing train vs. test metrics directly exposes the degree of overfitting.

**[RESULT]**
The performance progresses sequentially:
- **ElasticNetCV:** Train RMSE: 0.0963, Test RMSE: 0.1604 (Baseline Linear)
- **XGBoost Model 1:** Train RMSE: 0.0820, Test RMSE: 0.1316 (Default Tree)
- **XGBoost Model 2:** Train RMSE: 0.0699, Test RMSE: 0.1272 (Early Stopping)
- **XGBoost Model 3:** Train RMSE: 0.0869, Test RMSE: 0.1325 (Modified Learning Rate)
- **XGBoost Model 4:** Train RMSE: 0.0455, Test RMSE: 0.1290 (Manual Params - Overfit)
- **XGBoost Model 5:** Train RMSE: 0.0787, Test RMSE: 0.1261 (Feature Selected)

**[INSIGHT]**
The evaluation metrics clearly demonstrate that non-linear, ensemble methods (XGBoost) capture the complexities of the housing dataset far better than regularized linear methods (ElasticNet), improving the Test RMSE from 0.1604 to 0.1316 instantly. Furthermore, the metrics show that while manual hyperparameter tuning (Model 4) can drastically reduce training error, it fails to improve test performance. The most effective optimization strategies observed are dynamic early stopping (Model 2) and algorithmic feature pruning (Model 5). However, passing `X_test` and `y_test` into the `eval_set` parameter for early stopping in Models 2, 3, and 5 means the test set is actively influencing the training loop (determining when it stops). This introduces a form of data leakage, meaning the reported Test RMSEs for these models might be slightly optimistic and not truly representative of performance on entirely unseen data.

## 6. Conclusions and Recommendations
The notebook outlines a highly logical and effective machine learning methodology. It successfully navigates the prerequisites of linear modeling by addressing skewness through logarithmic transformations and handling multicollinearity via ElasticNet. It then pivots to gradient boosting, establishing that complex tree ensembles provide superior predictive power for this dataset. Finally, it demonstrates an advanced technique by using the ensemble's own feature importance metrics to recursively strip away noise and optimize the final model.

**Recommendations:**
1. **Address Data Leakage in Early Stopping:** The use of the final holdout test set (`X_test`, `y_test`) as the `eval_set` for XGBoost's `early_stopping_rounds` compromises the purity of the test evaluation. A more statistically sound approach requires splitting the training data a second time to create a dedicated validation set. This validation set should be used for early stopping, reserving the test set exclusively for the final, unbiased performance calculation.
2. **Translate Metrics for Business Interpretability:** The final Test RMSE of 0.1261 is in log-space. To make the model's error interpretable to stakeholders (e.g., real estate agents, buyers), the predictions and the target variables should be transformed back to absolute dollars using `np.expm1()` before the final RMSE is calculated. This will provide an error metric representing the average dollar-value mistake the model makes.
3. **Refine Imputation Strategies:** Dropping features with high missingness (like `PoolQC`) permanently destroys potential predictive signal. A house having a pool is likely highly relevant to its price. Creating a binary indicator column (e.g., `Has_Pool`: 1 if present, 0 if NaN) before dropping the original column could preserve this information. Additionally, universally filling remaining NaNs with the column mean is mathematically safe but logically flawed for certain features. For example, a missing value in a garage-related numeric feature likely indicates the absence of a garage (a true value of 0), not an average-sized garage. Applying domain-specific, conditional imputation would likely yield a cleaner, more accurate dataset and further improve the final model's performance.
"""

with open('d:/THESIS_IMPLEMENTATION/Ground Truth Technical Documentation notebooks/Notebooks from GitHub/Modeling House Price with Regularized Linear Model & Xgboost_v2.md', 'w', encoding='utf-8') as f:
    f.write(content)

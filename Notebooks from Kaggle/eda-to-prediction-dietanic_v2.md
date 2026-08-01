# Technical Documentation: EDA To Prediction (DieTanic)

## 1. Notebook Overview

### Problem Statement
The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. That's why the name **DieTanic**. This notebook focuses on the Titanic Dataset, emphasizing that it is an ideal dataset for beginners to start a journey in data science. The objective of this notebook is to give an idea of how the workflow operates in any predictive modeling problem, detailing how to check features, add new features, and understand Machine Learning Concepts in a basic, understandable way.

### Overall Analytical Approach
The analytical approach is highly structured and split into three definitive parts:
- **Part 1: Exploratory Data Analysis (EDA):** Analysis of the features and finding relations or trends considering multiple features. The author breaks down analysis by feature type (Categorical, Ordinal, and Continuous).
- **Part 2: Feature Engineering and Data Cleaning:** Adding new features, removing redundant ones, and converting features into a suitable numerical form for predictive modeling.
- **Part 3: Predictive Modeling:** Running basic algorithms (Logistic Regression, SVM, Random Forest, KNN, Naive Bayes, Decision Tree), Cross Validation (K-Fold), Ensembling (Voting, Bagging, Boosting), and extracting important features.

### Notable Characteristics
- Heavily relies on categorical splits using `seaborn` factorplots and `pandas` crosstabs.
- The notebook takes a very pedagogical tone, explaining basic concepts like "Types of Features" (Categorical vs Ordinal vs Continuous) and "Correlation" directly in markdown.
- It actively avoids complex multivariate modeling initially, instead opting for bivariate analysis against the target variable (`Survived`).

## 2. Environment and Dependencies

The notebook uses standard data science libraries in Python. Their specific roles in this notebook are as follows:
- **numpy (`np`)**: Used for numerical operations, though its explicit use in the notebook code is minimal, mostly imported as a standard dependency alongside pandas.
- **pandas (`pd`)**: Fundamental for data loading (`pd.read_csv`), manipulation, creating bins (`pd.qcut`), and generating cross-tabulations (`pd.crosstab`) to show survivability across categories.
- **matplotlib.pyplot (`plt`)**: Used to configure subplots (`plt.subplots`), adjust figure sizes, set titles, and render the visualizations. The notebook sets the plot style to `'fivethirtyeight'`.
- **seaborn (`sns`)**: The primary visualization engine. Used for pie charts, countplots, barplots, violinplots, heatmaps for correlation and confusion matrices, and extensive use of `factorplot` (to show relationships across multiple categories).
- **warnings**: Used specifically to filter out and ignore warning messages (`warnings.filterwarnings('ignore')`).
- **sklearn.linear_model.LogisticRegression**: Used as a baseline classification algorithm and as one of the estimators in the Voting Classifier.
- **sklearn.svm.SVC**: Support Vector Machine classifier, trained with both 'linear' and 'rbf' kernels for baseline comparison and ensembling.
- **sklearn.ensemble**:
  - `RandomForestClassifier`: Used as a baseline, in the voting classifier, and for extracting feature importances.
  - `VotingClassifier`: Used to create a soft-voting ensemble from multiple baselines.
  - `BaggingClassifier`: Used to create bagged versions of KNN and Decision Trees.
  - `AdaBoostClassifier`: Used for adaptive boosting, showing the highest tuned performance.
  - `GradientBoostingClassifier`: Used for gradient boosting comparison.
- **sklearn.neighbors.KNeighborsClassifier**: Used as a baseline algorithm and plotted across different `n_neighbors` values.
- **sklearn.naive_bayes.GaussianNB**: Used for probabilistic baseline classification.
- **sklearn.tree.DecisionTreeClassifier**: Used as a baseline model and as the base estimator for bagging and boosting.
- **sklearn.model_selection**:
  - `train_test_split`: Used to create a 70-30 split of the data, stratified by the target.
  - `KFold`: Used to define a 10-fold cross-validation strategy.
  - `cross_val_score`: Used to compute the CV mean accuracy.
  - `cross_val_predict`: Used to generate out-of-fold predictions for confusion matrices.
  - `GridSearchCV`: Used to perform hyperparameter tuning for SVM and Random Forest.
- **sklearn.metrics**: `accuracy_score` and `confusion_matrix` are used to evaluate model predictions.
- **xgboost (`xg`)**: Used specifically for `XGBClassifier` to evaluate extreme gradient boosting against sklearn's ensemble methods.

## 3. Per-Stage Documentation

### Stage 1: Data Loading and Initial Assessment
**[CODE]** The notebook imports libraries, reads the training dataset from `../input/train.csv`, and displays the first few rows using `data.head()`. It then checks for null values using `data.isnull().sum()`.
**[PROCESS/CONTEXT]** The author notes that "Age, Cabin and Embarked have null values" and promises to fix them.
**[RESULT]** No code output is present in the JSON file. The text notes that only around 350 out of 891 training passengers survived (38.4%).
**[INSIGHT]** The `outputs` fields for all code cells in the provided notebook JSON are empty, indicating the notebook was cleared of outputs before saving. All actual numbers documented below are extracted from the author's markdown cells where they explicitly state the observed results.

### Stage 2: EDA on Sex (Categorical Feature)
**[CODE]** Uses `data.groupby` to count survivals by sex, plots a bar chart of mean survival by sex, and a seaborn `countplot` showing survived vs dead split by sex.
**[PROCESS/CONTEXT]** To check the survival rate based on gender.
**[RESULT]** The markdown states survival for women is around 75% while for men it is around 18-19%.
**[INSIGHT]** The author correctly identifies this as a "very important feature for modeling". However, analyzing features in isolation might lead to missing interaction effects, which the author subsequently addresses.

### Stage 3: EDA on Pclass (Ordinal Feature)
**[CODE]** Generates a pandas crosstab styled with a background gradient for `Pclass` vs `Survived`. Plots a bar chart of passenger counts by `Pclass` and a countplot of survival. Next, it generates a 3-way crosstab of `Sex`, `Survived`, and `Pclass`, visualized with `sns.factorplot`.
**[PROCESS/CONTEXT]** To verify if money and status mattered during the rescue.
**[RESULT]** Markdown states Pclass 1 survival is around 63%, Pclass 2 is 48%, and Pclass 3 is around 25%. For Women from Pclass 1, survival is about 95-96%.
**[INSIGHT]** The conclusion "Money Can't Buy Everything... But we can clearly see that Passengers Of Pclass 1 were given a very high priority" is directly supported by the crosstab analysis. 

### Stage 4: EDA on Age (Continuous Feature)
**[CODE]** Prints min, max, and mean age. Plots violinplots comparing `Pclass` and `Age` split by `Survived`, and `Sex` and `Age` split by `Survived`.
**[PROCESS/CONTEXT]** To observe the survival distribution across ages.
**[RESULT]** The oldest passenger was 80 years old. Maximum deaths occurred in the 30-40 age group.
**[INSIGHT]** The author notes that survival chances for men decrease with an increase in age, while children below 10 look to have good survival regardless of Pclass.

### Stage 5: Imputing Age using Name Initials
**[CODE]** Uses regex `([A-Za-z]+)\.` to extract salutations into an `Initial` column. Replaces misspelled or rare initials (Mlle, Mme, Ms, Dr, Major, Lady, Countess, Jonkheer, Col, Rev, Capt, Sir, Don) with common ones (Mr, Mrs, Miss, Other). Fills NaN `Age` values by assigning the ceiling mean age of each `Initial` group using `.loc`.
**[PROCESS/CONTEXT]** The author points out that blindly assigning the overall mean age (29) to a 4-year-old kid is problematic. Extracting titles from names allows group-specific mean imputation.
**[RESULT]** All 177 missing Age values are filled.
**[INSIGHT]** This is a very robust imputation methodology observable directly in the code. It leverages an otherwise useless string column (`Name`) to fix missing numeric data intelligently.

### Stage 6: EDA on Embarked
**[CODE]** Generates crosstabs and multiple factorplots/countplots analyzing `Embarked` against `Pclass`, `Sex`, and `Survived`. Replaces NaN values in `Embarked` with 'S' using `fillna`.
**[PROCESS/CONTEXT]** To analyze survival by port of embarkation and handle missing port data.
**[RESULT]** The text notes port C has the highest survival (~0.55). Most passengers boarded from S, and Port Q had almost 95% Pclass 3 passengers.
**[INSIGHT]** The author correctly deduces that the low survival rate at Port S is because the vast majority of boarding passengers were Pclass 3, avoiding a false correlation between port geography and survival.

### Stage 7: EDA on SibSp and Parch
**[CODE]** Generates crosstabs and barplots analyzing `SibSp` (siblings/spouses) and `Parch` (parents/children) against `Survived` and `Pclass`.
**[PROCESS/CONTEXT]** To see if being alone or with a family affects survival.
**[RESULT]** Survival for families with 5-8 members is 0%.
**[INSIGHT]** The author observes that large families (SibSp > 3) were all in Pclass 3, which explains their 0% survival rate. This is excellent cross-feature validation.

### Stage 8: EDA on Fare
**[CODE]** Prints max, min, mean fare. Plots `distplot` of Fare for each Pclass.
**[PROCESS/CONTEXT]** To observe the distribution of fares.
**[RESULT]** Lowest fare is 0.0.
**[INSIGHT]** The text notes a large distribution in Pclass 1 fares and plans to use binning.

### Stage 9: Correlation Analysis
**[CODE]** Plots a `sns.heatmap(data.corr())`.
**[PROCESS/CONTEXT]** To check for positive or negative correlation and multi-collinearity to eliminate redundant features.
**[RESULT]** Highest correlation is between SibSp and Parch (0.41).
**[INSIGHT]** Since no features are highly correlated (e.g. > 0.8), the author decides to keep all current features.

### Stage 10: Feature Engineering - Age_band and Fare_cat
**[CODE]** Creates `Age_band` by manually binning `Age` into 5 categories (<=16, 17-32, 33-48, 49-64, >64) using `.loc`. Creates `Fare_Range` using `pd.qcut(data['Fare'], 4)` to split fares into 4 equal quantiles, then creates a discrete `Fare_cat` (0,1,2,3) using manual `.loc` boundaries derived from the qcut ranges.
**[PROCESS/CONTEXT]** The author states continuous variables are problematic for machine learning and need to be converted to categorical via binning.
**[RESULT]** Continuous variables are successfully discretized.
**[INSIGHT]** While the code executes binning successfully, manually hardcoding the bin boundaries (e.g., `data['Fare']<=7.91`) instead of dynamically applying the `qcut` codes creates data leakage and breaking logic if new data with different quantiles is introduced.

### Stage 11: Feature Engineering - Family_Size and Alone
**[CODE]** Creates `Family_Size = Parch + SibSp`. Creates `Alone` feature which equals 1 if `Family_Size == 0`, else 0.
**[PROCESS/CONTEXT]** To combine family data into a single feature for better signal.
**[RESULT]** Plots show being alone is harmful except for females in Pclass 3.
**[INSIGHT]** This is a straightforward logical extraction that simplifies the model's decision boundaries.

### Stage 12: Data Cleaning and Dropping Features
**[CODE]** Replaces categorical strings with numbers (`Sex`, `Embarked`, `Initial`). Drops unneeded columns (`Name`, `Age`, `Ticket`, `Fare`, `Cabin`, `Fare_Range`, `PassengerId`). Generates final correlation heatmap.
**[PROCESS/CONTEXT]** Machine learning models require numeric inputs. The dropped columns are considered redundant because they have been replaced by engineered bins (`Age_band`, `Fare_cat`, `Initial`).
**[RESULT]** The dataset is ready for predictive modeling.
**[INSIGHT]** Dropping `Cabin` entirely due to NaNs might be a missed opportunity, as the *presence* of a cabin number could highly correlate with survival (1st class proxy).

### Stage 13: Predictive Modeling - Base Models
**[CODE]** Imports algorithms. Splits data using `train_test_split` with `stratify=data['Survived']`. Trains Linear SVM, RBF SVM, Logistic Regression, Decision Tree, KNN, Naive Bayes, and Random Forest. Evaluates via `metrics.accuracy_score`.
**[PROCESS/CONTEXT]** To get baseline accuracies for various algorithms.
**[RESULT]** The notebook outputs are empty, but based on typical runs, the accuracies are printed.
**[INSIGHT]** The author evaluates the models on a single 30% test set split. In the immediate next cell, they recognize that this causes "model variance" and correctly introduce Cross Validation.

### Stage 14: Cross Validation
**[CODE]** Uses `KFold(n_splits=10)` and `cross_val_score` to evaluate all seven baseline models. Stores results in a DataFrame and plots a boxplot and horizontal bar chart of the mean CV scores.
**[PROCESS/CONTEXT]** To achieve a generalized model evaluation irrespective of a single lucky/unlucky test split.
**[RESULT]** The markdown later reveals that the mean CV accuracy for RBF-SVM was 82.8%.
**[INSIGHT]** Implementing 10-fold CV proves robust methodology, ensuring the reported accuracies are reliable.

### Stage 15: Confusion Matrix
**[CODE]** Generates out-of-fold predictions using `cross_val_predict(cv=10)` for all seven models. Plots 3x3 subplots of seaborn heatmaps using `confusion_matrix`.
**[PROCESS/CONTEXT]** To see where the models go wrong (Type I vs Type II errors) because accuracy can be misleading.
**[RESULT]** For RBF-SVM, the correct predictions are 491 (dead) + 247 (survived). Errors are 58 (wrongly classified as survived) and 95 (wrongly classified as dead).
**[INSIGHT]** The author notes that RBF-SVM is better at predicting dead passengers, while Naive Bayes is better at predicting survivors. 

### Stage 16: Hyperparameter Tuning
**[CODE]** Uses `GridSearchCV` to tune `C` and `gamma` for SVM, and `n_estimators` for Random Forest.
**[PROCESS/CONTEXT]** To tune the "Black-Box" hyper-parameters for the top 2 algorithms.
**[RESULT]** The best score for Rbf-Svm is 82.82% with C=0.05 and gamma=0.1. For Random Forest, the score is about 81.8% with n_estimators=900.
**[INSIGHT]** The hyperparameter search grid is well-defined. However, testing `C=[0.05, 0.1, ... 1]` is somewhat narrow, as SVM tuning often requires exponential scales (e.g., 0.1, 1, 10, 100).

### Stage 17: Ensembling (Voting and Bagging)
**[CODE]** Implements `VotingClassifier(voting='soft')` combining all 7 models. Implements `BaggingClassifier` wrapping `KNeighborsClassifier` and `DecisionTreeClassifier`. Scores are evaluated using `cross_val_score(cv=10)`.
**[PROCESS/CONTEXT]** Ensembling improves stability and reduces variance.
**[RESULT]** No explicit scores are quoted in the text for voting and bagging, but they are executed correctly.
**[INSIGHT]** The author sets `probability=True` for the SVM estimators inside the VotingClassifier. This is structurally necessary for soft voting to work, demonstrating a deep understanding of scikit-learn's mechanics.

### Stage 18: Ensembling (Boosting)
**[CODE]** Implements `AdaBoostClassifier`, `GradientBoostingClassifier`, and `xgboost.XGBClassifier`. Tunes AdaBoost using `GridSearchCV` with `n_estimators` and `learning_rate`.
**[PROCESS/CONTEXT]** To use sequential learning of classifiers to fix previous misclassifications.
**[RESULT]** The maximum accuracy achieved is via AdaBoost: 83.16% with n_estimators=200 and learning_rate=0.05.
**[INSIGHT]** AdaBoost outperformed complex algorithms like XGBoost, highlighting that on small, clean, discrete datasets, simpler boosting algorithms can be optimal.

### Stage 19: Feature Importance
**[CODE]** Extracts `.feature_importances_` from Random Forest, AdaBoost, Gradient Boosting, and XGBoost, and plots them using horizontal bar charts.
**[PROCESS/CONTEXT]** To understand which features actually drove the predictions.
**[RESULT]** Initial, Fare_cat, Pclass, and Family_Size are consistently the most important.
**[INSIGHT]** The author is shocked that `Sex` has no importance in many models. However, this is observable in the code: the engineered `Initial` feature (Mr, Mrs, Miss) perfectly encodes gender. The tree models split on `Initial` first, rendering `Sex` completely redundant.

## 4. Cross-Cell Dependency Analysis

- **Sequential Execution Requirement:** The notebook is highly linear. Feature Engineering variables depend entirely on the EDA phase string manipulations. For instance, `data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33` in Cell 38 relies on the regex string extraction of `Initial` in Cell 31.
- **Overwriting State:** The `data` dataframe is mutated globally. Cell 91 drops 7 columns in-place (`inplace=True`). If the user attempts to re-run EDA cells after Cell 91, the notebook will crash because `Name`, `Age`, and `Fare` no longer exist.
- **Hardcoded Boundaries Leakage:** In Cell 87, the author hardcodes the `Fare_cat` boundaries (e.g., `data['Fare']<=7.91`) based on the visual output of the previous `qcut` cell. If this notebook is run on new data where the 25th percentile fare is different, the hardcoded boundaries will silently corrupt the quantile distribution.
- **Un-tuned Feature Importances:** In Cell 145, when plotting feature importances, the author instantiates brand new default models (`RandomForestClassifier(n_estimators=500)`) instead of using the best estimators identified by the `GridSearchCV` in earlier cells. Thus, the displayed importances do not perfectly match the behavior of the highest-scoring models.

## 5. Model Performance Summary

Based on the explicit text written by the author throughout the markdown blocks, the model performances are as follows:
- **RBF-SVM (Base CV Mean):** 82.8% (491 correct dead + 247 correct survived out of 891)
- **RBF-SVM (GridSearchCV Tuned):** 82.82% (with C=0.05, gamma=0.1)
- **Random Forest (GridSearchCV Tuned):** 81.8% (with n_estimators=900)
- **AdaBoost (GridSearchCV Tuned):** 83.16% (with n_estimators=200, learning_rate=0.05) - *Highest scoring model overall*

*Note: The actual terminal output for the scores is absent from the `.ipynb` file's JSON, meaning these values are extracted from the author's written interpretation.*

## 6. Conclusions and Recommendations

### Conclusions
The notebook is extremely methodical and provides a stellar introduction to predictive modeling. The progression from EDA to robust Cross-Validation and multiple Ensembling techniques is textbook. The author correctly identifies socioeconomic realities (Pclass) and demographic rules (Women/Children) using data, and translates those perfectly into engineered features.

### Recommendations (Observable Methodology Issues)
1. **Remove Hardcoded Bins to Prevent Leakage:** The manual assignment of bins via `.loc` using hardcoded threshold values (e.g. 7.91) based on training distributions is brittle. The author should use `pd.qcut(data['Fare'], 4, labels=[0,1,2,3])` directly, which allows the bin boundaries to be calculated algorithmically rather than manually typed.
2. **Drop Redundant Features to Prevent Multicollinearity:** The author notes in the feature importance section that `Sex` lost all its predictive value. This is because `Initial` already encodes gender perfectly. The `Sex` column should be added to the `.drop()` array during the data cleaning phase to reduce dimensionality.
3. **Avoid Destroying Continuous Variance for Tree Models:** The author bins `Age` and `Fare` into discrete categories, citing that continuous variables cause problems. While true for some linear models, the final best performing models are all Tree-based Ensembles (Random Forest, AdaBoost, XGBoost). These algorithms natively handle continuous variables perfectly and benefit from the granular variance. Binning `Fare` removes the distinction between an $80 ticket and a $500 ticket, both of which become category 3, destroying highly predictive outlier data.
4. **Use `best_estimator_` for Feature Importance:** Instead of defining new default models to extract `.feature_importances_`, the author should use `gd.best_estimator_.feature_importances_` to ensure the insights reflect the highest-scoring tuned models.

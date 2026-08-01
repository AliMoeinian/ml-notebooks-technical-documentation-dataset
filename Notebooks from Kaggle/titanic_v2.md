# Technical Documentation: Titanic Notebook (Version 2)

## 1. Notebook Overview
This technical documentation provides a comprehensive, cell-by-cell analysis of a Jupyter Notebook intended for binary classification on the Kaggle Titanic dataset. The primary objective of the notebook is to predict passenger survival based on various demographic and travel-related features. The workflow implemented by the author is a streamlined, relatively simple machine learning pipeline that covers data ingestion, exploratory data analysis (EDA), data cleaning, feature preprocessing, model training, evaluation, and finally, submission generation and model serialization. 

The notebook begins by importing necessary libraries and loading the training dataset. During the EDA and cleaning phase, the author makes significant simplifications by immediately discarding non-numeric text columns such as `Name`, `Ticket`, and `Cabin`, as well as the identifier `PassengerId`. Rows with missing values in the `Embarked` column are dropped, and histograms are generated to visualize the distributions of `Age` and `Fare`. 

In the preprocessing phase, categorical variables (`Sex` and `Embarked`) are encoded into numeric formats using a label encoder. Numeric features (`Age` and `Fare`) are scaled using a Min-Max scaler. Missing values in the dataset (primarily the `Age` column) are filled using a K-Nearest Neighbors (KNN) imputer. 

For the modeling phase, the author isolates the target variable (`Survived`) from the feature set and splits the data into training and testing subsets using an 80/20 ratio. Four distinct machine learning models are instantiated and trained: Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, and Extra Trees Classifier. Hyperparameters for the tree-based models are manually specified in the code. The models are evaluated based on their accuracy scores on both the training and testing sets, and a confusion matrix is plotted for the Extra Trees Classifier. A summary table is then generated to compare the performance of the four models. 

Finally, in the submission and serialization phase, the author processes a separate test dataset by replicating the preprocessing steps, generates survival predictions using the Random Forest Classifier, and saves these predictions to a CSV file for Kaggle submission. The notebook concludes by exporting the Gradient Boosting Classifier model to a file using the pickle module.

## 2. Environment and Dependencies
The notebook relies on a standard Python data science stack, importing libraries across various cells as needed for specific tasks. The dependencies observed directly in the code include:

- **pandas** (imported as `pd`): Used extensively for data manipulation, including loading CSV files (`pd.read_csv`), filtering data (`df.dropna`), dropping columns (`df.drop`), and creating tabular structures to summarize model results and format the final submission (`pd.DataFrame`).
- **matplotlib.pyplot** (imported as `plt`): Utilized for generating basic visualizations, specifically the histograms for the `Age` variable and setting up figure sizes and labels for seaborn plots.
- **numpy** (imported as `np`): Imported in the initial cell alongside pandas, although explicit usage of numpy functions is not prominent in the visible code.
- **seaborn** (imported as `sns`): Used for statistical data visualization, specifically to plot the histogram for the `Fare` variable (`sns.histplot`) and to create a heatmap visualization of the confusion matrix (`sns.heatmap`).
- **scikit-learn** (imported via specific modules):
  - `sklearn.preprocessing`: `LabelEncoder` is imported to transform categorical string variables into integers, and `MinMaxScaler` is used to scale numeric features into a specific range.
  - `sklearn.impute`: `KNNImputer` is imported to handle missing data by imputing values based on nearest neighbors.
  - `sklearn.model_selection`: `train_test_split` is imported to partition the data into training and evaluation sets.
  - `sklearn.linear_model`: `LogisticRegression` is imported as the first baseline classification model.
  - `sklearn.ensemble`: Three ensemble models are imported: `RandomForestClassifier`, `GradientBoostingClassifier`, and `ExtraTreesClassifier`.
  - `sklearn.metrics`: `confusion_matrix` is imported to evaluate the classification accuracy by comparing true versus predicted values.
- **pickle**: The standard Python object serialization library is imported in the final section to save the trained model object to disk (`pickle.dump`).

## 3. Per-Stage Documentation

### Exploratory Data Analysis (EDA) and Cleaning
**[CODE]**
Cell 1 begins with a markdown header `# EDA`. 
Cell 2 imports `pandas`, `matplotlib.pyplot`, `numpy`, and `seaborn`.
Cell 3 reads the data using `df = pd.read_csv("/kaggle/input/titanic/train.csv")` and displays the first few rows using `df.head()`. The output shows 12 columns: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.
Cell 4 executes `df = df.drop(['PassengerId','Name','Ticket','Cabin'] , axis='columns')`.
Cell 5 calls `df.info()`. The output shows a DataFrame with 891 entries and 8 columns. The `Survived`, `Pclass`, `SibSp`, and `Parch` columns are `int64` with 891 non-null values. `Sex` is `object` with 891 non-null values. `Fare` is `float64` with 891 non-null values. `Age` is `float64` with only 714 non-null values. `Embarked` is `object` with 889 non-null values.
Cell 6 drops rows with missing `Embarked` values using `df.dropna(subset=['Embarked'], inplace=True)`.
Cell 7 plots a histogram of the `Age` column using `plt.hist(df['Age'], bins = 50)` on a figure of size 8x6.
Cell 8 plots a histogram of the `Fare` column using `sns.histplot(data=df, x="Fare")`.

**[PROCESS/CONTEXT]**
The author initiates the project by loading the standard training dataset into a pandas DataFrame. To simplify the modeling process, the author immediately discards several columns that contain high-cardinality string data or unique identifiers (`PassengerId`, `Name`, `Ticket`, `Cabin`). After checking the dataset's composition and missing values via `df.info()`, the author notes that `Embarked` has only 2 missing values and removes those specific rows entirely. Finally, histograms are generated to inspect the distributions of the continuous numeric variables, `Age` and `Fare`.

**[RESULT]**
The dataset is reduced from 12 columns to 8 columns. The number of rows is reduced from 891 to 889 (due to dropping the two missing `Embarked` rows). The `df.info()` output explicitly confirms that missing values remain exclusively in the `Age` column (712 non-null out of 889). The plotted histograms provide a visual understanding of the data: the Age distribution appears roughly normal but slightly right-skewed, while the Fare distribution (generated by seaborn) is heavily right-skewed.

**[INSIGHT]**
The decision to drop the `Cabin` column is justified by the fact that it would require significant processing and likely contains many nulls (though not explicitly shown in `info()`, it is implicitly treated as non-essential). However, dropping the `Name` and `Ticket` columns removes potentially valuable information (such as passenger titles and family groupings). Furthermore, applying `dropna` for `Embarked` early in the pipeline permanently removes two training examples, which could have been imputed instead. 

### Data Preprocessing
**[CODE]**
Cell 9 contains a markdown header `# Data preprocessing`.
Cell 10 imports `LabelEncoder` and `MinMaxScaler`.
Cell 11 instantiates `encoder = LabelEncoder()` and applies it sequentially: `df['Sex'] = encoder.fit_transform(df['Sex'])` and `df['Embarked'] = encoder.fit_transform(df['Embarked'])`.
Cell 12 instantiates `scaler = MinMaxScaler()` and scales two columns: `df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])`.
Cell 13 contains a markdown header `# Replace NaN values in Age`.
Cell 14 stores the column names: `columns = df.columns`. The output is an Index array containing `['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`.
Cell 15 imports `KNNImputer`.
Cell 16 initializes the imputer with `imputer = KNNImputer(n_neighbors=5)` and applies it to the entire DataFrame: `df = imputer.fit_transform(df)`.
Cell 17 reconstructs the DataFrame from the resulting numpy array: `df = pd.DataFrame(df, columns= columns)`.
Cell 18 replots the `Age` histogram exactly as done in Cell 7.

**[PROCESS/CONTEXT]**
The author prepares the data for machine learning algorithms by ensuring all features are numeric and scaled. Categorical variables (`Sex` and `Embarked`) are mapped to integer values using `LabelEncoder`. The continuous variables (`Age` and `Fare`) are normalized to a range between 0 and 1 using `MinMaxScaler`. To resolve the missing values in the `Age` column, the author utilizes a K-Nearest Neighbors imputer, setting it to look at the 5 closest neighbors. Because scikit-learn transformers output numpy arrays, the author explicitly converts the imputed array back into a pandas DataFrame using the column names saved prior to imputation.

**[RESULT]**
All text-based data is converted to numeric formats. The `Age` and `Fare` features are bound within a [0, 1] range. The `KNNImputer` processes the dataset and fills the missing `Age` values. The final result is a complete, fully numeric DataFrame ready for modeling. The replotted `Age` histogram visually confirms that the missing values have been filled, altering the shape of the distribution.

**[INSIGHT]**
A critical instance of **data leakage** occurs in Cell 16. The author applies `KNNImputer.fit_transform(df)` to the *entire* DataFrame, which at this point still includes the target variable `Survived`. Consequently, the imputation of the missing `Age` values is being influenced by the passenger's survival status. In a real-world scenario, the survival status of an unknown passenger would not be available, making this imputation strategy invalid and overly optimistic. Additionally, applying `LabelEncoder` to `Embarked` is mathematically questionable, as it assigns ordinal integers to nominal categories (ports of embarkation), potentially leading linear models to misinterpret the relationship between ports.

### Model Training and Evaluation
**[CODE]**
Cell 19 contains the markdown `# ML Models`.
Cell 20 separates the features and target variable: `X = df.iloc[ : , 1:]` (taking columns index 1 through 7) and `y = df.iloc[ : , 0]` (taking the `Survived` column).
Cell 21 imports `train_test_split`, `confusion_matrix`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, and `ExtraTreesClassifier`.
Cell 22 splits the data: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)`. No random state is provided.
Cells 23-24 train a Logistic Regression model: `logreg.fit(X_train, y_train)`. It calculates training and testing accuracy, outputting exactly: `Train: 81.435`, `Test: 83.146`.
Cells 25-26 instantiate and train a Random Forest model with specific parameters (`max_features=.7`, `min_samples_leaf=10`, `min_samples_split=8`, `n_estimators=10`, `max_depth=15`, `verbose=1`). Outputs exactly: `Train: 84.951`, `Test: 85.393`.
Cells 27-28 instantiate and train a Gradient Boosting model (`learning_rate=0.01`, `max_depth=10`, `max_features=0.8`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=100`, `subsample=0.4`, `random_state=42`). Outputs exactly: `Train: 90.577`, `Test: 87.079`.
Cells 29-30 instantiate and train an Extra Trees model (`bootstrap=True`, `criterion='gini'`, `max_features=0.65`, `min_samples_leaf=1`, `min_samples_split=6`, `n_estimators=100`, `random_state=42`). Outputs exactly: `Train: 90.999`, `Test: 87.079`.
Cells 31-32 generate predictions using the `ExtraTreesClassifier` on the test set and plot a confusion matrix using `sns.heatmap(cm, annot=True,cmap='Blues')`. The plot axes are labeled 'Predicted' and 'Truth'.
Cell 33 compiles the scores into a pandas DataFrame `models` with columns `Model`, `Training_score`, and `Testing_score`, and sorts them descending by `Testing_score`. The output shows ExtraTrees and GradientBoosting tied at 87.079 testing score, followed by Random Forest at 85.393, and Logistic Regression at 83.146.

**[PROCESS/CONTEXT]**
The author sets up the supervised learning task by explicitly slicing the DataFrame to isolate the features (`X`) from the target (`y`). The data is then randomly partitioned into an 80% training set and a 20% testing set to allow for unbiased evaluation. Four different classification algorithms are evaluated. For the tree-based ensemble methods, the author manually defines numerous hyperparameters (such as depths, minimum samples, and estimators) indicating an attempt at model tuning, though the process of how these parameters were selected is not shown. The effectiveness of the models is measured by calculating accuracy on both the train and test splits, and a confusion matrix is visualized for the Extra Trees model to show the distribution of true positives, false positives, true negatives, and false negatives.

**[RESULT]**
All four models are successfully trained and evaluated. The Logistic Regression and Random Forest models show slightly higher testing accuracy than training accuracy, which is somewhat unusual but possible given a specific random split of the data. The Gradient Boosting and Extra Trees classifiers achieve much higher training accuracy (~90-91%) and identical test accuracy (87.079%). The summary DataFrame provides a clean comparison of the performance metrics.

**[INSIGHT]**
The absence of a `random_state` parameter in the `train_test_split` function means that the data partition will be different every time the notebook is run, rendering the exact accuracy scores irreproducible. Interestingly, both the Gradient Boosting Classifier and the Extra Trees Classifier achieve exactly 87.079% accuracy on the test set. Because both models utilize `random_state=42` and the test split is identical for both within the same runtime session, they happen to make the exact same number of correct predictions, although their underlying logic and training accuracies differ.

### Submission and Serialization
**[CODE]**
Cell 35 contains the markdown `# Submission`.
Cell 36 executes a large block of code for processing the test dataset:
```python
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test = test_data.drop(['PassengerId','Name','Ticket','Cabin'] , axis='columns')
test['Sex'] = encoder.fit_transform(test['Sex'])
test['Embarked'] = encoder.fit_transform(test['Embarked'])
imputer = KNNImputer(n_neighbors=2)
test = imputer.fit_transform(test)
test = pd.DataFrame(test, columns=columns[1:])
predictions = RandomForestClassifier.predict(test)
predictions = predictions.astype(int)
```
Cell 37 constructs the submission DataFrame: `output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})` and saves it using `output.to_csv('/kaggle/working/submission.csv', index=False)`.
Cell 38 contains the markdown `# Save the model`.
Cell 39 imports `pickle`.
Cell 40 saves the model using `with open('Titanic_model','wb') as file: pickle.dump(GradientBoostingClassifier,file)`.

**[PROCESS/CONTEXT]**
To generate predictions for the Kaggle competition, the author loads the unlabelled `test.csv` file and attempts to replicate the preprocessing steps applied to the training data. The same columns are dropped, the data is label-encoded, and missing values are imputed. The reconstructed test DataFrame is then passed into the previously trained Random Forest model to generate survival predictions. These predictions are combined with the `PassengerId` column and exported to a CSV file. Finally, the author uses the pickle library to serialize a trained model to a file named `Titanic_model`, allowing the model to be loaded and used in external applications without retraining.

**[RESULT]**
The code generates an output file `submission.csv` containing binary survival predictions for the test dataset. A binary model file `Titanic_model` is also successfully created and stored on the local disk.

**[INSIGHT]**
This section contains multiple severe logical errors and instances of data leakage that invalidate the machine learning pipeline:
1. **Incorrect Method Usage:** The author uses `.fit_transform()` on the test dataset for both the `LabelEncoder` and the `KNNImputer`. In machine learning, transformers must only be fitted on the training data. Using `.fit_transform()` on the test set causes the encoders and imputers to learn new parameters from the test data itself, which breaks the consistency of the feature space and introduces data leakage.
2. **Inconsistent Imputer Parameters:** The author instantiates a brand new `KNNImputer` for the test set with `n_neighbors=2`, whereas the training set was imputed using `n_neighbors=5`. This means the methodology for handling missing data is fundamentally different between the training and testing phases.
3. **Model Selection Discrepancy:** In Cell 33, the author's summary table clearly shows that the `ExtraTreesClassifier` and `GradientBoostingClassifier` achieved the highest testing scores (87.079%), while the `RandomForestClassifier` scored significantly lower (85.393%). Despite this, the author explicitly uses `RandomForestClassifier.predict(test)` to generate the final competition predictions.
4. **Serialization Mismatch:** The author uses the Random Forest model to create the `submission.csv` file, but in Cell 40, they execute `pickle.dump(GradientBoostingClassifier,file)`. The model saved for future use is completely different from the model used to generate the submitted predictions.

## 4. Cross-Cell Dependency Analysis
- **Column Name Tracking (`columns`):** In Cell 14, the author saves the columns of the training DataFrame (`['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`) into a variable named `columns`. In Cell 17, this variable is used to rebuild the training DataFrame. In Cell 36, the author uses `columns[1:]` to rebuild the test DataFrame. Because the test DataFrame naturally lacks the `Survived` column (which was at index 0), `columns[1:]` perfectly isolates the 7 feature names and successfully applies them to the test set. This is a functional, albeit brittle, dependency across the notebook.
- **Encoder Object Overwriting (`encoder`):** The `LabelEncoder` instantiated in Cell 11 is reused in Cell 36. However, because the author uses `encoder.fit_transform()` on the test set, the internal state (the mapping dictionary) of the encoder learned from the training data is overwritten by the test data. This cross-cell interaction leads directly to data inconsistency.
- **Imputer Redefinition:** The `KNNImputer` created in Cell 16 is completely discarded and overwritten by a new `KNNImputer(n_neighbors=2)` in Cell 36, breaking the continuity of the preprocessing pipeline.

## 5. Model Performance Summary
The notebook evaluates four separate classifiers. Based strictly on the printed output generated by the notebook's evaluation cells, the performance is as follows:

| Model | Training Score | Testing Score |
| :--- | :---: | :---: |
| ExtraTreesClassifier | 90.999% | 87.079% |
| GradientBoostingClassifier | 90.577% | 87.079% |
| RandomForestClassifier | 84.951% | 85.393% |
| Logistic Regression | 81.435% | 83.146% |

**Performance Analysis:**
The Extra Trees and Gradient Boosting classifiers demonstrate a high capacity to fit the training data (over 90%) and generalize well to the test data (87.079%). The Random Forest and Logistic Regression models exhibit lower overall accuracy, but interestingly achieve higher scores on the test set than on the training set. This anomaly is likely a byproduct of the random split lacking a set `random_state`, resulting in a test set that happened to be slightly "easier" to predict for those specific model architectures.

## 6. Conclusions and Recommendations
This notebook establishes a functional, end-to-end baseline pipeline for binary classification on the Titanic dataset. It correctly imports necessary libraries, handles missing data, scales features, and evaluates multiple ensemble models. The manual hyperparameter tuning of the ensemble models demonstrates an effort to optimize performance.

However, the implementation is heavily compromised by fundamental methodological errors related to data leakage, test set processing, and pipeline consistency. 

**Recommendations for Remediation:**
1. **Prevent Target Leakage During Imputation:** The target variable `Survived` must be isolated from the feature set `X` *before* the `KNNImputer` is applied. Imputing features using the target variable invalidates the predictive integrity of the models.
2. **Correct Application of Transformers to Test Data:** In Cell 36, all instances of `.fit_transform()` must be changed to `.transform()`. The test dataset must be transformed using the exact `LabelEncoder`, `MinMaxScaler`, and `KNNImputer` objects that were fitted on the training data.
3. **Standardize Imputation Parameters:** Remove the re-instantiation of the `KNNImputer` in Cell 36. Use the original imputer trained with `n_neighbors=5` to transform the test set, ensuring that missing values are handled identically across all datasets.
4. **Align Predictions and Serialization:** Ensure logical consistency in the final stages. If the `GradientBoostingClassifier` is identified as the best model and serialized via `pickle`, it should also be the model used to generate `predictions` for `submission.csv`. Currently, the notebook predicts with one model and saves another.
5. **Set Random Seeds for Reproducibility:** Add a `random_state` argument to the `train_test_split` function, as well as to the Logistic Regression and Random Forest models, to ensure that the exact splits and accuracy scores can be reproduced in future runs.

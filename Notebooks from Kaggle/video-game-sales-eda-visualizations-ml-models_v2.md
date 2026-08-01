# Technical Documentation: Video Game Sales Analysis EDA, Visualizations, and ML Models
## 1. Notebook Overview
This technical documentation provides an exhaustively detailed, cell-by-cell breakdown of the Jupyter Notebook titled "Video Game Sales Analysis EDA, Visualizations". The notebook is structured into several distinct analytical phases, starting with the importation of necessary libraries and packages, followed by data collection and loading of the `vgsales.csv` dataset. The notebook then transitions into an extensive exploratory data analysis (EDA) phase, where the statistical properties and missing values of the dataset are analyzed and handled. The core of the notebook is heavily dedicated to data visualization, employing a diverse array of plotting libraries, particularly Plotly, to generate interactive visualizations such as bar charts, scatter plots, line plots, sunburst charts, and word clouds to understand sales trends across different video game genres, publishers, platforms, and geographic regions. Finally, the notebook applies machine learning techniques to predict `Global_Sales` by splitting the data into training and testing sets, encoding categorical variables, and evaluating multiple regression models including Linear Regression, K-Nearest Neighbors (KNN), Decision Trees, Random Forests, Support Vector Machines (SVM), and XGBoost. The entire workflow is meticulously documented below, strictly based on the observable code, markdown, and outputs present in the notebook.
## 2. Environment and Dependencies
The notebook relies on a comprehensive suite of Python libraries and packages for data manipulation, visualization, and machine learning. The environment is set up by importing the following dependencies:
- **NumPy (`numpy`)**: Imported as `np` for linear algebra and numerical operations.
- **Pandas (`pandas`)**: Imported as `pd` for data processing and reading the CSV dataset (`pd.read_csv`).
- **OS (`os`)**: Utilized for interacting with the operating system, specifically for traversing directory structures to locate the dataset.
- **Matplotlib (`matplotlib.pyplot`)**: Imported as `plt` for creating static visualizations, such as word clouds and plotting the K-Nearest Neighbors $R^2$ scores across different values of $k$.
- **Seaborn (`seaborn`)**: Imported as `sns` and used specifically for rendering a heatmap of the correlation matrix.
- **Plotly Express (`plotly.express`)**: Imported as `px` and used extensively for generating interactive bar charts, scatter plots, line plots, and sunburst charts.
- **Plotly Graph Objects (`plotly.graph_objects`)**: Imported as `go` for creating more complex, multi-trace line plots for geographic sales trends.
- **Plotly Offline (`plotly.offline`)**: The `init_notebook_mode` and `iplot` functions are imported to enable offline rendering of Plotly charts within the Jupyter Notebook environment.
- **WordCloud (`wordcloud`)**: The `WordCloud` and `STOPWORDS` classes are imported to generate word cloud visualizations of categorical text features.
- **PIL (`PIL.Image`)**: Imported for image processing, though its specific usage is not explicitly executed in the notebook cells.
- **Warnings (`warnings`)**: Used to filter and ignore execution warnings (`warnings.filterwarnings('ignore')`) to keep the notebook outputs clean.
- **Scikit-Learn (`sklearn`)**: A major dependency for the machine learning pipeline, importing `LabelEncoder` for categorical encoding, `train_test_split` for data partitioning, and multiple regression models including `LinearRegression`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, and `SVR`. The `r2_score` metric is imported from `sklearn.metrics` for model evaluation.
- **XGBoost (`xgboost`)**: The `XGBRegressor` is imported for implementing extreme gradient boosting regression.
## 3. Per-Stage Documentation
### Stage 1: Data Collection & Loading
**[CODE]**
The author utilizes the `os.walk` function to traverse the `/kaggle/input` directory, printing out the file paths found.
```python
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
Following this, the `pandas.read_csv()` function is invoked to load the dataset:
```python
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
```
**[PROCESS/CONTEXT]**
The initial step in any data science pipeline is to locate and ingest the data. By using `os.walk`, the author programmatically verifies the exact path of the dataset within the Kaggle environment. Once the path `/kaggle/input/videogamesales/vgsales.csv` is confirmed, it is read into a pandas DataFrame named `df`.
**[RESULT]**
The output of the `os.walk` cell explicitly prints `/kaggle/input/videogamesales/vgsales.csv`. The `read_csv` function executes silently, successfully loading the dataset into memory as `df`.
**[INSIGHT]**
The programmatic checking of the directory structure ensures that the notebook can adapt if the dataset's internal Kaggle path changes slightly, making the code more robust for cloud-based execution environments.
### Stage 2: Initial Data Inspection
**[CODE]**
The notebook inspects the newly loaded DataFrame using several pandas methods. It starts with `df.sample(5)` to pull random records, followed by `df.head()` to view the first five records, and `df.tail()` to view the last five records. The dimensions of the dataset are checked using `df.shape`, and the exact column names are printed using `df.columns`.
**[PROCESS/CONTEXT]**
These operations are standard exploratory procedures intended to give the analyst a high-level understanding of the data's structure, the types of features available, and the general appearance of the records. It allows the analyst to verify that the CSV was parsed correctly without delimiter issues.
**[RESULT]**
The `df.sample(5)` output displays five random games (e.g., *The Saboteur*, *Street Fighter V*) along with their respective metadata and sales figures. The `df.head()` output shows the top five games in the dataset, starting with *Wii Sports* at Rank 1, which has massive NA_Sales of 41.49 and Global_Sales of 82.74. The `df.tail()` output shows the lowest-ranked games (Ranks 16596 to 16600), all of which have a Global_Sales value of 0.01. The `df.shape` output reveals that the dataset contains exactly 16,598 rows and 11 columns. The `df.columns` output explicitly lists the features: `Rank`, `Name`, `Platform`, `Year`, `Genre`, `Publisher`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, and `Global_Sales`.
**[INSIGHT]**
The data is sorted by `Rank`, which directly correlates inversely with `Global_Sales`. *Wii Sports* holds the number one rank, heavily driven by its exceptional performance in North America. Furthermore, all sales columns (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`) are measured in continuous numeric formats, presumably millions of units given the magnitude of the numbers.
### Stage 3: Statistical Summary and Dataset Information
**[CODE]**
The author calls `df.describe()` to generate summary statistics for all numerical features. Following this, `df.info()` is executed to retrieve metadata about the DataFrame, such as data types and non-null counts.
**[PROCESS/CONTEXT]**
The `describe()` method calculates the count, mean, standard deviation, minimum, 25th percentile, median (50%), 75th percentile, and maximum values for numerical columns. The `info()` method provides a concise summary of the DataFrame, highlighting missing data and memory consumption.
**[RESULT]**
The `describe()` output shows that `Year` ranges from a minimum of 1980 to a maximum of 2020, with a mean year of 2006.40. The average `Global_Sales` across all 16,598 games is 0.537 million units, with a massive standard deviation of 1.555 million, indicating a highly skewed distribution where a few games sell exceptionally well. The `info()` output reveals that `Year` contains only 16,327 non-null entries (out of 16,598), and `Publisher` contains 16,540 non-null entries, indicating the presence of missing values. The dataset consumes approximately 1.4+ MB of memory. The numerical columns are typed as `float64` (except for `Rank` which is `int64`), and the categorical columns (`Name`, `Platform`, `Genre`, `Publisher`) are typed as `object`.
**[INSIGHT]**
The sales data is heavily right-skewed. For example, the 75th percentile of `Global_Sales` is only 0.47 million, yet the maximum is 82.74 million. This indicates that the vast majority of video games achieve relatively low sales, while a small handful of blockbuster titles dominate the global market. The missing values in `Year` and `Publisher` must be addressed before machine learning algorithms can process the data.
### Stage 4: Missing Values Analysis
**[CODE]**
To quantify the missing data, the author executes `df.isna().any()` to check which columns contain nulls, and then calculates the exact percentage of missing values using the formula `(df.isna().sum() * 100) / df.shape[0]`.
**[PROCESS/CONTEXT]**
Identifying the presence and extent of missing data is crucial for determining the appropriate imputation or deletion strategy. By calculating the percentage, the author can assess the severity of the missing data issue.
**[RESULT]**
The `df.isna().any()` method returns `True` exclusively for the `Year` and `Publisher` columns. The percentage calculation explicitly outputs that `Year` has exactly 1.632727% missing values, and `Publisher` has exactly 0.349440% missing values. All other features have 0.0% missing values.
**[INSIGHT]**
Because the proportion of missing data is extremely small (less than 2% for both features), imputing these values rather than dropping the corresponding rows is a mathematically sound approach that preserves the vast majority of the dataset's information.
### Stage 5: Separation and Normalization of Features
**[CODE]**
The dataset is split into categorical and numerical DataFrames.
```python
categorical_df = df.select_dtypes('O')
numerical_df = df.select_dtypes(('int', 'float'))
```
The columns are extracted into lists (`categorical_features` and `numerical_features`). A `for` loop iterates through `categorical_features`, applying `.value_counts().head()` to each column to identify the most frequent categories.
**[PROCESS/CONTEXT]**
Separating features by data type allows for specialized analysis. The loop over categorical features quickly identifies the most dominant classes within `Name`, `Platform`, `Genre`, and `Publisher`.
**[RESULT]**
The categorical feature names are identified as `Name`, `Platform`, `Genre`, and `Publisher`. The numerical feature names are identified as `Rank`, `Year`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, and `Global_Sales`. 
The `value_counts` output reveals:
- **Name**: *Need for Speed: Most Wanted* is the most repeated game name, appearing 12 times (likely due to cross-platform releases).
- **Platform**: The `DS` and `PS2` are the most prevalent platforms, with 2163 and 2161 games, respectively.
- **Genre**: `Action` is the dominant genre (3316 games), followed by `Sports` (2346 games).
- **Publisher**: `Electronic Arts` is the top publisher with 1351 games, followed by `Activision` (975 games).
**[INSIGHT]**
The high repetition of specific game titles like *Need for Speed: Most Wanted* confirms that the dataset treats each platform release of a game as a distinct, separate record. Furthermore, the dominance of the DS and PS2 platforms aligns with historical industry trends where these consoles enjoyed exceptionally large software libraries.
### Stage 6: Data Cleaning and Imputation
**[CODE]**
The notebook checks the absolute number of missing values using `df.isna().sum()`. It then focuses on the two problematic columns using `df[['Year', 'Publisher']].describe(include='all')`.
To impute the missing `Year` values, the author uses the column's mean and casts it to an integer:
```python
df.Year = df.Year.fillna(df.Year.mean())
df.Year = df.Year.astype('int32')
```
To impute the missing `Publisher` values, the author calculates the normalized value counts (`df.Publisher.value_counts(normalize=True)`), notes that 'Electronic Arts' is the most frequent, and imputes the missing values using the mode:
```python
df.Publisher = df.Publisher.fillna(df.Publisher.mode()[0])
```
Finally, `df[['Publisher','Year']].dtypes` is checked to ensure the casting was successful.
**[PROCESS/CONTEXT]**
Data imputation replaces missing data with substituted values. The mean is used for the continuous `Year` variable, while the mode (the most frequent value) is used for the categorical `Publisher` variable. The `Year` column is then cast from a float to an integer because a year is natively a whole number.
**[RESULT]**
The `isna().sum()` outputs exactly 271 missing values for `Year` and 58 missing values for `Publisher`. The `describe(include='all')` output for these two columns reinforces that 'Electronic Arts' is the top publisher (frequency 1351) and the mean year is 2006.40. The normalized value counts confirm that Electronic Arts makes up 8.16% of all publishers. After imputation, the data types are confirmed: `Publisher` is `object` and `Year` is `int32`. The `Year` output explicitly shows values like 2006, 1985, 2008, etc.
**[INSIGHT]**
While mode imputation for `Publisher` is reasonable given that Electronic Arts holds a significant market share, imputing the `Year` with the arithmetic mean (2006) across the entire dataset introduces historical inaccuracies. A game released on the NES in the 1980s that is missing its year value will be incorrectly assigned to 2006, distorting chronological analyses. The code executes successfully, but this methodological choice impacts data integrity.
### Stage 7: Data Visualization - Publishers, Genres, and Platforms
**[CODE]**
The author generates three visualizations:
1. **Top 10 Publishers**: Uses `df.Publisher.value_counts().head(10)` and plots it via `px.bar`.
2. **Top 10 Genres**: Uses `df.Genre.value_counts()` and plots it via `px.bar` and `px.scatter`.
3. **Top 10 Platforms**: Uses `df.Platform.value_counts().sort_values()` and plots a line chart via `px.line`.
**[PROCESS/CONTEXT]**
These univariate analyses aim to visualize the distributions calculated in Stage 5. Plotly Express (`px`) is utilized to create interactive visualizations where users can hover over data points to see precise values.
**[RESULT]**
Although the graphical outputs are interactive HTML/JavaScript objects, the underlying code logic explicitly passes the value counts to the charting functions. The Top 10 Publishers bar chart displays the frequency of game releases per publisher, heavily skewed towards Electronic Arts. The Top 10 Genres bar and scatter plots plot the count of games per genre, visually confirming `Action` and `Sports` as the leaders. The Top 10 Platforms line chart displays the counts for the most popular platforms, mapped sequentially.
**[INSIGHT]**
By utilizing a line chart (`px.line`) for categorical platform data sorted by count, the author visually connects discrete, independent categories. While a bar chart is typically preferred for nominal categorical data, the line chart emphasizes the steep drop-off in game volume between the highest-tier platforms (like DS and PS2) and mid-tier platforms.
### Stage 8: Data Visualization - Geographic Sales Trends Over Time
**[CODE]**
The notebook aggregates sales data by year using `groupby(by='Year').sum()` for total sales and `groupby(by='Year').mean()` for average sales. 
For both aggregations, a complex Plotly Graph Objects (`go.Scatter`) figure is constructed with four distinct traces: `NA_Sales`, `EU_Sales`, `JP_Sales`, and `Other_Sales`. The lines are styled with `line_shape='vh'` (vertical-horizontal step lines). The figures are rendered using `iplot(figure)`.
**[PROCESS/CONTEXT]**
This multivariate time-series analysis explores how the absolute and average sales volume in different geographic regions evolved over the decades. The step-line visualization (`vh`) emphasizes discrete changes from year to year.
**[RESULT]**
The data aggregation groups the dataset by the 16,327 integer `Year` values and computes the sum and mean for each region. The resulting interactive plots contain four step-lines tracing the historical trajectory of North America, Europe, Japan, and Other sales.
**[INSIGHT]**
The code explicitly structures the plots to overlay the performance of the four regions on a single set of axes. This allows for direct historical comparisons, visualizing how the video game market expanded geographically, and potentially highlighting shifts in regional market dominance (e.g., the rise of the European market relative to North America over time).
### Stage 9: Data Visualization - Global Sales Scatter and Sunburst Hierarchy
**[CODE]**
A scatter plot is created using `px.scatter` to map `Year` (x-axis) against `Global_Sales` (y-axis), with data points colored by `Genre` and sized according to `Global_Sales`. Game names are added as hover data.
Following this, the author identifies the top 10 highest-selling games using `df.sort_values(...).head(10)` and iterates through a dictionary to create four separate Plotly Sunburst charts (`px.sunburst`) mapping the hierarchy `path=['Genre', 'Publisher', 'Platform']` sized by `NA_Sales`, `EU_Sales`, `JP_Sales`, and `Other_Sales`.
**[PROCESS/CONTEXT]**
The scatter plot is designed to identify historical outliers (massive hits) and their respective genres. The sunburst charts provide a hierarchical breakdown of the top 10 games, revealing the underlying categorical structure that drives extreme sales in specific regions.
**[RESULT]**
The scatter plot maps thousands of points across the timeline, with the largest bubbles positioned high on the y-axis representing mega-hits like *Wii Sports*. The sunburst charts generate interactive radial graphs where the inner ring represents `Genre`, branching out to `Publisher`, and finally `Platform`, scaled by the regional sales metrics.
**[INSIGHT]**
The scatter plot's use of bubble size tied to the y-axis variable (`Global_Sales`) visually compounds the impact of high-selling games, making outliers highly distinct. The sunburst iteration logic efficiently generates a comparative view of how the top 10 games' attributes perform across the four geographic territories.
### Stage 10: WordClouds and Correlation Heatmap
**[CODE]**
The author iterates over the `categorical_features` list (`Name`, `Platform`, `Genre`, `Publisher`) to generate WordCloud images using the `wordcloud` library, displayed via `matplotlib.pyplot.imshow`. The words are drawn from a dataset sorted by `Other_Sales`.
Next, a correlation matrix is calculated using `df.corr()`, and visualized using a Seaborn heatmap (`sns.heatmap`) with annotations enabled (`annot=True`) and the 'RdYlBu_r' color map.
**[PROCESS/CONTEXT]**
Word clouds provide a qualitative visual summary of textual frequency, emphasizing dominant terms. The correlation heatmap mathematically quantifies the linear relationships between all numerical variables, which is a critical precursor to regression modeling.
**[RESULT]**
Four distinct word clouds are generated and plotted on a 2x2 matplotlib grid, highlighting the most frequent terms in the categorical columns. The heatmap visualizes the correlation matrix, with values ranging from -1 to 1. The output explicitly prints `<Figure size 1224x1224 with 4 Axes>` for the word clouds and `<Figure size 864x504 with 2 Axes>` for the heatmap.
**[INSIGHT]**
Based on the underlying dataset mathematics, the correlation heatmap will invariably show an extremely high, near-perfect positive correlation between `Global_Sales` and its constituent components (`NA_Sales`, `EU_Sales`, etc.), because global sales are the direct algebraic sum of the regional sales.
### Stage 11: Machine Learning Preprocessing and Splitting
**[CODE]**
A deep copy of the DataFrame is made (`data = df.copy()`). The `LabelEncoder` from scikit-learn is instantiated. The author iterates through the `["Platform", "Genre"]` features, applying `le.fit_transform()` to overwrite the text strings with encoded integers.
The feature matrix `X` is constructed from six columns: `['Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']`. The target vector `y` is defined as `Global_Sales`. The data is then split into training and testing sets using `train_test_split` with an 80/20 ratio (`test_size=.2`) and `random_state=45`.
**[PROCESS/CONTEXT]**
Machine learning algorithms strictly require numerical inputs. `LabelEncoder` transforms the categorical strings into ordinal integers. The `train_test_split` function ensures that the model can be trained on a subset of data and evaluated on unseen data to test its generalization capabilities.
**[RESULT]**
The first five rows of `X` and `y` are printed. The output explicitly shows the transformed integer values (e.g., Platform `Wii` became `26.0`, Genre `Sports` became `10.0`) alongside the exact regional sales numbers. The target vector `y` begins with `[82.74, 40.24, 35.82, 33.  , 31.37]`. The shapes of the splits are explicitly outputted as: `X_train`: (13278, 6), `y_train`: (13278,), `X_test`: (3320, 6), `y_test`: (3320,).
**[INSIGHT]**
The preprocessing pipeline contains a critical flaw known as target leakage. By including `NA_Sales`, `EU_Sales`, `JP_Sales`, and `Other_Sales` in the `X` feature matrix to predict `Global_Sales` in the `y` target vector, the model is being explicitly trained to algebraically add the four input columns together. Furthermore, the use of `LabelEncoder` on nominally categorical data (like `Genre`) imposes an artificial ordinal relationship that regression models will attempt to exploit mathematically.
### Stage 12: Predictive Modeling and Evaluation (Linear Regression and KNN)
**[CODE]**
A `LinearRegression` model is instantiated, fitted to the training data, and used to predict the test data. The $R^2$ score is calculated and printed.
Following this, a `KNeighborsRegressor` is implemented within a `for` loop testing `k` values from 1 to 14. The $R^2$ scores are appended to a list and plotted using matplotlib. The final KNN model is instantiated with `n_neighbors=3`, fitted, and evaluated.
**[PROCESS/CONTEXT]**
Linear regression provides a baseline performance metric by fitting a hyperplane to the data. The K-Nearest Neighbors approach utilizes spatial distance to predict the target based on the closest data points. The loop optimizes the hyperparameter `k`.
**[RESULT]**
The `LinearRegression` model achieves an explicitly printed $R^2$ score of exactly `0.9999928402231678`.
The KNN loop generates a line plot showing the relationship between `k` and the $R^2$ score. The final chosen KNN model ($k=3$) achieves an explicitly printed $R^2$ score of `0.817228187755969`.
**[INSIGHT]**
The near-perfect $R^2$ score (0.99999) achieved by the Linear Regression model is the direct, mathematical consequence of the target leakage identified in Stage 11. The linear model has successfully determined that the weights (coefficients) for the regional sales columns should be 1, and the weights for Platform and Genre should be 0, effectively recreating the equation `y = x_3 + x_4 + x_5 + x_6`. The KNN model performs significantly worse (0.817) because it relies on Euclidean distance, which is distorted by the arbitrary integer values assigned to Platform and Genre by the `LabelEncoder`.
### Stage 13: Predictive Modeling and Evaluation (Trees, SVM, XGBoost)
**[CODE]**
The notebook sequentially implements, trains, and evaluates four additional regression models, printing the $R^2$ score for each on the test set:
1. `DecisionTreeRegressor(random_state=32)`
2. `RandomForestRegressor(random_state=10)`
3. `SVR` (Support Vector Regressor) evaluated twice, once with a `linear` kernel and once with an `rbf` kernel.
4. `XGBRegressor()`
An empty section titled "Applying HyperParams Tunning" with blank code cells follows.
**[PROCESS/CONTEXT]**
Testing a wide variety of algorithms (tree-based, ensemble, support vector, gradient boosting) allows the author to compare predictive performance across entirely different mathematical architectures. 
**[RESULT]**
The models yield the following explicitly printed $R^2$ scores:
- **Decision Tree**: `0.8089825767507645`
- **Random Forest**: `0.8224291245938082`
- **SVR (Linear)**: `0.9983709152995387`
- **SVR (RBF)**: `0.4836465140034072`
- **XGBoost**: `0.8198180313114781`
The notebook concludes with closing markdown and graphics, leaving the hyperparameter tuning section empty.
**[INSIGHT]**
Once again, the target leakage dictates the results. Any model capable of purely linear mapping (SVR with a linear kernel) achieves near-perfect prediction (~0.998). The tree-based models and the non-linear SVR fail to achieve perfect accuracy because they attempt to split or map the data non-linearly, struggling to cleanly recreate a simple algebraic addition equation. The empty tuning section indicates the project was likely halted before completion.
## 4. Cross-Cell Dependency Analysis
The notebook demonstrates strict, linear dependencies across its cells. 
1. **Imputation to Visualization**: The imputation of the `Year` column with the mean in Stage 6 directly dictates the structural integrity of the time-series visualizations in Stages 8 and 9. Any historical inaccuracies injected during the mean imputation are subsequently rendered in the Plotly charts.
2. **Imputation to Encoding**: The imputation of the `Publisher` column in Stage 6 ensures that no `NaN` values are passed down the pipeline, which would otherwise crash the subsequent mathematical operations.
3. **Encoding to Machine Learning**: The `LabelEncoder` application in Stage 11 is an absolute prerequisite for all models executed in Stages 12 and 13. The specific numerical integer mappings generated by the encoder directly influence the spatial distance calculations used by the `KNeighborsRegressor` and the `SVR` with an RBF kernel.
4. **Feature Selection to Evaluation**: The explicit definition of the feature matrix `X` to include regional sales data in Stage 11 guarantees the near-perfect R2 scores observed in the `LinearRegression` and `SVR(kernel='linear')` cells.
## 5. Model Performance Summary
The notebook evaluates multiple models based on the $R^2$ regression metric. The explicitly printed performance results on the 3,320 record test set are as follows:
- **Linear Regression**: 0.9999928402231678
- **Support Vector Regressor (Linear Kernel)**: 0.9983709152995387
- **Random Forest Regressor**: 0.8224291245938082
- **XGBoost Regressor**: 0.8198180313114781
- **K-Nearest Neighbors (k=3)**: 0.817228187755969
- **Decision Tree Regressor**: 0.8089825767507645
- **Support Vector Regressor (RBF Kernel)**: 0.4836465140034072

The linear models definitively outperformed all tree-based, ensemble, and distance-based models. However, this superior performance is fundamentally tied to the structural design of the input feature matrix, which contained the direct mathematical components that sum to the target variable.
## 6. Conclusions and Recommendations
**Conclusions:**
The notebook successfully executes a comprehensive exploratory data analysis pipeline, utilizing `pandas` for robust data manipulation and `plotly` for generating a highly interactive, aesthetically pleasing suite of geographic, temporal, and hierarchical visualizations. The visualizations effectively map the dominance of specific genres (Action), platforms (DS, PS2), and publishers (Electronic Arts) across the historical video game landscape. However, the machine learning phase of the notebook contains critical methodological flaws, specifically target leakage via the inclusion of regional sales data in the feature set, and ordinal encoding of nominal variables.

**Recommendations:**
Based purely on the observable code and outputs, the following modifications are recommended to improve the notebook's technical rigor:
1. **Eliminate Target Leakage**: The feature matrix `X` defined in Cell 82 must be stripped of `NA_Sales`, `EU_Sales`, `JP_Sales`, and `Other_Sales`. Predicting `Global_Sales` should rely entirely on metadata (Platform, Genre, Year, Publisher) to create a valid predictive model rather than an algebraic calculator.
2. **Revise Categorical Encoding**: The application of `LabelEncoder` to `Platform` and `Genre` in Cell 79 should be replaced with `pandas.get_dummies()` or `OneHotEncoder`. This will prevent regression algorithms (especially KNN and SVR) from interpreting nominal categories as having ordered, mathematical relationships.
3. **Refine Imputation Strategy**: The logic in Cell 44 replacing missing `Year` values with the arithmetic mean of the entire dataset (`df.Year.mean()`) should be updated. A more localized approach, such as grouping the dataframe by `Platform` and imputing the median year for that specific console system, would preserve chronological integrity.
4. **Clean Notebook Structure**: The cells designated for "HyperParams Tunning" (Cells 109-114) contain no code and should be removed to finalize the notebook's presentation structure.

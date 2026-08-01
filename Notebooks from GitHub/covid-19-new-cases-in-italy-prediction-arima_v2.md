# Covid-19 New Cases Prediction with SARIMAX Technical Documentation

## 1. Notebook Overview
The primary problem statement for this notebook is clearly defined in its introductory sections: it aims at exploring the COVID-19 pandemic through data analysis and projections, specifically predicting the new COVID-19 cases in Italy for the future 7 days. The overall analytical approach begins with a comprehensive data loading and preparation phase, followed by extensive exploratory data analysis (EDA), which encompasses geographic data visualization and the use of the Pareto principle to identify the timeframes with the highest case concentrations. The time series is then decomposed and evaluated for stationarity using an Augmented Dickey-Fuller (ADF) test. Subsequently, a SARIMAX model (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) is utilized to account for seasonality and exogenous factors (such as holidays). Hyperparameters are optimized using Auto-ARIMA, and finally, the model is evaluated using a suite of error metrics and diagnostic plots. 

Notable characteristics of this notebook include a heavy reliance on automated profiling tools (like `pandas-profiling`), geographic mapping utilizing shapefiles provided by ISTAT (the Italian National Institute of Statistics), and the integration of exogenous variables (specifically a boolean flag for holidays) into the SARIMAX time series forecasting model. The notebook is meticulously documented with extensive markdown cells explaining the theoretical background of each statistical method utilized, such as the Pareto principle, Seasonal Decomposition, ACF/PACF, Auto-ARIMA, and the error metrics used for final evaluation.

## 2. Environment and Dependencies
The environment for this notebook requires the installation of several specialized libraries via `pip`, particularly for geospatial analysis and time series forecasting. The explicitly installed or imported libraries and their roles, as demonstrated in the code cells, are as follows:

*   **`pandas` (pd) / `numpy` (np)**: Fundamental data manipulation, cleaning, and mathematical operations.
*   **`seaborn` (sns) / `matplotlib.pyplot` (plt) / `matplotlib`**: Core plotting libraries for creating static visualizations, heatmaps, and diagnostic charts.
*   **`sklearn` (specifically `metrics`)**: Used for evaluating the performance of the predictive model, providing functions for R2, MSE, RMSE, MAE, MdAE, and MAPE.
*   **`plotly.express` (px)**: Utilized to create interactive plots that allow the user to navigate and zoom across different time ranges (e.g., 1m, 6m, YTD, 1y).
*   **`datetime` / `calendar` / `holidays`**: Used for temporal feature engineering, specifically to extract day of week, month, year, day of year, day of month, week of year, and to generate a binary feature indicating whether a specific date is an Italian holiday.
*   **`statsmodels`**: Essential for time series analysis. Used functions include `adfuller` for checking stationarity, `plot_acf` and `plot_pacf` for analyzing autocorrelations, `seasonal_decompose` for separating trend and seasonality, and `SARIMAX` for the modeling framework.
*   **`pmdarima` (specifically `auto_arima`)**: An automated hyperparameter tuning library that performs a stepwise search to identify the optimal (p, d, q) and (P, D, Q, m) parameters for the SARIMAX model based on AIC.
*   **`pandas_profiling` (`ProfileReport`)**: Used to rapidly generate an exploratory data analysis report summarizing the distributions, missing values, and statistics of the dataset.
*   **`geopandas` (gpd) / `shapefile` (shp)**: Libraries required to parse geospatial shapefiles (`.shp`) provided by ISTAT and merge them with the pandemic data to plot choropleth maps of new cases across Italian regions.
*   **`ipywidgets` / `shapely`**: Dependencies required for interactive elements and geospatial geometry processing.

## 3. Per-Stage Documentation

### Stage: Custom Functions Definition
**[CODE]** The notebook defines four custom python functions. `create_features(df)` extracts temporal features (dayofweek, month, year, dayofyear, dayofmonth, weekofyear, dayname, holiday) from the index. `my_plot(df, ...)` utilizes Plotly to generate an interactive line chart with a range slider. `last_n_months(df, n_months)` slices the dataframe to return only the records from the last N months, aligning by day names to ensure consistent week boundaries. Finally, `adf_test(series, title)` and `error_metrics(y_test, y_pred)` are wrapper functions that print formatted outputs for the Augmented Dickey-Fuller test and various error metrics respectively.
**[PROCESS/CONTEXT]** These functions are created early in the notebook to encapsulate repetitive logic that will be needed during the feature engineering, visualization, stationarity testing, and model evaluation phases.
**[RESULT]** No immediate numeric output; functions are compiled and loaded into memory.
**[INSIGHT]** The `create_features` function introduces the `holidays.Italy()` module, directly creating an exogenous variable that will be fed into the SARIMAX model later.

### Stage: Data Loading
**[CODE]** The code imports data directly from the official Protezione Civile GitHub repository (`https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv`). It selects a subset of columns, renames them for clarity (e.g., `nuovi_positivi` to `new_cases`, `denominazione_regione` to `region`), and checks for null values. It then extracts the maximum date to print a formatted string indicating the dataset's freshness.
**[PROCESS/CONTEXT]** The notebook notes that data collection started on February 24th. The dataset contains regional data that needs to be imported, cleaned, and verified for missing values before any analysis can begin. 
**[RESULT]** The `isnull().sum()` operation outputs zeros for all columns, indicating complete data. The final print statement outputs: `===================================== \n = The data is updated to 2023-12-06 = \n =====================================`
**[INSIGHT]** The dataset is perfectly clean with no missing values. The data update timestamp indicates the notebook pulls live data that extends into late 2023.

### Stage: Data Preparation for Region Analysis and General EDA
**[CODE]** The notebook creates two distinct dataframes. `covid_reg` aggregates `new_cases` grouped by `date` and `region`. The main `covid` dataframe aggregates `new_cases` strictly by `date`, summing cases across all regions to provide a national total. The custom `create_features` function is applied to the national dataframe.
**[PROCESS/CONTEXT]** Data preparation is an essential step to format the raw data for two distinct analytical paths: the geographical EDA (which requires regional granularity) and the time series forecasting (which requires national aggregates and temporal features).
**[RESULT]** A `.head(5)` call displays the first five days of national data starting from 2020-02-24, showing `new_cases` of 221, 93, 78, 250, and 238 for the first five days, alongside engineered features like `dayofweek`, `month`, `year`, `holiday` (all 0 for these days), etc.
**[INSIGHT]** The initial days of the pandemic in Italy showed high volatility in daily reported cases, jumping from 78 to 250 in a single day.

### Stage: EDA with Pandas Profiling
**[CODE]** The notebook filters the national `covid` dataframe to only include data for the year 2022. It then drops the index and unused columns, retaining only temporal features and `new_cases`. A `ProfileReport` is instantiated and rendered in the notebook.
**[PROCESS/CONTEXT]** The author explains that Pandas profiling generates a detailed report summarizing DataFrame content, which is useful for quickly understanding dataset characteristics, distributions, and potential issues. This specific EDA focuses solely on the year 2022.
**[RESULT]** The output generates a `<IPython.core.display.HTML object>` containing the interactive pandas profiling report. 
**[INSIGHT]** The analysis is explicitly restricted to 2022 for this specific profiling step, likely because analyzing the entire timeline at once might obscure granular seasonal trends, or 2022 represents a specific phase of the pandemic (the "pandemic tail").

### Stage: Geographic Data Analysis
**[CODE]** The `covid_reg` dataframe is modified to merge data from 'P.A. Bolzano' and 'P.A. Trento' into a single region called 'Trentino-Alto Adige'. The dataframe is then filtered to extract only the data from the most recent day (`last_day`). A shapefile (`Reg01012022_g_WGS84.shp`) is loaded via geopandas and joined with the filtered COVID data. A choropleth map and a horizontal bar chart are generated to visualize the new cases per region on the last day.
**[PROCESS/CONTEXT]** The author notes that geographic areas correspond to ISTAT definitions, and ISTAT categorizes the two autonomous provinces as one single region (Trentino-Alto Adige), necessitating the merge. The goal is explicitly stated as plotting Italy's regional new cases for the last day.
**[RESULT]** Two visual outputs are generated: an annotated choropleth map of Italy and a horizontal bar chart, both displaying `new_cases` for the most recent date in the dataset.
**[INSIGHT]** The geographical distribution requires manual coordinate conversion (via `representative_point()`) to properly overlay text annotations on the map, highlighting a quirk in how geopandas handles polygon centroids.

### Stage: Exploratory Data Analysis (Pareto and Distributions)
**[CODE]** The notebook calculates a cumulative percentage column (`cumperc`) of `new_cases` and plots it over the daily case bars to create a Pareto diagram. It then uses the `last_n_months` function to extract data for the last 3 months, generating a density plot (`sns.displot`) and a heatmap showing the mean cases per weekday per month. Finally, a bar plot is generated for the last month.
**[PROCESS/CONTEXT]** The notebook provides extensive theory on the Pareto principle (the 80-20 rule) and how it can identify the most significant time periods accounting for the majority of cases. It also explains that heatmaps help visualize patterns related to weekdays and seasons.
**[RESULT]** A Pareto chart is rendered. An observation is printed: "The majority of case are in 2022 in the pandemic tail!". A heatmap is generated for the last 3 months. Another observation is printed: "On Tuesday we have the peak for new cases". A color-gradient formatted table and bar chart for the last month are also generated.
**[INSIGHT]** The Pareto analysis reveals that despite the intense early waves of the pandemic, the sheer volume of cases was heavily concentrated in the later stages (2022). The heatmap reveals a strong weekly seasonality where Tuesdays consistently report the highest number of new cases, likely due to delayed reporting from weekend testing lulls.

### Stage: Data Preparation for SARIMAX
**[CODE]** The national `covid` dataframe is reduced to just two columns: `new_cases` and `holiday`. The frequency of the datetime index is strictly set to 'D' (daily). The data is split into a `train` set and a `test` set, where the test set is strictly the last 7 days (`steps = 7`). Exogenous variables (`holiday`) are separated and passed through `pd.get_dummies`.
**[PROCESS/CONTEXT]** The author explains the theoretical framework of SARIMAX, noting that it extends ARIMA by including seasonal components and exogenous variables. In this case, the exogenous variable is the holiday flag.
**[RESULT]** The data is successfully split. The `train` set contains all data except the last 7 days, and `train_exog` contains the corresponding holiday flags.
**[INSIGHT]** The decision to use a 7-day test set directly corresponds to the stated goal in the introduction: "predict Italy COVID-19 new cases for the future 7 days." The use of a boolean holiday flag as the sole exogenous variable assumes that holidays significantly disrupt testing and reporting cycles.

### Stage: SARIMA ACF/PACF and Stationarity
**[CODE]** The `seasonal_decompose` function is applied to the time series, splitting it into trend, seasonal, and residual components which are then plotted. The `adf_test` function is called on the raw `new_cases` series. ACF and PACF plots are generated for up to 200 lags. A new differenced series (`covid_diff`) is created using a lag of 1, and the ADF test and ACF/PACF plots are repeated on this differenced data.
**[PROCESS/CONTEXT]** The notebook provides deep theoretical context, explaining that seasonal decomposition isolates long-term direction, repeating patterns, and noise. It outlines the ADF test's role in proving stationarity. It also explains that ACF and PACF help determine the AutoRegressive (AR) and Moving Average (MA) parameters.
**[RESULT]** 
*   **Decomposition Plot**: Shows strong weekly seasonality.
*   **Original ADF Test**: ADF Statistic = -4.195436, p-value = 0.000671. Output states: "Data has no unit root and is stationary".
*   **Differenced ADF Test**: ADF Statistic = -7.984108, p-value = 2.576416e-12. Output states: "Data has no unit root and is stationary".
**[INSIGHT]** Interestingly, the ADF test on the raw data indicates it is already stationary (p-value < 0.05). However, the author explicitly notes in a markdown cell: "If the series has positive autocorrelations out to a high number of lags, then it probably needs a higher order of differencing. d = 1". The author ignores the ADF test's strict finding and manually forces a first-order difference (`d=1`) based on visual inspection of the slow decay in the ACF plot. 

### Stage: Find optimal hyperparameters (Auto-ARIMA)
**[CODE]** The `auto_arima` function is invoked on the training data (`endog`) with the exogenous holiday data. The parameters set are `seasonal = True`, `m = 7` (weekly seasonality), and `maxiter = 100`.
**[PROCESS/CONTEXT]** The author details the hyperparameters of SARIMAX (p, d, q, P, D, Q, m) and explains that Auto-ARIMA automatically searches over possible hyperparameter values to find the model that best fits the data, minimizing an information criterion.
**[RESULT]** The stepwise search runs multiple models. The initial model `ARIMA(2,1,2)(1,0,1)[7] intercept` yields an AIC of 28283.182. Several other models are tested, with some failing (`AIC=inf`). 
The final output is: `Best model:  ARIMA(2,1,2)(2,0,1)[7] intercept` with an AIC of `28205.653`. The total fit time was 231.029 seconds.
**[INSIGHT]** The Auto-ARIMA search confirmed the author's visual hypothesis from the ACF/PACF plots by selecting a model with `d=1` (first-order integrated). It also identified a complex seasonal component `(2,0,1)[7]`. The optimization process took a significant amount of compute time (~4 minutes), highlighting the cost of fitting SARIMAX models on large daily datasets spanning multiple years.

### Stage: Prediction and Evaluation
**[CODE]** The notebook forecasts the next 7 steps using the fitted `stepwise_fit` model and the exogenous holiday data corresponding to the test period. The actual vs. predicted values are plotted. The custom `error_metrics` function is invoked. Finally, `stepwise_fit.plot_diagnostics()` is called to generate residual analysis plots.
**[PROCESS/CONTEXT]** The final steps validate the model's accuracy on the holdout set. The notebook defines R2, MSE, RMSE, and MAE. It also defines the four components of the diagnostic plot: Residual plot, Q-Q plot, Scale-location plot, and Leverage/Correlogram plot.
**[RESULT]**
*   **Error Metrics**:
    *   R2: 0.870
    *   MSE: 1,459,916
    *   RMSE: 1,208
    *   MAE: 1,143
    *   MdAE: 946
    *   MAPE: 17.75%
*   **Diagnostics Plot**: Generates a 2x2 grid showing standardized residuals, histogram plus estimated density, Normal Q-Q, and Correlogram.
**[INSIGHT]** The model performs reasonably well, capturing 87% of the variance (R2 = 0.870). The MAPE of 17.75% suggests that on average, the predictions deviate from the actual daily cases by roughly 18%. The absolute errors (RMSE of 1,208 and MAE of 1,143) show that the model's forecasts are typically off by over a thousand cases, which, given the scale of cases during peaks (often tens of thousands), represents an acceptable forecasting margin for a 7-day horizon.

## 4. Cross-Cell Dependency Analysis
*   **Concrete Variable Dependencies**:
    *   The geographic analysis heavily relies on the `covid_reg` dataframe created much earlier in the notebook. Modifications to the region names (merging Trento and Bolzano) must execute exactly once. If the cell merging the autonomous provinces is executed multiple times, it may corrupt the underlying dataframe by appending data repeatedly or failing due to missing keys.
    *   The `train` and `test` dataframes are entirely dependent on the integer length of the full dataset minus `steps = 7`. Any filtering of the `covid` dataframe applied upstream will silently alter the dates that fall into the test set.
    *   The `exog_forecast` variable fundamentally depends on the `exog` dataframe generated via `pd.get_dummies(covid_ex)`. If the holiday array changes length, the entire forecasting cell will crash due to dimensionality mismatch.
*   **Fragile Execution Order Dependencies**:
    *   The function `last_n_months(df, n_months)` relies on a globally scoped variable `covid` to determine the maximum index date (`covid.index.max()`), even though `df` is passed as an argument. This is a severe fragility; the function will produce incorrect results or crash if it is passed any dataframe with a different temporal boundary than the global `covid` dataframe, or if the `covid` dataframe has not been instantiated in memory prior to the function definition being called.
    *   The notebook dynamically updates the string `updated_to` based on the data pulled at runtime. Consequently, the test set (the "last 7 days") is a sliding window that changes every time the notebook is run with a fresh pull from GitHub. The SARIMAX model's performance metrics will fluctuate unpredictably depending on the specific volatility of the 7 days preceding the notebook's execution.

## 5. Model Performance Summary
*   **Selected Model**: The Auto-ARIMA stepwise search determined that the optimal configuration is `ARIMA(2,1,2)(2,0,1)[7] intercept`.
*   **Information Criteria**: The best model achieved an AIC (Akaike Information Criterion) of `28205.653`, outperforming the base model `ARIMA(0,1,0)(0,0,0)[7]` which had an AIC of `30097.042`.
*   **Evaluation Metrics (7-day forecast)**:
    *   **R-Squared (R2)**: `0.870`
    *   **Root Mean Squared Error (RMSE)**: `1,208`
    *   **Mean Absolute Error (MAE)**: `1,143`
    *   **Median Absolute Error (MdAE)**: `946`
    *   **Mean Absolute Percentage Error (MAPE)**: `17.75%`

The model successfully captures the strong weekly seasonality of the reporting cycle, primarily driven by the `m=7` parameter and the integration of the exogenous holiday variable.

## 6. Conclusions and Recommendations

### Observable Problems in the Notebook
1.  **Global Variable Leakage**: As identified in the dependency analysis, the custom `last_n_months` function references the global variable `covid` instead of the passed parameter `df` when calculating `last_day_name = df.loc[covid.index.max(),'dayname']`. This is a strict coding error that compromises the reusability of the function.
2.  **Ignored Statistical Tests**: The notebook performs an Augmented Dickey-Fuller test which conclusively outputs that the original time series is stationary (p-value = 0.000671). However, the author subsequently states that the series needs differencing (`d=1`) based on the slow decay of the ACF plot, entirely overriding the statistical test they just performed.
3.  **Moving Target for Evaluation**: Because the notebook pulls live data from the master branch of a GitHub repository, the "last 7 days" used as the test set will change constantly. This means the evaluation metrics (R2, RMSE) are irreproducible without hardcoding a specific cutoff date.

### Concrete Suggestions
1.  **Fix Function Scoping**: Refactor the `last_n_months` function to calculate the maximum date based strictly on the dataframe passed to it: `last_day_name = df.loc[df.index.max(), 'dayname']`.
2.  **Lock the Dataset for Reproducibility**: To ensure the model's performance metrics remain stable and the notebook serves as a reproducible artifact, the data import step should pull from a specific commit hash in the Protezione Civile repository, or the notebook should manually truncate the data at a fixed date prior to the train/test split.
3.  **Reconcile ADF Test and Differencing**: The documentation should address why the ADF test's conclusion of stationarity was rejected in favor of visual inspection. If the visual slow decay in the ACF is deemed more reliable for this specific epidemiological data, this rationale should be explicitly stated to justify forcing `d=1` in the subsequent analysis.

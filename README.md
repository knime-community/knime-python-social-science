# Statistics and Social Science Extension for KNIME

This repository contains the code for the Statistics and Social Science Extension for the KNIME Analytics Platform. This extension provides nodes for multivariate statistical analysis, time series modeling, and data visualizations, enabling users to perform in-depth explorations of structured data.

The extension is curated and maintained by Francesco Tuscolano (KNIME), Prof. Daniele Tonini, and Pietro Maran (Bocconi University, Milan).

The project's goal is to integrate advanced statistical methodologies within KNIME by leveraging bundled Python packages and transforming them into native KNIME nodes.

## Current Nodes

* **Auto-SARIMA Learner**: Automatically finds the optimal parameters for and trains a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model on a given time series using simulated annealing optimization. Returns in-sample predictions, residuals, comprehensive model diagnostics, and optimization history. Features intelligent fallback strategies and adaptive diagnostic testing.

* **Auto-SARIMA Predictor**: Generates future forecasts using a pre-trained SARIMA model from the Auto-SARIMA Learner node.

* **(Partial) Autocorrelation Functions and Plots**: Calculates and visualizes Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for time series analysis. Provides both plots and numerical values with confidence intervals, plus Ljung-Box Q-statistics for testing autocorrelation. Supports multiple PACF calculation methods.

* **Correspondence Analyzer**: Performs Correspondence Analysis (CA) or Multiple Correspondence Analysis (MCA) on categorical data to explore associations between variables. Automatically selects the appropriate method based on input dimensionality.

* **ETS Learner**: Trains an Error-Trend-Seasonal (ETS) exponential smoothing model for time series forecasting. Automatically selects optimal model components (error, trend, seasonal) or allows manual specification. Supports both additive and multiplicative components with optional damping for trend.

* **ETS Predictor**: Generates out-of-sample forecasts from a fitted ETS model with prediction intervals. Produces forecast plots showing historical data, predictions, and confidence bounds.

* **Factor Analyzer**: Trains a dimensionality-reduction model using either Principal Component Analysis (PCA) or Exploratory Factor Analysis (EFA). Supports optional component/factor rotation for interpretability (varimax, promax, quartimax) and incremental PCA for large datasets.

* **Factor Predictor**: Applies a trained PCA or EFA model to new datasets, projecting data onto the learned components or factors. Maintains full mathematical consistency with all model variants including rotated solutions and incremental PCA.

* **Linear Model Learner**: Estimates linear regression models with support for Ordinary Least Squares (OLS), Ridge, Lasso regularization, and panel data methods with fixed or random effects. Provides comprehensive diagnostics including heteroskedasticity tests, normality tests, and multicollinearity analysis.

* **Time Series Interpolator**: Fills missing values in time series data using various interpolation methods (linear, polynomial, spline, PCHIP, Akima). Supports optional Box-Cox transformation for handling non-stationary variance and automatically back-transforms results.

## Package Organization

* **`icons/`**: Directory containing visual assets and icon images for the extension nodes, including specialized icons for time series analysis, correspondence analysis, and factor analysis components.
* **`config.yml`**: Configuration file specifying the path to the extension source code directory. This file works in conjunction with `knime.yml` to define the extension structure and dependencies.
* **`knime.yml`**: YAML configuration file containing extension metadata, including extension identification, module paths, and KNIME integration specifications.
* **`pixi.toml & pixi.lock`**: Pixi package manager configuration file defining Python dependencies, environment setup, and reproducible package management for the extension.
* **`ruff.toml`**: Configuration file for Ruff linter and formatter, ensuring code quality and consistent styling across the project.
* **`LICENSE.TXT`**: License file containing the terms and conditions under which this extension is distributed.
* **`src/social_science_ext.py`**: Main extension module that registers and initializes all nodes within the Social Science Extension for KNIME Analytics Platform.
* **`src/nodes/`**: Core implementation directory organized by analysis type:
  - **`multivariate_analysis/`**: Statistical methods for analyzing relationships between multiple variables:
    - `correspondence_analysis.py`: Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA) for categorical data
    - `factor_analysis.py`: Factor analysis learner (PCA/EFA) with multiple rotation methods
    - `factor_scorer.py`: Factor scoring node for applying trained models to new data
    - `linear_regression.py`: Linear regression learner with OLS, Ridge, Lasso, and panel data support
    - `multivariate_cat.py`: It contains the subcategory used for the multivariate analysis nodes.
  - **`timeseries_analysis/`**: Time series modeling and forecasting methods:
    - `acf_pacf_plot.py`: Autocorrelation and Partial Autocorrelation Functions with plots and diagnostics
    - `arima_learner.py`: SARIMA/SARIMAX model fitting with simulated annealing optimization
    - `arima_predictor.py`: Forecasting predictor for trained ARIMA models
    - `exponential_smoothing_learner.py`: Error-Trend-Seasonal (ETS) exponential smoothing model learner
    - `exponential_smoothing_predictor.py`: ETS model predictor for generating forecasts
    - `ts_interpolator.py`: Time series missing value interpolation with various methods
    - `timeseries_cat.py`: It contains the subcategory used for the timeseries analysis nodes.
* **`src/util/`**: Utility module directory containing helper functions and shared code components:
  - `utils.py`: Collection of utility functions providing common data processing and validation methods

This extension provides a comprehensive suite of statistical analysis tools specifically designed for social science research within the KNIME Analytics Platform, supporting both time series analysis and multivariate statistical methods for categorical and continuous data.

"""
Several utility functions are reused from Harvard's spatial data lab repository for Geospatial Analytics Extension.
https://github.com/spatial-data-lab/knime-geospatial-extension/blob/main/knime_extension/src/util/knime_utils.py
"""

import knime.extension as knext
import pandas as pd
from typing import Callable
import logging


############################################
# Timestamp column selection helper
############################################

# Strings of IDs of date/time value factories
ZONED_DATE_TIME_ZONE_VALUE = "org.knime.core.data.v2.time.ZonedDateTimeValueFactory2"
LOCAL_TIME_VALUE = "org.knime.core.data.v2.time.LocalTimeValueFactory"
LOCAL_DATE_VALUE = "org.knime.core.data.v2.time.LocalDateValueFactory"
LOCAL_DATE_TIME_VALUE = "org.knime.core.data.v2.time.LocalDateTimeValueFactory"


DEF_ZONED_DATE_LABEL = "ZonedDateTimeValueFactory2"
DEF_DATE_LABEL = "LocalDateValueFactory"
DEF_TIME_LABEL = "LocalTimeValueFactory"
DEF_DATE_TIME_LABEL = "LocalDateTimeValueFactory"

# Timestamp formats
ZONED_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"


def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return column.ktype == knext.double() or column.ktype == knext.int32() or column.ktype == knext.int64()


def is_string(column: knext.Column) -> bool:
    """
    Checks if column is a string type.
    @return: True if Column is a string
    """
    return column.ktype == knext.string()


def is_zoned_datetime(column: knext.Column) -> bool:
    """
    Checks if date&time column contains has the timezone or not.
    @return: True if selected date&time column has time zone
    """
    return __is_type_x(column, ZONED_DATE_TIME_ZONE_VALUE)


def is_datetime(column: knext.Column) -> bool:
    """
    Checks if a column is of type Date&Time.
    @return: True if selected column is of type date&time
    """
    return __is_type_x(column, LOCAL_DATE_TIME_VALUE)


def is_time(column: knext.Column) -> bool:
    """
    Checks if a column is of type Time only.
    @return: True if selected column has only time.
    """
    return __is_type_x(column, LOCAL_TIME_VALUE)


def is_date(column: knext.Column) -> bool:
    """
    Checks if a column is of type date only.
    @return: True if selected column has date only.
    """
    return __is_type_x(column, LOCAL_DATE_VALUE)


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def is_type_timestamp(column: knext.Column):
    """
    This function checks on all the supported timestamp columns supported in KNIME.
    Note that legacy date&time types are not supported.
    @return: True if date&time column is compatible with the respective logical types supported in KNIME.
    """

    return boolean_or(is_time, is_date, is_datetime, is_zoned_datetime)(column)


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type whereas type can be :
    DateTime, Date, Time, ZonedDateTime
    @return: True if column type is of type timestamp
    """

    return isinstance(column.ktype, knext.LogicalType) and type in column.ktype.logical_type


############################################
# General Helper Class
############################################


def column_exists_or_preset(
    context: knext.ConfigurationContext,
    column: str,
    schema: knext.Schema,
    func: Callable[[knext.Column], bool] = None,
    none_msg: str = "No compatible column found in input table",
) -> str:
    """
    Checks that the given column is not None and exists in the given schema. If none is selected it returns the
    first column that is compatible with the provided function. If none is compatible it throws an exception.
    """
    if column is None:
        for c in schema:
            if func(c):
                context.set_warning(f"Preset column to: {c.name}")
                return c.name
        raise knext.InvalidParametersError(none_msg)
    __check_col_and_type(column, schema, func)
    return column


def __check_col_and_type(
    column: str,
    schema: knext.Schema,
    check_type: Callable[[knext.Column], bool] = None,
) -> None:
    """
    Checks that the given column exists in the given schema and that it matches the given type_check function.
    """
    # Check that the column exists in the schema and that it has a compatible type
    try:
        existing_column = schema[column]
        if check_type is not None and not check_type(existing_column):
            raise knext.InvalidParametersError(f"Column '{str(column)}' has incompatible data type")
    except IndexError:
        raise knext.InvalidParametersError(f"Column '{str(column)}' not available in input table")


def seasonality_performance_warning(
    context: knext.ExecutionContext,
    LOGGER: logging.Logger,
    seasonality: int,
    seasonality_warning_threshold: int = 100,
) -> None:
    """
    Issues a performance warning if the seasonal period is very large.
    This helps users avoid severe computational issues when fitting models with long seasonal periods.
    Default threshold is 100 seasonal periods.
    """
    if seasonality > seasonality_warning_threshold:
        context.set_warning(
            f"⚠️ PERFORMANCE WARNING: Large seasonal period detected: ({seasonality})."
            f"This will cause severe computational issues:\n"
            f"• Extremely slow model fitting (potentially hours)\n"
            f"• Risk of system freeze or out-of-memory errors\n\n"
            f"• High memory consumption that may exhaust available RAM\n"
            f"• Potential numerical instability and fitting failures\n"
            f"RECOMMENDED SOLUTIONS:\n"
            f"1. Use Fourier Transform or STL decomposition to remove long-term seasonality first\n"
            f"2. Aggregate data to reduce seasonal period (daily→weekly: s=52, daily→monthly: s=12)\n"
            f"3. Focus on dominant seasonality (e.g., weekly s=7 instead of yearly s=365)\n"
            f"Consider canceling execution if you have limited computational resources."
        )
        LOGGER.warning(f"Large seasonal period warning issued for seasonality = {seasonality}")


def validate_missing_values(column: pd.Series) -> None:
    """
    Validates input time series for missing values.
    Models for time series forecasting require complete time series data without gaps. This method checks for
    any missing (NaN) values and raises an error if found, preventing model fitting failures.

    Parameters:
    - column: pd.Series
        Time series data to validate for completeness.

    Raises:
    - knext.InvalidParametersError
        If missing values are detected, with count information.
    """
    if check_missing_values(column):
        missing_count = count_missing_values(column)
        raise knext.InvalidParametersError(f'There are "{missing_count}" number of missing values in the target column.')


def enhance_predictions_table(
    model, input_table: knext.Table, input_column: str, seasonality: int, DEFAULT_SKIP_OBSERVATIONS: int, pd
) -> pd.DataFrame:
    """
    Create an enhanced predictions table that includes original values, predictions, and residuals.

    This method automatically excludes the first 2*seasonal_period predictions (or first 10 for
    non-seasonal models) as these initial predictions are typically unstable due to parameter
    estimation effects and can produce misleadingly high residuals.

    Parameters:
    - model: statsmodels.tsa.statespace.sarimax.ETSResults / statsmodels.tsa.statespace.sarimax.SARIMAXResults
        The fitted ETS/SARIMAX model object.
    - input_table: knext.Table
        The original input table containing the time series data.
    - input_column: str
        The name of the target column to extract original values from.
    - seasonality: int
        The seasonal period of the time series (0 for non-seasonal).
    - DEFAULT_SKIP_OBSERVATIONS: int
        Default number of initial observations to skip for non-seasonal models.
    - pd: pandas module
        Pandas module for DataFrame creation.

    Returns:
    - pd.DataFrame
        Enhanced predictions DataFrame with stable predictions (excluding initial unstable period).
    """

    # Get model predictions and residuals
    fitted_values = model.fittedvalues
    residuals = model.resid

    # Convert input table to pandas to get original values
    input_df = input_table.to_pandas()

    # Create enhanced predictions table
    predictions_data = []

    # Skip initial unstable period in predictions (same logic as diagnostics)
    seasonal_period = seasonality
    skip_initial = max(2 * seasonal_period, DEFAULT_SKIP_OBSERVATIONS) if seasonal_period > 0 else DEFAULT_SKIP_OBSERVATIONS

    # Match the length of fitted values (may be shorter due to differencing)
    start_idx = len(input_df) - len(fitted_values)

    for i, (fitted_val, residual) in enumerate(zip(fitted_values, residuals)):
        # Skip initial unstable predictions
        if i < skip_initial and len(fitted_values) > skip_initial:
            continue

        original_idx = start_idx + i
        if original_idx < len(input_df):
            # Get the original value from the correct target column
            original_value = input_df[input_column].iloc[original_idx]

            # Ensure the original value is numeric - convert safely
            try:
                original_value_float = float(original_value)
            except (ValueError, TypeError) as e:
                # If conversion fails, provide more info about the problematic value
                raise knext.InvalidParametersError(
                    f"Cannot convert value '{original_value}' (type: {type(original_value)}) "
                    f"from column '{input_column}' to numeric. "
                    f"Please ensure the selected column contains only numeric values. Error: {str(e)}"
                )

            predictions_data.append(
                {
                    "Original Value": original_value_float,
                    "Fitted Value": float(fitted_val),
                    "Residual": float(residual),
                    "Absolute Error": float(abs(residual)),
                }
            )

    predictions_df = pd.DataFrame(predictions_data)

    # Ensure proper column order and types
    expected_columns = ["Original Value", "Fitted Value", "Residual", "Absolute Error"]
    if list(predictions_df.columns) != expected_columns:
        raise knext.InvalidParametersError(f"Predictions columns mismatch. Expected: {expected_columns}, Got: {list(predictions_df.columns)}")

    # Ensure all columns are float type
    for col in expected_columns:
        predictions_df[col] = predictions_df[col].astype("float64")

    return predictions_df


def get_model_stats(model, data: list | None = None) -> pd.DataFrame:
    """
    Compiles comprehensive model summary with parameters, coefficients, and fit statistics.

    Creates or expands a detailed table containing key goodness-of-fit metrics.
    Each entry includes explanatory text to help interpret the results.

    Parameters:
    - model: Results object from statsmodels' model fitting (e.g., ETSResults, SARIMAXResults)
        Fitted model containing coefficients and statistics.
    - data: list | None
        Optional initial data list to append model stats to. If None, a new list is created.
    - pd: pandas module
        Pandas module for DataFrame creation.

    Returns:
    - pd.DataFrame
        Model summary table with parameters, coefficients, and interpretive explanations.
    """

    data = data if data is not None else []
    # Model statistics
    data.append(
        {
            "Parameter": "Log Likelihood",
            "Value": float(model.llf),
            "Explanation": "Logarithm of the likelihood function; higher values indicate better fit.",
        }
    )
    data.append(
        {
            "Parameter": "AIC",
            "Value": float(model.aic),
            "Explanation": "Akaike Information Criterion; lower values indicate better model balance of fit and complexity.",
        }
    )
    data.append(
        {
            "Parameter": "BIC",
            "Value": float(model.bic),
            "Explanation": "Bayesian Information Criterion; lower values indicate better model with penalty for complexity.",
        }
    )
    data.append(
        {
            "Parameter": "MSE",
            "Value": float(model.mse),
            "Explanation": "Mean Squared Error of residuals; lower values indicate better predictions.",
        }
    )
    data.append(
        {
            "Parameter": "MAE",
            "Value": float(model.mae),
            "Explanation": "Mean Absolute Error of residuals; lower values indicate better predictions.",
        }
    )

    # Create DataFrame
    summary = pd.DataFrame(data)

    # Ensure proper column order and types
    expected_columns = ["Parameter", "Value", "Explanation"]
    if list(summary.columns) != expected_columns:
        raise knext.InvalidParametersError(f"Model summary columns mismatch. Expected: {expected_columns}, Got: {list(summary.columns)}")

    # Ensure proper data types
    summary["Parameter"] = summary["Parameter"].astype(str)
    summary["Value"] = summary["Value"].astype(float)
    summary["Explanation"] = summary["Explanation"].astype(str)

    return summary


def compute_residual_diagnostics(model, seasonality: int, DEFAULT_SKIP_OBSERVATIONS: int, DEFAULT_LJUNG_BOX_LAGS: int, pd) -> pd.DataFrame:
    """
    Computes comprehensive residual diagnostic tests for the fitted model. Allowed models so far are ETS and SARIMAX models from statsmodels.

    This function performs several statistical tests on the model residuals to assess
    model adequacy and assumptions:
    - Ljung-Box test for autocorrelation in residuals
    - Jarque-Bera test for normality of residuals (sensitive to large samples)
    - Shapiro-Wilk test for normality (more reliable for smaller to medium samples)

    Note: Excludes the first 2*seasonal_period observations from testing as these
    initial predictions are typically unstable due to parameter estimation effects.

    Parameters:
    - model: statsmodels.tsa.statespace.sarimax.ETSResults / statsmodels.tsa.statespace.sarimax.SARIMAXResults
        The fitted ETS/SARIMAX model object.
    - pd: pandas module
        Pandas module for DataFrame creation.

    Returns:
    - pd.DataFrame
        DataFrame containing test names, statistics, p-values, and interpretations.
    """
    # Import additional dependencies for diagnostics
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera, shapiro

    residuals = model.resid

    # Skip initial unstable period (2x seasonal period)
    # For non-seasonal models (seasonality = 0), skip first DEFAULT_SKIP_OBSERVATIONS as default
    skip_initial = max(2 * seasonality, DEFAULT_SKIP_OBSERVATIONS) if seasonality > 0 else DEFAULT_SKIP_OBSERVATIONS

    # Ensure we have enough observations after skipping
    if len(residuals) <= skip_initial:
        # If not enough data, use all residuals but add warning to interpretations
        stable_residuals = residuals
        stability_note = " (Warning: Insufficient data to skip initial unstable period)"
    else:
        stable_residuals = residuals.iloc[skip_initial:]
        stability_note = f" (Excluding first {skip_initial} observations)"
    diagnostics_data = []

    # Adaptive Ljung-Box test with seasonality-based lag selection
    try:
        if seasonality == 0:
            # Non-seasonal ARIMA: use default 10 lags
            ljung_box_lags = DEFAULT_LJUNG_BOX_LAGS
            lag_note = f" (using {ljung_box_lags} lags for non-seasonal model)"
        else:
            # Seasonal SARIMA: adaptive lag selection to capture seasonal patterns
            # Use max(10, 1.5 * seasonality) but cap at 165 for computational efficiency
            ljung_box_lags = min(max(DEFAULT_LJUNG_BOX_LAGS, int(1.5 * seasonality)), 165)
            lag_note = f" (using {ljung_box_lags} lags for seasonal period s={seasonality})"

        # Use return_df=True to get proper DataFrame output
        lb_result = acorr_ljungbox(stable_residuals, lags=ljung_box_lags, return_df=True)
        # Get the test statistic and p-value for the highest lag (last row)
        lb_stat = float(lb_result["lb_stat"].iloc[-1])
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])
        lb_interpretation = ("No autocorrelation" if lb_pvalue > 0.05 else "Autocorrelation detected") + lag_note + stability_note
        diagnostics_data.append(["Ljung-Box Test", float(lb_stat), float(lb_pvalue), lb_interpretation])
    except Exception as e:
        diagnostics_data.append(["Ljung-Box Test", float("nan"), float("nan"), f"Test failed: {str(e)[:1000]}"])

    # Jarque-Bera test for normality
    try:
        jb_stat, jb_pvalue = jarque_bera(stable_residuals)

        # Improve interpretation considering sample size sensitivity
        n_obs = len(stable_residuals)
        if n_obs > 500:
            # For large samples, be more lenient as JB test becomes overly sensitive
            threshold = 0.01  # More stringent threshold for large samples
            jb_interpretation = (
                f"Residuals approximately normal (n={n_obs}, large sample)"
                if jb_pvalue > threshold
                else f"Residuals deviate from normality (n={n_obs}, JB test sensitive to large samples)"
            )
        else:
            # Standard interpretation for smaller samples
            threshold = 0.05
            jb_interpretation = f"Residuals are normal (n={n_obs})" if jb_pvalue > threshold else f"Residuals are non-normal (n={n_obs})"

        jb_interpretation += stability_note
        diagnostics_data.append(["Jarque-Bera Test", float(jb_stat), float(jb_pvalue), jb_interpretation])
    except Exception as e:
        diagnostics_data.append(["Jarque-Bera Test", float("nan"), float("nan"), f"Test failed: {str(e)[:50]}"])

    # Shapiro-Wilk test for normality (more reliable for smaller to medium samples)
    try:
        n_obs = len(stable_residuals)
        if 3 <= n_obs <= 5000:  # Shapiro-Wilk has limitations on sample size
            sw_stat, sw_pvalue = shapiro(stable_residuals)
            sw_interpretation = (
                f"Residuals are normal (n={n_obs}, Shapiro-Wilk)" if sw_pvalue > 0.05 else f"Residuals are non-normal (n={n_obs}, Shapiro-Wilk)"
            ) + stability_note
            diagnostics_data.append(["Shapiro-Wilk Test", float(sw_stat), float(sw_pvalue), sw_interpretation])
        else:
            reason = "Too few observations" if n_obs < 3 else "Too many observations (>5000)"
            diagnostics_data.append(["Shapiro-Wilk Test", float("nan"), float("nan"), f"{reason} for Shapiro-Wilk test"])
    except Exception as e:
        diagnostics_data.append(["Shapiro-Wilk Test", float("nan"), float("nan"), f"Test failed: {str(e)[:50]}"])

    # Create DataFrame
    diagnostics_df = pd.DataFrame(diagnostics_data, columns=["Test", "Statistic", "P-Value", "Interpretation"])

    # Ensure proper column order and types
    expected_columns = ["Test", "Statistic", "P-Value", "Interpretation"]
    if list(diagnostics_df.columns) != expected_columns:
        raise knext.InvalidParametersError(f"Diagnostics columns mismatch. Expected: {expected_columns}, Got: {list(diagnostics_df.columns)}")

    # Ensure proper data types
    diagnostics_df["Test"] = diagnostics_df["Test"].astype(str)
    diagnostics_df["Statistic"] = diagnostics_df["Statistic"].astype(float)
    diagnostics_df["P-Value"] = diagnostics_df["P-Value"].astype(float)
    diagnostics_df["Interpretation"] = diagnostics_df["Interpretation"].astype(str)

    return diagnostics_df


def box_cox_transform(
    series,
    LOGGER: logging.Logger,
    lambda_value: float | None = None,
) -> tuple[pd.Series, float]:
    """
    Applies Box-Cox transformation to the given Pandas Series.
    If lambda_value is None, it estimates the optimal lambda using scipy's boxcox.
    Bounds are set to [-0.9, 2], consistently with forecast from R.
    If bounds are hit from automatic selection, they are enforced, and a warning is logged.
    If bounds are hit from manual selection, only a warning is logged.

    Parameters:
    - series: array-like
        The input data array to transform. Must contain only positive values.
    - lambda_value: float | None
        The Box-Cox lambda parameter. If None, it will be estimated.

    Returns:
    - pd.Series
        The Box-Cox transformed data series.
    """
    from scipy.stats import boxcox

    if (series <= 0).any():
        raise knext.InvalidParametersError("Box-Cox transformation requires all observed values to be positive.")

    # Estimate optimal lambda
    if lambda_value is None:
        transformed_series, lambda_value = boxcox(series)
        if lambda_value < -0.9 or lambda_value > 2.0:
            lambda_value = max(min(lambda_value, 2.0), -0.9)  # Bound lambda to [-0.9, 2]
            LOGGER.warning(
                f"Automatic selection of lambda hit the reasonable bounds [-0.9, 2]. Recomputing transformation with lambda = {lambda_value}"
            )
            transformed_series = boxcox(series, lmbda=lambda_value)
    # Apply Box-Cox transformation with provided lambda
    else:
        if lambda_value < -0.9 or lambda_value > 2.0:
            LOGGER.warning(
                "Provided Box-Cox lambda is out of bounds [-0.9, 2]. Consider using a value within this range to avoid unstable transformations."
            )
        transformed_series = boxcox(series, lmbda=lambda_value)

    if transformed_series.any() == float("inf") or transformed_series.any() == float("-inf"):
        raise knext.InvalidParametersError("Box-Cox transformation generated infinite values. Please check the input data and lambda parameter.")

    return (pd.Series(transformed_series), lambda_value)


def inv_box_cox_transform(series, lambda_value: float) -> pd.Series:
    """
    Applies the inverse Box-Cox transformation to the given Pandas Series.

    Parameters:
    - series: array-like
        The Box-Cox transformed data array.
    - lambda_value: float
        The Box-Cox lambda parameter used during the forward transformation.

    Returns:
    - pd.Series
        The inverse Box-Cox transformed data series.
    """
    from scipy.special import inv_boxcox

    inverse_transformed_series = inv_boxcox(series, lambda_value)

    return pd.Series(inverse_transformed_series)


############################################
# Generic pandas dataframe/series helper function
############################################


def check_missing_values(column: pd.Series) -> bool:
    """
    This function checks for missing values in the Pandas Series.
    @return: True if missing values exist in column
    """
    return column.hasnans


def count_missing_values(column: pd.Series) -> int:
    """
    This function counts the number of missing values in the Pandas Series.
    @return: sum of boolean 1s if missing value exists.
    """
    return column.isnull().sum()


def number_of_rows(df: pd.Series) -> int:
    """
    This function returns the number of rows in the dataframe.
    @return: numerical value, denoting length of Pandas Series.
    """
    return len(df.index)


def count_negative_values(column: pd.Series) -> int:
    total_neg = (column <= 0).sum()

    return total_neg

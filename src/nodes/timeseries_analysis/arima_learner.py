import logging
import knime.extension as knext
from util import utils as kutil
from .timeseries_cat import timeseries_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.parameter_group("Non Seasonal Parameters")
class NonSeasonalParams:
    """
    Maximum search bounds for non-seasonal ARIMA components (p,d,q). These constraints define the upper limits
    for parameter exploration during optimization. Higher bounds allow more complex models but increase computation time.
    """

    max_ar = knext.IntParameter(
        label="Max AR Order (p)",
        description="Maximum autoregressive order - how many previous observations can influence current value. Higher values capture longer-term dependencies but increase model complexity. Typical range: 0-5.",
        default_value=5,
        min_value=0,
    )
    max_i = knext.IntParameter(
        label="Max I Order (d)",
        description="Maximum non-seasonal differencing order - how many times the series can be differenced to achieve stationarity. Usually 0-2 is sufficient for most time series. Higher values rarely needed.",
        default_value=2,
        min_value=0,
    )
    max_ma = knext.IntParameter(
        label="Max MA Order (q)",
        description="Maximum moving average order - how many previous forecast errors can influence current prediction. Captures short-term noise patterns. Typical range: 0-5.",
        default_value=5,
        min_value=0,
    )


@knext.parameter_group("Seasonal Parameters")
class SeasonalParams:
    """
    Maximum search bounds for seasonal ARIMA components (P,D,Q). These constraints control the complexity
    of seasonal patterns the model can capture. Set to 0 to disable specific seasonal components.
    """

    max_s_ar = knext.IntParameter(
        label="Max Seasonal AR Order (P)",
        description="Maximum seasonal autoregressive order - how many seasonal lags can influence current value. Captures year-over-year or cycle-over-cycle dependencies. Typical range: 0-3.",
        default_value=0,
        min_value=0,
    )
    max_s_i = knext.IntParameter(
        label="Max Seasonal I Order (D)",
        description="Maximum seasonal differencing order - how many seasonal differences needed for stationarity. Usually 0-1 is sufficient. Set to 0 if no seasonal trends are present.",
        default_value=0,
        min_value=0,
    )
    max_s_ma = knext.IntParameter(
        label="Max Seasonal MA Order (Q)",
        description="Maximum seasonal moving average order - how many seasonal forecast errors can influence predictions. Captures seasonal noise patterns. Typical range: 0-3.",
        default_value=0,
        min_value=0,
    )


@knext.parameter_group("Model Selection")
class ModelSelectionParams:
    """
    Information criterion selection for model comparison during optimization. Different criteria emphasize
    different trade-offs between model fit quality and complexity penalty.
    """

    class SelectionCriteria(knext.EnumParameterOptions):
        AIC = ("AIC", "Akaike Information Criterion - Optimizes prediction accuracy, allows moderate complexity. Best for forecasting applications.")
        BIC = (
            "BIC",
            "Bayesian Information Criterion - Strong complexity penalty, favors simpler models. Best for model interpretation and explanation.",
        )
        HQIC = ("HQIC", "Hannan-Quinn Information Criterion - Balanced approach between prediction and parsimony. Good general-purpose choice.")

    selection_criterion = knext.EnumParameter(
        label="Model Selection Criterion",
        description="Information criterion for comparing models during optimization. Lower values indicate better models. AIC emphasizes prediction accuracy, BIC favors simplicity, HQIC balances both concerns.",
        default_value=SelectionCriteria.AIC.name,
        enum=SelectionCriteria,
    )


@knext.parameter_group("Optimization Loop Parameters")
class OptimizationLoopParams:
    """
    Advanced simulated annealing configuration for parameter search. Controls the thoroughness and behavior
    of the optimization process. Higher values provide more thorough search but increase computation time.
    """

    anneal_steps = knext.IntParameter(
        label="Number of Annealing Steps",
        description="Temperature schedule length for optimization. More steps provide thorough search but increase runtime. Default (5) balances quality and speed for most applications.",
        default_value=5,
        min_value=2,
        is_advanced=True,
    )
    mcmc_steps = knext.IntParameter(
        label="MCMC Steps per Annealing Step",
        description="Parameter proposals at each temperature level. More steps improve exploration but increase computation. Default (10) is sufficient for typical parameter spaces.",
        default_value=10,
        min_value=1,
        is_advanced=True,
    )
    beta0 = knext.DoubleParameter(
        label="Initial Annealing Temperature (beta0)",
        description="Starting temperature (exploration phase). Lower values allow more random exploration initially. Must be less than beta1. Default (0.1) works well for most cases.",
        default_value=0.1,
        min_value=0.001,
        is_advanced=True,
    )
    beta1 = knext.DoubleParameter(
        label="Final Annealing Temperature (beta1)",
        description="Final temperature before greedy phase. Higher values become more selective sooner. Must be greater than beta0. Default (3.0) provides good convergence.",
        default_value=3.0,
        min_value=0.001,
        is_advanced=True,
    )
    step_size = knext.IntParameter(
        label="Step Size for Parameter Proposals",
        description="Maximum parameter change per proposal. Larger steps explore more aggressively but may skip good solutions. Default (1) provides steady progress.",
        default_value=1,
        min_value=1,
        max_value=3,
        is_advanced=True,
    )
    early_stopping_patience = knext.IntParameter(
        label="Early Stopping Patience",
        description="Stops optimization after this many steps without improvement. Prevents unnecessary computation when convergence is reached. Set to 0 to disable early stopping.",
        default_value=5,
        min_value=0,
        is_advanced=True,
    )


@knext.node(
    name="Auto-SARIMA Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/TimeSeriesLearner.png",
    category=timeseries_analysis_category,
    keywords=[
        "ARIMA",
        "SARIMA",
        "Time Series",
        "Forecasting",
        "Seasonal",
        "Autoregressive",
    ],
    id="auto_sarima_learner",
)
@knext.input_table(
    name="Input Data",
    description="Time series data table for training the Auto-SARIMA model. The selected target column must be numeric and contain no missing values.",
)
@knext.output_table(
    name="In-sample Predictions and Residuals",
    description="Model fit quality assessment table containing original values, fitted predictions, residuals, and absolute errors. Initial unstable predictions (first max(2 * seasonal period, 10) observations; 10 for non-seasonal models) are excluded to ensure reliable performance metrics.",
)
@knext.output_table(
    name="Coefficients and Statistics",
    description="Complete model summary including optimal SARIMA parameters (p,d,q)(P,D,Q,s), fitted coefficients with standard errors, and goodness-of-fit statistics (AIC, BIC, Log Likelihood, MSE, MAE).",
)
@knext.output_table(
    name="Residual Diagnostics",
    description="Statistical tests for model adequacy assessment: Ljung-Box test for autocorrelation, Jarque-Bera test for normality (large samples), and Shapiro-Wilk test for normality (small-medium samples).",
)
@knext.output_table(
    name="Optimization History",
    description="Detailed optimization transparency table showing all parameter combinations tested during simulated annealing search, including information criteria values (AIC/BIC/HQIC), log likelihood, and fitting status for each attempt.",
)
@knext.output_binary(
    name="Model",
    description="Trained SARIMA model object (pickled SARIMAXResults) ready for forecasting. Can be used with the Auto-SARIMA Predictor node to generate forecasts of any length without model retraining.",
    id="auto_sarima.model",
)
class AutoSarimaLearner:
    """

    Automatically identifies optimal parameters and trains a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model for time series forecasting. Uses simulated annealing optimization to find the best-fitting model configuration.
    **Note:** For non-seasonal data, set seasonal period to 0 to fit standard ARIMA models. This is the default option.

    ## Model Components

    **SARIMA(p,d,q)(P,D,Q,s)** models capture both short-term and seasonal patterns in time series data:

    - **AR (p):** AutoRegressive component - models dependence on `p` previous observations
    - **I (d):** Integration order - degree of non-seasonal differencing needed for stationarity
    - **MA (q):** Moving Average component - models dependence on `q` previous forecast errors
    - **Seasonal AR (P):** Seasonal autoregressive component at lag `s`
    - **Seasonal I (D):** Seasonal integration order - seasonal differencing for stationarity
    - **Seasonal MA (Q):** Seasonal moving average component at lag `s`
    - **Period (s):** Length of seasonal cycle (e.g., 12 for monthly data with yearly patterns)

    ## Optimization Strategy

    **Two-Phase Parameter Search:**

    1. **Stationarity Testing (d, D):** Uses Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests to automatically
       determine required differencing orders. Applies differencing until series achieves stationarity
       (p-value ≥ 0.05) or reaches user-defined maximum orders.

    2. **Simulated Annealing (p, q, P, Q):** Intelligently explores parameter combinations within user constraints
       to minimize the selected information criterion (AIC/BIC/HQIC). The annealing process allows escaping
       local optima by occasionally accepting worse solutions early in optimization, then becoming increasingly
       selective as the "temperature" decreases.

    ## Configuration Options

    - **Target Column:** Numeric time series variable (no missing values allowed)
    - **Seasonal Period:** Cycle length (12=monthly/yearly, 7=daily/weekly, 0=no seasonality), For long-term seasonal patterns (periods >100), consider using Fourier Transform or STL decomposition to pre-process and remove seasonality before SARIMA modeling, as large seasonal periods cause severe computational bottlenecks, excessive memory consumption, and potential numerical instability.
    - **Log Transformation:** Optional variance stabilization (requires positive values)
    - **Parameter Bounds:** Maximum search ranges for all SARIMA components
    - **Information Criterion:** AIC (prediction focus), BIC (parsimony focus), or HQIC (balanced)
    - **Optimization Control:** Annealing steps, MCMC iterations, temperature schedule, early stopping

    ## Model Outputs

    1. **Predictions & Residuals:** In-sample fit assessment with quality metrics
    2. **Model Statistics:** Optimal parameters, coefficients, standard errors, and fit statistics
    3. **Diagnostic Tests:** Residual autocorrelation and normality assessments
    4. **Optimization History:** Complete search transparency showing all tested parameter combinations
    5. **Trained Model:** Ready-to-use model object for forecasting with the Auto-SARIMA Predictor

    ## Use Cases

    - **Economic Forecasting:** GDP, inflation, unemployment with seasonal patterns
    - **Sales Analytics:** Revenue, demand planning with monthly/quarterly cycles
    - **Operations Research:** Inventory management, capacity planning with known seasonality
    - **Environmental Science:** Temperature, precipitation with annual cycles
    - **Web Analytics:** Traffic patterns with daily/weekly seasonality

    """

    # Constants for diagnostic calculations
    DEFAULT_SKIP_OBSERVATIONS = 10
    DEFAULT_LJUNG_BOX_LAGS = 10

    # General settings for the SARIMA model
    input_column = knext.ColumnParameter(
        label="Target Column",
        description="Numeric time series column for model training. Must contain no missing values. This is the variable the model will learn to predict and forecast.",
        port_index=0,
        column_filter=kutil.is_numeric,
    )
    seasonal_period_param = knext.IntParameter(
        label="Seasonal Period (s)",
        description="""Specify the length of the seasonal period for your time series data. This parameter determines how many observations constitute one complete seasonal cycle:

**Common Seasonal Periods by Data Granularity:**
• **Monthly data with yearly seasonality**: Set to 12 (12 months = 1 year)
• **Weekly data with yearly seasonality**: Set to 52 (52 weeks = 1 year)
• **Daily data with weekly seasonality**: Set to 7 (7 days = 1 week)
• **Hourly data with daily seasonality**: Set to 24 (24 hours = 1 day)
• **Quarterly data with yearly seasonality**: Set to 4 (4 quarters = 1 year)
• **Minute data with hourly seasonality**: Set to 60 (60 minutes = 1 hour)

**⚠️ AVOID Large Seasonal Periods (>100):**
• **Daily data with yearly seasonality (s=365)**: ❌ Causes severe computational issues
• **Hourly data with weekly patterns (s=168)**: ❌ Extremely slow fitting
• **Instead**: Use data aggregation (daily→weekly, hourly→daily) or remove seasonality first

**Recommended Approach for Beginners:**
1. **Start with s=0** (ARIMA) to understand basic time series patterns
2. **Add seasonality gradually** only if clear seasonal patterns exist

**Guidelines for Selection:**
- Examine your data visually or with autocorrelation plots to identify repeating patterns
- The seasonal period should match the frequency of recurring patterns in your data
- For multiple seasonalities (e.g., daily + yearly), choose the most prominent one
- **Set to 0** to disable seasonal components and fit a standard ARIMA model first

**Example**: If analyzing monthly sales data that peaks every December, use seasonal period = 12 to capture the yearly pattern.""",
        default_value=0,
        min_value=0,
    )
    natural_log = knext.BoolParameter(
        label="Log-transform data for modelling",
        description="Apply natural logarithm transformation before modeling to stabilize variance (requires all values > 0). Fitted values are back-transformed in the in-sample output; residuals and residual-based diagnostics remain on the log scale.",
        default_value=False,
    )

    # The parameters constraints for the automatic ARIMA model
    non_seasonal_params = NonSeasonalParams()
    seasonal_params = SeasonalParams()

    # Model selection parameters
    model_selection_params = ModelSelectionParams()

    # Optimization loop parameters (Advanced)
    optimization_loop_params = OptimizationLoopParams()

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema) -> knext.Schema:
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.input_column,
            input_schema,
            kutil.is_numeric,
        )

        if self.optimization_loop_params.beta0 >= self.optimization_loop_params.beta1:
            raise knext.InvalidParametersError("The initial annealing temperature (beta0) must be less than the final annealing temperature (beta1).")

        # Enhanced predictions table schema (removed Index column)
        predictions_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double(), knext.double()], ["Original Value", "Fitted Value", "Residual", "Absolute Error"]
        )

        # Enhanced model summary schema (3 columns with explanations)
        model_summary_schema = knext.Schema([knext.string(), knext.double(), knext.string()], ["Parameter", "Value", "Explanation"])

        # Diagnostics schema (unchanged)
        diagnostics_schema = knext.Schema(
            [knext.string(), knext.double(), knext.double(), knext.string()], ["Test", "Statistic", "P-Value", "Interpretation"]
        )

        # Optimization history schema
        optimization_history_schema = knext.Schema(
            [knext.string(), knext.double(), knext.double(), knext.double(), knext.double(), knext.string()],
            ["ARIMA_Parameters", "AIC", "BIC", "HQIC", "Log_Likelihood", "Status"],
        )

        binary_model_schema = knext.BinaryPortObjectSpec("auto_sarima.model")

        return (
            predictions_schema,
            model_summary_schema,
            diagnostics_schema,
            optimization_history_schema,
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input: knext.Table):
        # Import heavy dependencies
        import pandas as pd
        import numpy as np
        from statsmodels.tsa.stattools import kpss
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        import pickle
        import warnings

        # Initialize optimization history at start of execution
        self.optimization_history = []

        df = input.to_pandas()
        target_col = df[self.input_column]

        # if enabled, apply natural logarithm transformation. If negative values are present, raise an error
        if self.natural_log:
            num_negative_vals = kutil.count_negative_values(target_col)

            if num_negative_vals > 0:
                raise knext.InvalidParametersError(f" There are '{num_negative_vals}' non-positive values in the target column.")
            target_col = np.log(target_col)

        exec_context.set_progress(0.1)

        # check if the number of obsevations is greater than or equal to twice the seasonal period
        if len(target_col) < 2 * self.seasonal_period_param:
            LOGGER.warning(
                f"The number of observations in the target column ({len(target_col)}) is lower than twice the seasonal period ({self.seasonal_period_param}). This may lead to unreliable model estimates and forecasts."
            )
            exec_context.set_warning(
                f"The number of observations in the target column ({len(target_col)}) is lower than twice the seasonal period ({self.seasonal_period_param}). This may lead to unreliable model estimates and forecasts."
            )
            if len(target_col) < self.seasonal_period_param:
                raise knext.InvalidParametersError(
                    f"The number of observations in the target column ({len(target_col)}) is lower than twice the seasonal period ({self.seasonal_period_param})."
                )

        # Check for missing values
        kutil.validate_missing_values(target_col)

        # Add performance warning for large seasonal periods
        kutil.seasonality_performance_warning(
            context=exec_context,
            LOGGER=LOGGER,
            seasonality=self.seasonal_period_param,
            seasonality_warning_threshold=100,
        )

        exec_context.set_progress(0.2)

        best_params = self.__params_optimization_loop(
            target_col,
            exec_context=exec_context,
            np=np,
            kpss=kpss,
            SARIMAX=SARIMAX,
            warnings=warnings,
            ConvergenceWarning=ConvergenceWarning,
        )

        exec_context.set_progress(0.8)

        trained_model = self.__evaluate_arima_model(
            target_col,
            best_params,
            exec_context=exec_context,
            SARIMAX=SARIMAX,
            warnings=warnings,
            np=np,
            ConvergenceWarning=ConvergenceWarning,
            track_history=False,  # Don't track history for final model fitting
        )[1]

        # Check if final model fitting succeeded
        if trained_model is None:
            raise knext.InvalidParametersError(
                f"Final model fitting failed with optimal parameters {best_params}. "
                f"This can happen with:\n"
                f"1. Very large seasonal periods (e.g., 365) that cause numerical issues\n"
                f"2. Insufficient data relative to model complexity\n"
                f"3. Non-stationary data that cannot be modeled with current parameters\n"
                f"Try: reducing seasonal period, increasing data size, or adjusting parameter constraints."
            )

        exec_context.set_progress(0.9)

        # Create enhanced predictions table with original values
        enhanced_predictions = kutil.enhance_predictions_table(
            trained_model, input, self.input_column, self.seasonal_period_param, self.DEFAULT_SKIP_OBSERVATIONS, pd
        )

        # Apply log transformation reverse if needed
        if self.natural_log:
            if "Fitted Value" in enhanced_predictions.columns:
                enhanced_predictions["Fitted Value"] = np.exp(enhanced_predictions["Fitted Value"])

        # populate model coefficients and statistics with enhanced formatting
        coeffs_and_stats = self.__get_coeffs_and_stats(trained_model, best_params, pd)

        residual_diagnostics = kutil.compute_residual_diagnostics(
            trained_model, self.seasonal_period_param, self.DEFAULT_SKIP_OBSERVATIONS, self.DEFAULT_LJUNG_BOX_LAGS, pd
        )

        # generate optimization history table
        optimization_history = self.__get_optimization_history_table(pd)

        model_binary = pickle.dumps(trained_model)

        exec_context.set_progress(0.99)

        return (
            knext.Table.from_pandas(enhanced_predictions, row_ids="keep"),
            knext.Table.from_pandas(coeffs_and_stats, row_ids="keep"),
            knext.Table.from_pandas(residual_diagnostics, row_ids="keep"),
            knext.Table.from_pandas(optimization_history, row_ids="keep"),
            model_binary,
        )

    def __validate_params(self, column, p, q, P, Q):
        """
        Validates the proposed (S)ARIMA parameters against the time series data length and potential overlaps.

        This function performs two checks:
        1. Ensures the time series (`column`) has enough data points to estimate the model given the highest order parameter (p, q, S*P, S*Q).
        2. Checks for invalid parameter combinations where seasonal and non-seasonal AR or MA terms might overlap (e.g., p >= S when P > 0, or q >= S when Q > 0).

        Parameters:
        - column: pd.Series
            The time series data.
        - p: int
            The non-seasonal AR order.
        - q: int
            The non-seasonal MA order.
        - P: int
            The seasonal AR order.
        - Q: int
            The seasonal MA order.

        Returns:
        - bool
            True if the parameters are valid for the given series, False otherwise.
        """
        S = self.seasonal_period_param
        set_val = set([p, q, S * P, S * Q])
        num_of_rows = kutil.number_of_rows(column)

        if num_of_rows < max(set_val):
            return False

        # Parameter overlap validation - clearer logic for seasonality handling
        if S == 0:
            # ARIMA mode: no seasonal parameters allowed
            if P > 0 or Q > 0:
                return False
        else:
            # SARIMA mode: prevent overlap between seasonal and non-seasonal terms
            if (P > 0 and p >= S) or (Q > 0 and q >= S):
                return False

        return True

    def __get_coeffs_and_stats(self, model, best_params, pd):
        """
        Compiles comprehensive model summary with parameters, coefficients, and fit statistics.

        Creates a detailed table containing the optimal SARIMA parameters, all model coefficients
        with their standard errors. Key goodness-of-fit metrics are obtained from utility get_model_stats.
        Each entry includes explanatory text to help interpret the results.

        Parameters:
        - model: statsmodels.tsa.statespace.sarimax.SARIMAXResults
            Fitted SARIMA model containing coefficients and statistics.
        - best_params: dict
            Optimal parameters: {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.

        Returns:
        - pd.DataFrame
            Model summary table with parameters, coefficients, and interpretive explanations.
        """
        # Create the data structure
        data = []

        # Best model parameters first (top of table)
        data.append(
            {"Parameter": "p (AR Order)", "Value": float(best_params["p"]), "Explanation": "Number of autoregressive terms (lagged observations)."}
        )
        data.append(
            {
                "Parameter": "d (Differencing Order)",
                "Value": float(best_params["d"]),
                "Explanation": "Number of non-seasonal differences needed to make series stationary.",
            }
        )
        data.append(
            {
                "Parameter": "q (MA Order)",
                "Value": float(best_params["q"]),
                "Explanation": "Number of lagged forecast errors in the prediction equation.",
            }
        )
        data.append(
            {"Parameter": "P (Seasonal AR Order)", "Value": float(best_params["P"]), "Explanation": "Number of seasonal autoregressive terms."}
        )
        data.append(
            {
                "Parameter": "D (Seasonal Differencing Order)",
                "Value": float(best_params["D"]),
                "Explanation": "Number of seasonal differences needed for stationarity.",
            }
        )
        data.append(
            {"Parameter": "Q (Seasonal MA Order)", "Value": float(best_params["Q"]), "Explanation": "Number of seasonal lagged forecast errors."}
        )

        # Model coefficients and standard errors
        if hasattr(model, "params") and len(model.params) > 0:
            for param_name, coeff_val in model.params.items():
                data.append(
                    {
                        "Parameter": f"{param_name} (Coefficient)",
                        "Value": float(coeff_val),
                        "Explanation": "Model coefficient representing the relationship strength.",
                    }
                )

                # Add standard error if available
                if hasattr(model, "bse") and param_name in model.bse:
                    data.append(
                        {
                            "Parameter": f"{param_name} (Std. Error)",
                            "Value": float(model.bse[param_name]),
                            "Explanation": "Standard error of the coefficient estimate.",
                        }
                    )

        return kutil.get_model_stats(model, data)

    def __get_optimization_history_table(self, pd):
        """
        Creates a DataFrame containing the complete optimization history with all tested models.

        Parameters:
        - pd: pandas module
            Pandas module for DataFrame creation.

        Returns:
        - pd.DataFrame
            DataFrame containing all tested parameter combinations and their metrics.
        """
        if not self.optimization_history:
            # Return empty DataFrame with correct schema if no history
            empty_df = pd.DataFrame(columns=["ARIMA_Parameters", "AIC", "BIC", "HQIC", "Log_Likelihood", "Status"])
            # Set proper column types to match schema
            empty_df["ARIMA_Parameters"] = empty_df["ARIMA_Parameters"].astype("string")
            empty_df["AIC"] = empty_df["AIC"].astype("float64")
            empty_df["BIC"] = empty_df["BIC"].astype("float64")
            empty_df["HQIC"] = empty_df["HQIC"].astype("float64")
            empty_df["Log_Likelihood"] = empty_df["Log_Likelihood"].astype("float64")
            empty_df["Status"] = empty_df["Status"].astype("string")
            return empty_df

        # Create DataFrame from optimization history
        history_df = pd.DataFrame(self.optimization_history)

        # Ensure proper column order and types
        expected_columns = ["ARIMA_Parameters", "AIC", "BIC", "HQIC", "Log_Likelihood", "Status"]
        if list(history_df.columns) != expected_columns:
            # Reorder columns to match expected order
            history_df = history_df[expected_columns]

        # Ensure proper data types that match KNIME schema
        history_df["ARIMA_Parameters"] = history_df["ARIMA_Parameters"].astype("string")

        for col in ["AIC", "BIC", "HQIC", "Log_Likelihood"]:
            history_df[col] = history_df[col].astype("float64")

        history_df["Status"] = history_df["Status"].astype("string")

        # Keep insertion order (chronological order is preserved by list append order)

        return history_df

    def __find_optimal_integration_params(
        self,
        series,
        kpss,
    ):
        """
        Determines the optimal orders of non-seasonal (d) and seasonal (D) differencing required to make the time series stationary using the KPSS test.

        The function iteratively applies seasonal differencing (up to `max_i_s` times) and then non-seasonal differencing (up to `max_i` times).
        In each step, it performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. The null hypothesis of the KPSS test is that the series is stationary around a deterministic trend.
        Differencing continues as long as the p-value of the KPSS test is below the significance level `alpha`, indicating non-stationarity. The number of differencing steps taken determines the values of D and d.
        Note: This function modifies the input `series` by applying differencing directly.

        Parameters:
        - series: pd.Series
            The input time series data. This series will be modified in place.
        - seasonality: int
            The seasonal period of the time series. Used for seasonal differencing.
        - max_i: int, optional (default=2)
            The maximum order of non-seasonal differencing (d) to test.
        - max_i_s: int, optional (default=2)
            The maximum order of seasonal differencing (D) to test.
        - alpha: float, optional (default=0.05)
            The significance level for the KPSS test. If the p-value is greater than or equal to alpha, the series is considered stationary.

        Returns:
        - tuple (int, int)
            A tuple containing the determined optimal non-seasonal differencing order (d) and seasonal differencing order (D).
        """
        # Initialize d and D parameters
        d = 0
        D = 0

        # significance level for KPSS test
        alpha = 0.05

        seasonality = self.seasonal_period_param

        # Check for seasonal stationarity (D parameter) - only if seasonality > 0
        if seasonality > 0:
            for _ in range(self.seasonal_params.max_s_i):
                kpss_result = kpss(series)
                p_value_d_s = kpss_result[1]  # Fixed tuple unpacking
                if p_value_d_s >= alpha:
                    break
                # Apply seasonal differencing
                series = series.diff(seasonality).dropna()
                D += 1

        # Check for trend stationarity (d parameter)
        for _ in range(self.non_seasonal_params.max_i):
            kpss_result = kpss(series)
            p_value_d = kpss_result[1]  # Fixed tuple unpacking
            if p_value_d >= alpha:
                break
            # Apply non seasonal differencing
            series = series.diff().dropna()
            d += 1

        return d, D

    def __propose_initial_params(self, d, D):
        """
        Proposes initial simple parameters for the SARIMA model optimization process.

        Given the pre-determined differencing orders (d, D), this function sets the initial
        AR and MA orders (p, q, P, Q) to 1, unless the corresponding maximum allowed value
        (max_p, max_q, max_p_s, max_q_s) is 0. This provides a basic starting point for the
        simulated annealing optimization.

        Parameters:
        - d: int
            The non-seasonal differencing order.
        - D: int
            The seasonal differencing order.
        - max_p: int, optional (default=3)
            The maximum allowed non-seasonal AR order.
        - max_q: int, optional (default=3)
            The maximum allowed non-seasonal MA order.
        - max_p_s: int, optional (default=5)
            The maximum allowed seasonal AR order.
        - max_q_s: int, optional (default=5)
            The maximum allowed seasonal MA order.

        Returns:
        - dict
            A dictionary containing the initial proposed parameters:
            {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.
        """
        p = min([self.non_seasonal_params.max_ar, 1])
        q = min([self.non_seasonal_params.max_ma, 1])

        # Only propose seasonal parameters if seasonality > 0
        if self.seasonal_period_param > 0:
            P = min([self.seasonal_params.max_s_ar, 1])
            Q = min([self.seasonal_params.max_s_ma, 1])
        else:
            P = 0  # Force to 0 for ARIMA mode
            Q = 0

        return {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q}

    def __propose_new_params(
        self,
        series,
        current_params,
        exec_context: knext.ExecutionContext,
        np,
    ):
        """
        Proposes a new set of SARIMA parameters by randomly adjusting the current parameters.

        This function is used within the simulated annealing loop. It takes the current set of
        parameters and randomly decides whether to adjust the non-seasonal (p, q) or seasonal (P, Q)
        orders based on a random threshold. The chosen orders are incremented or decremented by `step_size`
        (randomly chosen direction) or kept the same. The new parameters are constrained within the allowed maximums
        (max_p, max_q, max_p_s, max_q_s) and non-negativity.

        ASYMMETRIC CONSTRAINT HANDLING:
        The function supports asymmetric seasonal constraints where only P or only Q is allowed:
        - If max_p_s > 0 and max_q_s = 0: Only seasonal AR terms are updated, Q remains 0
        - If max_p_s = 0 and max_q_s > 0: Only seasonal MA terms are updated, P remains 0
        - If both > 0: Both P and Q can be updated
        - If both = 0: Only non-seasonal parameters (p, q) are updated

        Finally, it validates the proposed parameters using `__validate_params`. If the proposed parameters
        are invalid, it logs a warning and returns the original `current_params`.

        Parameters:
        - series: pd.Series
            The time series data, used for validation via `__validate_params`.
        - current_params: dict
            A dictionary containing the current parameters: {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.
        - exec_context: knext.ExecutionContext
            The execution context for logging warnings if validation fails.
        - max_p: int, optional (default=5)
            The maximum allowed non-seasonal AR order.
        - max_q: int, optional (default=5)
            The maximum allowed non-seasonal MA order.
        - max_p_s: int, optional (default=3)
            The maximum allowed seasonal AR order.
        - max_q_s: int, optional (default=3)
            The maximum allowed seasonal MA order.

        Returns:
        - dict
            A dictionary containing the newly proposed (and validated) parameters, or the `current_params`
            if the proposed ones were invalid.

        Raises:
        - knext.InvalidParametersError
            If the internal logic fails to update any parameters (should not typically happen with the current logic).
        """
        updated_params = current_params.copy()
        threshold = np.random.rand()

        step_size = self.optimization_loop_params.step_size
        max_p = self.non_seasonal_params.max_ar
        max_q = self.non_seasonal_params.max_ma
        max_p_s = self.seasonal_params.max_s_ar
        max_q_s = self.seasonal_params.max_s_ma

        steps = np.arange(1, step_size + 1)

        if threshold <= (1 / 2) or (max_p_s == 0 and max_q_s == 0):  # update trend parameters (p, q)
            current_p, current_q = updated_params["p"], updated_params["q"]

            new_p, new_q = (
                current_p + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0]),
                current_q + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0]),
            )

            new_p = max([min([max_p, new_p]), 0])
            new_q = max([min([max_q, new_q]), 0])

            updated_params["p"], updated_params["q"] = new_p, new_q

        elif threshold > (1 / 2) and (max_p_s > 0 or max_q_s > 0):  # update seasonal parameters (P, Q) if at least one is allowed (fixed: AND -> OR)
            """
            FIXED LOGIC: Changed from (max_p_s > 0 and max_q_s > 0) to (max_p_s > 0 or max_q_s > 0)
            <br>
            Problem: Original logic required BOTH P and Q to be allowed for any seasonal updates.
            This failed when user sets asymmetric constraints like:
            -Max Seasonal AR (P): 1 ✅ (allows seasonal AR)
            -Max Seasonal MA (Q): 0 ❌ (disallows seasonal MA)
            Solution: Now updates seasonal parameters if AT LEAST ONE is allowed.
            Only modifies parameters that are actually permitted by user constraints.
            """
            current_ps, current_qs = updated_params["P"], updated_params["Q"]

            # Only update P if max_p_s > 0 (user allows seasonal AR terms)
            if max_p_s > 0:
                new_ps = current_ps + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0])
                new_ps = max(min(max_p_s, new_ps), 0)  # Constrain within [0, max_p_s]
                updated_params["P"] = new_ps
            # If max_p_s = 0, P remains unchanged at 0

            # Only update Q if max_q_s > 0 (user allows seasonal MA terms)
            if max_q_s > 0:
                new_qs = current_qs + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0])
                new_qs = max(min(max_q_s, new_qs), 0)  # Constrain within [0, max_q_s]
                updated_params["Q"] = new_qs
            # If max_q_s = 0, Q remains unchanged at 0

        else:
            raise knext.InvalidParametersError(
                f"No parameters were updated because the conditions for updating parameters were not met. Threshold: {threshold}, constraints: {max_p_s}, {max_q_s}."
            )

        if self.__validate_params(
            series,
            updated_params["p"],
            updated_params["q"],
            updated_params["P"],
            updated_params["Q"],
        ):
            return updated_params
        else:
            LOGGER.info(f"Proposed parameters {updated_params} are invalid for the series. Retaining current parameters.")
            return current_params

    def __evaluate_arima_model(
        self,
        series,
        params,
        exec_context: knext.ExecutionContext,
        SARIMAX,
        warnings,
        np,
        ConvergenceWarning,
        track_history=True,
    ):
        """
        Fits a SARIMAX model with the given parameters and evaluates its information criterion score.

        This function attempts to fit a `statsmodels.tsa.statespace.sarimax.SARIMAX` model
        using the provided time series, parameter dictionary, and seasonality. It captures
        the selected information criterion (AIC, BIC, or HQIC) as the primary evaluation metric (lower is better).
        It specifically checks for `ConvergenceWarning` during fitting using `warnings.catch_warnings`.
        If a `ConvergenceWarning` occurs or any other exception is raised during fitting, it logs a
        warning via the `exec_context` and returns an infinite score and the potentially partially
        fitted model object (or None if fitting failed early). This heavily penalizes problematic
        parameters in the optimization process. Successful fits are also logged with their score.

        Parameters:
        - series: pd.Series
            The time series data to fit the model on.
        - params: dict
            A dictionary containing the SARIMA parameters: {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.
        - seasonality: int
            The seasonal period for the SARIMA model.
        - exec_context: knext.ExecutionContext
            The execution context for logging warnings about fitting issues or convergence.

        Returns:
        - tuple (float, statsmodels.tsa.statespace.sarimax.SARIMAXResults or None)
            A tuple containing:
            - float: The information criterion score of the fitted model. Returns `np.inf` if fitting fails or results in a ConvergenceWarning.
            - SARIMAXResults or None: The fitted model object if successful, otherwise None or the object from a failed/non-converged fit.
        """
        ic_score = np.inf
        convergence_warning_occurred = False
        model_fit = None  # Initialize model_fit to handle potential early exceptions

        seasonality = self.seasonal_period_param

        # Get the selected criterion
        criterion = self.model_selection_params.selection_criterion
        criterion_value = ModelSelectionParams.SelectionCriteria[criterion].value[0].lower()

        try:
            # Use catch_warnings to capture any warnings during fit()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")  # Ensure all warnings are caught

                model = SARIMAX(
                    endog=series,
                    order=(params["p"], params["d"], params["q"]),
                    seasonal_order=(params["P"], params["D"], params["Q"], seasonality),
                )
                model_fit = model.fit(disp=False)

                # Check for degenerate log-likelihood (exclude models with log-likelihood ≈ 0)
                if abs(model_fit.llf) < 1e-6:
                    LOGGER.info(f"Rejecting model with degenerate log-likelihood: {model_fit.llf} for params {params}")
                    ic_score = np.inf
                else:
                    # Get the appropriate information criterion
                    ic_score = model_fit.aic  # default
                    if criterion_value == "bic":
                        ic_score = model_fit.bic
                    if criterion_value == "hqic":
                        ic_score = model_fit.hqic

                # Check if any ConvergenceWarning was caught
                for warning in caught_warnings:
                    if issubclass(warning.category, ConvergenceWarning):
                        convergence_warning_occurred = True
                        break

        except Exception as e:
            # Handle potential errors during model fitting (e.g., invalid parameters)
            LOGGER.warning(f"Model fitting failed for params {params}: {str(e)}")
            exec_context.set_warning(
                f"WARNING: SARIMAX model fitting failed with parameters {params}. "
                f"Error: {str(e)}. This may indicate numerical issues with large seasonal periods "
                f"or insufficient data for the model complexity."
            )

            # Store optimization history for completely failed models
            if track_history:
                seasonality = self.seasonal_period_param
                arima_notation = f"({params['p']},{params['d']},{params['q']})({params['P']},{params['D']},{params['Q']},{seasonality})"
                self.optimization_history.append(
                    {
                        "ARIMA_Parameters": str(arima_notation),
                        "AIC": float(np.inf),
                        "BIC": float(np.inf),
                        "HQIC": float(np.inf),
                        "Log_Likelihood": float(np.inf),
                        "Status": str(f"Fitting Failed: {str(e)[:100]}"),
                    }
                )

            return (
                np.inf,
                None,  # Explicitly return None when fitting fails completely
            )  # Treat errors as convergence issues or high cost

        if convergence_warning_occurred:
            # return infinity to make impossible to accept these parameters
            LOGGER.info(f"ConvergenceWarning occurred for params {params}. Model fitting failed, returning infinity.")

            # Store optimization history even for failed models
            if track_history:
                seasonality = self.seasonal_period_param
                arima_notation = f"({params['p']},{params['d']},{params['q']})({params['P']},{params['D']},{params['Q']},{seasonality})"
                self.optimization_history.append(
                    {
                        "ARIMA_Parameters": str(arima_notation),
                        "AIC": float(np.inf),
                        "BIC": float(np.inf),
                        "HQIC": float(np.inf),
                        "Log_Likelihood": float(np.inf),
                        "Status": str("Convergence Failed"),
                    }
                )

            return (
                np.inf,
                model_fit,
            )  # Return the model_fit object even if convergence failed

        LOGGER.info(f"Model fitted successfully for params {params}, {criterion_value.upper()}: {ic_score}")

        # Store optimization history for successful models
        if track_history:
            seasonality = self.seasonal_period_param
            arima_notation = f"({params['p']},{params['d']},{params['q']})({params['P']},{params['D']},{params['Q']},{seasonality})"
            self.optimization_history.append(
                {
                    "ARIMA_Parameters": str(arima_notation),
                    "AIC": float(model_fit.aic),
                    "BIC": float(model_fit.bic),
                    "HQIC": float(model_fit.hqic),
                    "Log_Likelihood": float(model_fit.llf),
                    "Status": str("Success"),
                }
            )

        return (ic_score, model_fit)

    def __accept_new_params(self, delta_cost, beta, np):
        """
        Decides whether to accept a new set of parameters based on the change in cost (AIC) and the current annealing temperature (beta).

        This function implements the Metropolis acceptance criterion used in simulated annealing:
        1. If the new parameters result in a lower or equal cost (`delta_cost` <= 0), they are always accepted.
        2. If the new parameters result in a higher cost (`delta_cost` > 0):
           - If `beta` is infinite (representing the final stage of annealing where only improvements are accepted), the move is rejected as in a greedy algorithm.
           - Otherwise, the move is accepted probabilistically based on the Boltzmann factor `exp(-beta * delta_cost)`. A random number is drawn from [0, 1); if it's less than the Boltzmann factor, the move is accepted. This allows the algorithm to occasionally escape local optima early in the process when beta is low (high temperature).

        Parameters:
        - delta_cost: float
            The change in the cost function (AIC) between the new parameters and the current parameters (new_cost - current_cost).
        - beta: float
            The inverse temperature parameter in the simulated annealing process. Higher beta means lower tolerance for accepting worse solutions. Can be `np.inf`.

        Returns:
        - bool
            True if the new parameters should be accepted, False otherwise.
        """
        # If the cost doesn't increase, we always accept
        if delta_cost <= 0:
            return True

        # If the cost increases and beta is infinite (last iteration), we always reject. Explicitly check delta_cost > 0 for clarity
        elif (beta == np.inf) and (delta_cost > 0):
            return False

        # In all other cases (delta_cost > 0 and beta < inf), accept based on probability p compared to a random draw from [0, 1)
        else:
            p = np.exp(-beta * delta_cost)
            return np.random.rand() < p

    def __params_optimization_loop(
        self,
        series,
        exec_context: knext.ExecutionContext,
        np,
        kpss,
        SARIMAX,
        warnings,
        ConvergenceWarning,
    ):
        """
        Optimizes SARIMA parameters using intelligent simulated annealing search.

        Automatically determines optimal model parameters through a sophisticated two-phase approach:
        stationarity testing (d,D) followed by simulated annealing optimization (p,q,P,Q). The
        algorithm balances exploration and exploitation to find globally optimal solutions.

        The algorithm works as follows:
        1. Determine initial d and D using KPSS tests (`find_optimal_integration_params`).
        2. Propose simple initial parameters (p=1, q=1, P=1, Q=1, respecting constraints) using `propose_initial_params`.
        3. Evaluate the initial model's AIC score (`evaluate_arima_model`).
        4. Initialize the best parameters and cost found so far.
        5. Start the simulated annealing loop:
           - Iterate through a schedule of inverse temperatures (`beta`), starting low (`beta0`) and increasing to high (`beta1`), ending with infinity.
           - For each `beta`, run a Markov Chain Monte Carlo (MCMC) simulation for `mcmc_steps`:
             - Propose new parameters (p, q, P, Q) by slightly modifying the current ones (`propose_new_params`).
             - Evaluate the AIC score of the model with the proposed parameters (`evaluate_arima_model`).
             - Calculate the change in cost (`delta_cost`).
             - Decide whether to accept the proposed parameters using the Metropolis criterion (`accept_new_params`).
             - If accepted, update the current parameters and cost.
             - If the accepted parameters yield the best cost seen so far, update the best parameters and cost.
           - Report progress and results for the current `beta`
        6. Return the best set of parameters found throughout the process.

        Parameters:
        - series: pd.Series
            The input time series data.
        - seasonality: int
            The seasonal period of the time series.
        - exec_context: knext.ExecutionContext
            The execution context for progress reporting and logging warnings.
        - anneal_steps: int, optional (default=5)
            The number of different temperature levels (betas) in the annealing schedule.
        - mcmc_steps: int, optional (default=10)
            The number of MCMC steps (parameter proposals) to perform at each temperature level.
        - beta0: float, optional (default=0.1)
            The initial (lowest) inverse temperature. Corresponds to high tolerance for accepting worse solutions.
        - beta1: float, optional (default=10.0)
            The final (highest) finite inverse temperature before switching to infinity. Corresponds to low tolerance.

        Returns:
        - dict
            A dictionary containing the best set of SARIMA parameters found:
            {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.

        Raises:
        - RuntimeError: If the initial model evaluation fails even after trying a fallback simple model.
        """
        anneal_steps = self.optimization_loop_params.anneal_steps
        mcmc_steps = self.optimization_loop_params.mcmc_steps
        beta0 = self.optimization_loop_params.beta0
        beta1 = self.optimization_loop_params.beta1

        # Set up the list of betas.
        beta_list = np.zeros(anneal_steps)
        # All but the last one are evenly spaced between beta0 and beta1 (included)
        beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
        # The last one is set to infinty
        beta_list[-1] = np.inf
        # Set up the progress bar
        progress = np.linspace(0.2, 0.8, anneal_steps)

        if len(progress) != len(beta_list):
            raise knext.InvalidParametersError("The number of progress steps must match the number of beta values.")

        LOGGER.info("Preparing to find optimal integration parameters...")
        # Create a copy to avoid modifying the original series outside this function scope
        series_copy_for_diff = series.copy()
        d, D = self.__find_optimal_integration_params(series_copy_for_diff, kpss)
        LOGGER.info(f"Optimal integration parameters found: d={d}, D={D}. Generating the remaining initial parameters...")
        current_params = self.__propose_initial_params(d, D)
        LOGGER.info(f"Initial parameters: {current_params}. Evaluating initial model based on the AIC...")
        current_cost = self.__evaluate_arima_model(
            series,
            current_params,
            exec_context=exec_context,
            SARIMAX=SARIMAX,
            warnings=warnings,
            np=np,
            ConvergenceWarning=ConvergenceWarning,
        )[0]

        # Handle case where initial evaluation fails
        if current_cost == np.inf:
            # Try a smarter fallback based on user constraints
            if self.non_seasonal_params.max_ar > 0:
                fallback_params = {"p": 1, "d": d, "q": 0, "P": 0, "D": D, "Q": 0}
                LOGGER.info(f"Initial model evaluation failed (AIC=inf) for params {current_params}. Trying AR(1) fallback model...")
            elif self.non_seasonal_params.max_ma > 0:
                fallback_params = {"p": 0, "d": d, "q": 1, "P": 0, "D": D, "Q": 0}
                LOGGER.info(f"Initial model evaluation failed (AIC=inf) for params {current_params}. Trying MA(1) fallback model...")
            elif self.seasonal_params.max_s_ar > 0:
                fallback_params = {"p": 0, "d": d, "q": 0, "P": 1, "D": D, "Q": 0}
                LOGGER.info(f"Initial model evaluation failed (AIC=inf) for params {current_params}. Trying SAR(1) fallback model...")
            elif self.seasonal_params.max_s_ma > 0:
                fallback_params = {"p": 0, "d": d, "q": 0, "P": 0, "D": D, "Q": 1}
                LOGGER.info(f"Initial model evaluation failed (AIC=inf) for params {current_params}. Trying SMA(1) fallback model...")
            else:
                fallback_params = {"p": 0, "d": d, "q": 0, "P": 0, "D": D, "Q": 0}
                LOGGER.info(f"Initial model evaluation failed (AIC=inf) for params {current_params}. Trying random walk model (0,{d},0)(0,{D},0)...")
            # Validate fallback params before trying to fit
            if self.__validate_params(
                series,
                fallback_params["p"],
                fallback_params["q"],
                fallback_params["P"],
                fallback_params["Q"],
            ):
                current_params = fallback_params
                current_cost = self.__evaluate_arima_model(
                    series,
                    current_params,
                    exec_context=exec_context,
                    SARIMAX=SARIMAX,
                    warnings=warnings,
                    np=np,
                    ConvergenceWarning=ConvergenceWarning,
                )[0]
                if current_cost == np.inf:
                    raise RuntimeError(
                        f"Even the fallback model {fallback_params} failed to fit. Cannot proceed with optimization. Try to apply a log-transformation to the series to allow the algorithm to start"
                    )
                else:
                    LOGGER.info(f"Using fallback initial parameters {current_params} with AIC: {current_cost}")
            else:
                # If even the fallback is invalid (e.g., series too short), raise error
                raise RuntimeError(
                    f"Initial parameters {current_params} failed to fit and fallback parameters {fallback_params} are invalid for the series length. Cannot proceed."
                )

        LOGGER.info(f"Initial model evaluated with AIC: {current_cost}. Starting optimization loop...")

        # Keep the best criterion seen so far, and its associated configuration.
        best_params = current_params.copy()
        best_cost = current_cost

        # create a set of parameters already proposed by the algorithm
        proposed_params_cache = set()

        # Early stopping variables
        early_stopping_patience = self.optimization_loop_params.early_stopping_patience
        steps_without_improvement = 0
        last_best_cost = best_cost

        # Main loop of the simulated annealing process: Loop over the betas (the list of tolerances for moves that increase the cost)
        for i, beta in enumerate(beta_list):
            # At each beta record the acceptance rate using a counter for the number of accepted moves
            accepted_moves = 0
            # For each beta, perform a number of MCMC steps
            for t in range(mcmc_steps):
                # Propose new parameters based on the current ones and the step size
                proposed_params = self.__propose_new_params(
                    series,
                    current_params,
                    exec_context,
                    np,
                )
                # If the proposed parameters are already in the cache, skip this iteration. Else, add them to the cache.
                if tuple(proposed_params.items()) in proposed_params_cache:
                    LOGGER.info(f"Proposed parameters {proposed_params} already evaluated. Moving to next parameters proposal.")
                    continue
                else:
                    proposed_params_cache.add(tuple(proposed_params.items()))
                # Only evaluate if parameters actually changed
                if proposed_params != current_params:
                    new_cost = self.__evaluate_arima_model(
                        series,
                        proposed_params,
                        exec_context=exec_context,
                        SARIMAX=SARIMAX,
                        warnings=warnings,
                        np=np,
                        ConvergenceWarning=ConvergenceWarning,
                    )[0]
                    delta_cost = new_cost - current_cost

                else:
                    continue
                LOGGER.info(f"MCMC step: {t + 1}/{mcmc_steps} for beta: {beta} Proposed params: {proposed_params}, Delta cost: {delta_cost}")

                # Metropolis rule
                if self.__accept_new_params(delta_cost, beta, np):
                    current_params = proposed_params.copy()
                    current_cost = new_cost
                    accepted_moves += 1

                    if current_cost <= best_cost:
                        best_cost = current_cost
                        best_params = current_params.copy()

            # Dynamic progress update based on the current beta
            exec_context.set_progress(progress[i])

            # Print in the console the current beta, the acceptance rate, and the best parameters found so far
            LOGGER.info(
                f"Iteration: {i + 1}, beta: {beta}, accept_freq: {accepted_moves / mcmc_steps}, best params: {best_params}, best cost: {best_cost}"
            )

            # Early stopping logic
            if early_stopping_patience > 0:
                if best_cost < last_best_cost:
                    # Improvement found, reset counter
                    steps_without_improvement = 0
                    last_best_cost = best_cost
                else:
                    # No improvement
                    steps_without_improvement += 1

                if steps_without_improvement >= early_stopping_patience:
                    LOGGER.info(
                        f"Early stopping triggered: No improvement for {early_stopping_patience} consecutive steps. "
                        f"Best parameters: {best_params}, best cost: {best_cost}"
                    )
                    break

        # Return the best instance
        LOGGER.debug(f"Optimization finished. Final best parameters: {best_params} with AIC: {best_cost:.2f}")
        return best_params

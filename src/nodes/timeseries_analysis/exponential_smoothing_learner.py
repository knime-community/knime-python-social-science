import logging
import knime.extension as knext
from util import utils as kutil
from .timeseries_cat import timeseries_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.parameter_group(
    label="Model Methods",
)
class MethodModels:
    """
    Model Methods Parameter Group
    """

    class TrendModels(knext.EnumParameterOptions):
        Z = ("Automatic", "Automatic selection based on data characteristics")
        N = ("None", "No trend component")
        A = ("Additive", "Additive trend component")
        M = ("Multiplicative", "Multiplicative trend component")
        Ad = ("Additive damped", "Additive damped trend component")
        Md = ("Multiplicative damped", "Multiplicative damped trend component")

    trend_model = knext.EnumParameter(
        label="Trend Model",
        description="Specifies the trend component to use in the ETS model.",
        default_value=TrendModels.Z.name,
        enum=TrendModels,
    )
    allow_multiplicative_trend = knext.BoolParameter(
        label="Allow Multiplicative Trend",
        description="Enable or disable the use of multiplicative trend components in the model. Multiplicative trends can capture exponential growth or decay but tend to produce poor forecasting performance.",
        default_value=False,
    )

    class SeasonalModels(knext.EnumParameterOptions):
        Z = ("Automatic", "Automatic selection based on data characteristics")
        N = ("None", "No seasonal component")
        A = ("Additive", "Additive seasonal component")
        M = ("Multiplicative", "Multiplicative seasonal component")

    seasonal_model = knext.EnumParameter(
        label="Seasonal Model",
        description="Specifies the seasonal component to use in the ETS model.",
        default_value=SeasonalModels.Z.name,
        enum=SeasonalModels,
    )
    seasonality = knext.IntParameter(
        label="Seasonality",
        description="Specify the length of the seasonal period for your time series data. This parameter determines how many observations constitute one complete seasonal cycle",
        default_value=2,
        min_value=2,
    ).rule(knext.OneOf(seasonal_model, [SeasonalModels.N.name]), knext.Effect.HIDE)

    class ErrorModels(knext.EnumParameterOptions):
        Z = ("Automatic", "Automatic selection based on data characteristics")
        A = ("Additive", "Additive error model")
        M = ("Multiplicative", "Multiplicative error model")

    error_model = knext.EnumParameter(
        label="Error Model",
        description="Specifies the error model to use in the ETS model.",
        default_value=ErrorModels.Z.name,
        enum=ErrorModels,
    )


@knext.parameter_group(label="Box-Cox settings")
class BoxCoxSettings:
    """
    Grouped parameters for Box-Cox transformation settings.
    """

    enabled = knext.BoolParameter(
        label="Use Box-Cox transformation",
        description="Apply Box-Cox transformation before interpolation. Requires all observed values to be positive. Enabling this option can help stabilize variance and make the data more normally distributed. If marked, only additive models are considered. Fitted values are back-transformed in the in-sample output; residuals and residual-based diagnostics remain on the log scale.",
        default_value=False,
    )
    estimate_lambda = knext.BoolParameter(
        label="Estimate lambda parameter for Box-Cox automatically",
        description="If enabled, lambda is estimated from the observed (non-missing) values.",
        default_value=True,
    ).rule(knext.OneOf(enabled, [False]), knext.Effect.HIDE)
    lambda_value = knext.DoubleParameter(
        label="Box-Cox lambda",
        description="Used only when Box-Cox is enabled and automatic lambda estimation is disabled.",
        default_value=0.0,
    ).rule(
        knext.Or(
            knext.OneOf(enabled, [False]),
            knext.OneOf(estimate_lambda, [True]),
        ),
        knext.Effect.HIDE,
    )


@knext.node(
    name="ETS Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/TimeSeriesLearner.png",
    category=timeseries_analysis_category,
    id="ets_learner",
    keywords=[
        "Exponential Smoothing",
        "ETS",
        "State Space",
        "Time Series",
        "Forecasting",
        "Seasonal",
    ],
)
@knext.input_table(
    name="Input Data",
    description="Time series data table for training the ETS model. The selected target column must be numeric and contain no missing values.",
)
@knext.output_table(
    name="In-sample Predictions and Residuals",
    description="Model fit quality assessment table containing original values, fitted predictions, residuals, and absolute errors. Initial unstable predictions (first max(2×seasonal period, 10) observations; 10 for non-seasonal models) are excluded to ensure reliable performance metrics.",
)
@knext.output_table(
    name="Coefficients and Statistics",
    description="Complete model summary including short model name, fitted coefficients with standard errors, and goodness-of-fit statistics (AIC, BIC, Log Likelihood, MSE, MAE).",
)
@knext.output_table(
    name="Residual Diagnostics",
    description="Statistical tests for model adequacy assessment: Ljung-Box test for autocorrelation, Jarque-Bera test for normality (large samples), and Shapiro-Wilk test for normality (small-medium samples).",
)
@knext.output_table(
    name="Optimization History",
    description="Detailed optimization transparency table showing all parameter combinations tested including error, trend, seasonal components, and corresponding fit statistics.",
)
@knext.output_binary(
    name="Model",
    description="Trained ETS results object (pickled statsmodels ETSResults) ready for forecasting. Use with the  ETS Predictor node to generate forecasts without retraining.",
    id="est.model",
)
class ExponentialSmoothingLearner:
    """
    Fits an ETS (Error-Trend-Seasonal) exponential smoothing model and selects the best specification.

    **ETS** models are state-space exponential smoothing models that combine:
    - an **error** component (Additive or Multiplicative),
    - an optional **trend** component (none/Additive/Multiplicative, optionally damped),
    - an optional **seasonal** component (none/Additive/Multiplicative).

    This learner evaluates a set of candidate ETS specifications (based on your settings),
    records their fit statistics, and selects the best model according to the chosen criterion.

    ## How model selection works
    1. Build the list of candidate component combinations from your settings
    (Automatic → multiple options, fixed → a single option).
    2. Fit each candidate using `statsmodels` ETS state-space maximum likelihood.
    3. Store per-candidate diagnostics (AIC/BIC/HQIC/LLF and error metrics) in **Optimization History**.
    4. Select the best candidate by:
    - **maximizing** LLF (Log Likelihood), or
    - **minimizing** AIC/BIC/HQIC/MSE/MAE/(relative metrics).

    ## Inputs
    - **Input Data**: Univariate time series stored as a numeric column.

    ## Key constraints and safety checks
    - The **Target Column** must contain **no missing values**.
    - If a **seasonal** model is used:
    - a warning is issued when the series has fewer than **2 full seasonal cycles**,
    - fitting is blocked when the series has fewer than **1 seasonal period**.
    - **Multiplicative error** requires **strictly positive** observed values.
    - **Multiplicative trend** is only allowed when “Allow Multiplicative Trend” is enabled
    (and is generally discouraged due to unstable long-horizon forecasts).

    ## Parameters
    - **Target Column**: Numeric time series variable used for training.
    - **Model Selection Criterion**:
    - LLF, AIC, BIC, HQIC (likelihood / penalized likelihood criteria),
    - MSE, MAE (absolute-scale errors),
    - MSE_REL, MAE_REL (relative-scale errors; expressed as a percentage).
    - **Error / Trend / Seasonal Model**:
    - Choose “Automatic” to evaluate multiple options,
    - or fix specific components to restrict the search.
    - **Seasonality**: Seasonal period length (only shown when seasonal component ≠ None).

    ## Outputs
    1. **In-sample Predictions and Residuals**:
    Original value, fitted value, residual, and absolute error.
    (Initial unstable predictions are excluded for more reliable diagnostics.)
    2. **Coefficients and Statistics**:
    Model short name plus fitted parameters (excluding initial state values) and fit metrics.
    3. **Residual Diagnostics**:
    Ljung-Box, Jarque-Bera, Shapiro-Wilk tests applied to model residuals.
    4. **Optimization History**:
    One row per attempted component combination with fit statistics and status.
    5. **Model**:
    Pickled fitted ETS results object, usable by the ETS Predictor.

    ## Notes
    - Very large seasonal periods (e.g., > 100) can be computationally expensive and numerically fragile.
    Consider aggregation or seasonal pre-processing (e.g., STL/Fourier features) in those cases.
    """

    # Constants for diagnostic calculations
    DEFAULT_SKIP_OBSERVATIONS = 10
    DEFAULT_LJUNG_BOX_LAGS = 10

    input_column = knext.ColumnParameter(
        label="Target Column",
        description="Numeric time series column for model training. Must contain no missing values. This is the variable the model will learn to predict and forecast.",
        port_index=0,
        column_filter=kutil.is_numeric,
    )

    class CriterionOptions(knext.EnumParameterOptions):
        LLF = ("LLF", "Likelihood - Focuses solely on model fit quality without penalizing complexity.")
        AIC = ("AIC", "Akaike Information Criterion - Optimizes prediction accuracy, allows moderate complexity. Best for forecasting applications.")
        BIC = (
            "BIC",
            "Bayesian Information Criterion - Strong complexity penalty, favors simpler models. Best for model interpretation and explanation.",
        )
        HQIC = ("HQIC", "Hannan-Quinn Information Criterion - Balanced approach between prediction and parsimony. Good general-purpose choice.")
        MSE = ("MSE", "Mean Squared Error - Measures average squared difference between observed and predicted values.")
        MAE = ("MAE", "Mean Absolute Error - Measures average absolute difference between observed and predicted values.")
        MSE_REL = ("MSE_REL", "Relative Mean Squared Error - MSE of relative residuals (residual / fitted value), expressed as a percentage.")
        MAE_REL = ("MAE_REL", "Relative Mean Absolute Error - MAE of relative residuals (residual / fitted value), expressed as a percentage.")

    criterion = knext.EnumParameter(
        label="Model Selection Criterion",
        description="Criterion used to select the best model during training. 'AIC' and 'BIC' penalize model complexity, while 'Log Likelihood' (default) focuses on fit quality.",
        default_value=CriterionOptions.LLF.name,
        enum=CriterionOptions,
    )

    model_methods = MethodModels()

    boxcox = BoxCoxSettings()

    error_model = ""
    seasonal_model = ""
    trend_model = ""

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema) -> knext.Schema:
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.input_column,
            input_schema,
            kutil.is_numeric,
        )

        self.error_model = MethodModels.ErrorModels[self.model_methods.error_model].value[0]
        self.seasonal_model = MethodModels.SeasonalModels[self.model_methods.seasonal_model].value[0]
        self.trend_model = MethodModels.TrendModels[self.model_methods.trend_model].value[0]

        if self.error_model == "Additive" and self.seasonal_model == "Multiplicative":
            LOGGER.warning(
                "Additive error with Multiplicative seasonality is not recommended as it can lead to numerical instability due to division by values potentially close to zero in the state equations"
            )

        if "Multiplicative" in self.trend_model:
            if not self.model_methods.allow_multiplicative_trend:
                raise knext.InvalidParametersError('Cannot enforce Multiplicative Trend model unless "Allow Multiplicative Trend" option is marked.')
            LOGGER.warning(
                'Multiplicative trend methods have generally poor forecasting performance and should be used with caution. Mark the option "Allow Multiplicative Trend" to allow selection of multiplicative trend components.'
            )

        if "Multiplicative" in [self.error_model, self.trend_model, self.seasonal_model]:
            LOGGER.warning(
                "At least one model component is forced to be multiplicative. Ensure that the target column contains only positive values to avoid fitting errors."
            )
            if self.boxcox.enabled:
                raise knext.InvalidParametersError(
                    "Box-Cox transformation cannot be used with multiplicative model components. Please disable Box-Cox or switch to additive components (setting them explicitly or with 'Automatic' option)."
                )

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

        optimization_history_schema = knext.Schema(
            [
                knext.string(),
                knext.string(),
                knext.string(),
                knext.double(),
                knext.double(),
                knext.double(),
                knext.double(),
                knext.double(),
                knext.double(),
                knext.double(),
                knext.double(),
                knext.string(),
            ],
            ["error", "trend", "seasonal", "llf", "aic", "bic", "hqic", "mse", "mae", "mse_rel", "mae_rel", "status"],
        )

        binary_model_schema = knext.BinaryPortObjectSpec("est.model")

        return (
            predictions_schema,
            model_summary_schema,
            diagnostics_schema,
            optimization_history_schema,
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input: knext.Table):
        import pandas as pd
        import pickle

        self.error_model = MethodModels.ErrorModels[self.model_methods.error_model].value[0]
        self.seasonal_model = MethodModels.SeasonalModels[self.model_methods.seasonal_model].value[0]
        self.trend_model = MethodModels.TrendModels[self.model_methods.trend_model].value[0]

        # Initialize table with evaluated models at start of execution
        self.optimization_history = []

        df = input.to_pandas()
        target_col = df[self.input_column].astype(float)

        exec_context.set_progress(0.1)

        if kutil.count_negative_values(target_col) > 0:
            if "Multiplicative" in [self.error_model, self.trend_model, self.seasonal_model]:
                raise knext.InvalidParametersError("Cannot enforce multiplicative models with negative values in the target column.")
            if self.boxcox.enabled:
                raise knext.InvalidParametersError("Box-Cox transformation cannot be used with negative values in the target column.")

        # Apply Box-Cox transformation if enabled
        if self.boxcox.enabled:
            target_col, self.boxcox.lambda_value = kutil.box_cox_transform(
                target_col, LOGGER, self.boxcox.lambda_value if (not self.boxcox.estimate_lambda) else None
            )
            LOGGER.info(f"Applied Box-Cox transformation with lambda = {self.boxcox.lambda_value}.")

        # check if the number of obsevations is greater than or equal to twice the seasonal period
        if (self.seasonal_model != "None") and (len(target_col) < 2 * self.model_methods.seasonality):
            LOGGER.warning(
                f"The number of observations in the target column ({len(target_col)}) is lower than twice the seasonal period ({self.model_methods.seasonality}). This may lead to unreliable model estimates and forecasts."
            )
            exec_context.set_warning(
                f"The number of observations in the target column ({len(target_col)}) is lower than twice the seasonal period ({self.model_methods.seasonality}). This may lead to unreliable model estimates and forecasts."
            )
            if len(target_col) < self.model_methods.seasonality:
                raise knext.InvalidParametersError(
                    f"The number of observations in the target column ({len(target_col)}) is lower than twice the seasonal period ({self.model_methods.seasonality})."
                )

        # Check for missing values
        kutil.validate_missing_values(target_col)

        # Add performance warning for large seasonal periods
        if self.seasonal_model != "None":
            kutil.seasonality_performance_warning(
                context=exec_context,
                LOGGER=LOGGER,
                seasonality=self.model_methods.seasonality,
                seasonality_warning_threshold=100,
            )

        exec_context.set_progress(0.2)

        combinations = self.__get_model_combinations()

        trained_model_dict = self.__find_best_ets_model(
            target_col,
            combinations,
            exec_context,
            pd,
        )
        trained_model = trained_model_dict["model"]

        exec_context.set_progress(0.8)

        # adjust seasonality to 0 for non-seasonal model for compatibility with utils functions
        if self.seasonal_model == "None":
            adj_seasonality = 0
        else:
            adj_seasonality = self.model_methods.seasonality

        # Create enhanced predictions table with original values
        enhanced_predictions = kutil.enhance_predictions_table(
            trained_model, input, self.input_column, adj_seasonality, self.DEFAULT_SKIP_OBSERVATIONS, pd
        )

        # Apply Box-Cox transformation reverse if needed
        if self.boxcox.enabled:
            if "Fitted Value" in enhanced_predictions.columns:
                enhanced_predictions["Fitted Value"] = kutil.inv_box_cox_transform(enhanced_predictions["Fitted Value"], self.boxcox.lambda_value)

        # populate model coefficients and statistics with enhanced formatting
        coeffs_and_stats = self.__get_coeffs_and_stats(trained_model, pd)

        # generate residual diagnostics
        residual_diagnostics = kutil.compute_residual_diagnostics(
            trained_model, adj_seasonality, self.DEFAULT_SKIP_OBSERVATIONS, self.DEFAULT_LJUNG_BOX_LAGS, pd
        )

        optimization_history = self.optimization_history.drop("model", axis=1)

        model_binary = pickle.dumps(trained_model)

        exec_context.set_progress(0.99)

        return (
            knext.Table.from_pandas(enhanced_predictions, row_ids="keep"),
            knext.Table.from_pandas(coeffs_and_stats, row_ids="keep"),
            knext.Table.from_pandas(residual_diagnostics, row_ids="keep"),
            knext.Table.from_pandas(optimization_history, row_ids="keep"),
            model_binary,
        )

    def __find_best_ets_model(self, target_col, combinations, exec_context: knext.ExecutionContext, pd):
        """
        Searches through all valid ETS model combinations to find the best-fitting model based on AIC.
        """
        import warnings
        from numpy import mean, abs, inf
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        for idx, (error, trend, seasonal) in enumerate(combinations, 1):
            try:
                # Use catch_warnings to capture any warnings during fit()
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")  # Ensure all warnings are caught

                    trained_model = self.__train_model(
                        target_col,
                        error,
                        trend,
                        seasonal,
                    )

                status_message = "Success"

                # Check for degenerate log-likelihood (exclude models with log-likelihood ≈ 0)
                if abs(trained_model.llf) < 1e-6:
                    status_message = "Model Rejected: Degenerate Log-Likelihood"
                    LOGGER.info(f"Rejecting model with degenerate log-likelihood: {trained_model.llf} for model ETS({error}, {trend}, {seasonal}).")

                # Check if any ConvergenceWarning was caught
                for warning in caught_warnings:
                    if issubclass(warning.category, ConvergenceWarning):
                        status_message = f"Convergence Failed: {str(warning.message)}"
                        LOGGER.info(f"ConvergenceWarning occurred for model ETS({error}, {trend}, {seasonal}).")

                if status_message == "Success":
                    pred = trained_model.fittedvalues
                    err_abs = target_col - pred
                    err_rel = err_abs / pred
                    mse_abs = mean(err_abs**2)
                    mae_abs = mean(abs(err_abs))
                    mse_rel = mean(err_rel**2) * 100
                    mae_rel = mean(abs(err_rel)) * 100
                    LOGGER.info(
                        f"Fitted ETS({error}, {trend}, {seasonal}) model {idx}/{len(combinations)} successfully. resid range: {trained_model.resid.min()} to {trained_model.resid.max()}."
                    )

                self.optimization_history.append(
                    {
                        "error": error,
                        "trend": trend,
                        "seasonal": seasonal,
                        "llf": trained_model.llf if status_message == "Success" else -inf,
                        "aic": trained_model.aic if status_message == "Success" else inf,
                        "bic": trained_model.bic if status_message == "Success" else inf,
                        "hqic": trained_model.hqic if status_message == "Success" else inf,
                        "mse": mse_abs if status_message == "Success" else inf,
                        "mae": mae_abs if status_message == "Success" else inf,
                        "mse_rel": mse_rel if status_message == "Success" else inf,
                        "mae_rel": mae_rel if status_message == "Success" else inf,
                        "status": status_message,
                        "model": trained_model,
                    }
                )

            except Exception as e:
                # Handle potential errors during model fitting (e.g., invalid parameters)
                LOGGER.warning(f"Model fitting failed for combination: ETS({error}, {trend}, {seasonal})", exc_info=True)
                exec_context.set_warning(
                    f"WARNING: ETS model fitting failed with parameters ETS({error}, {trend}, {seasonal}). "
                    f"Error: {str(e)}. This may indicate numerical issues with large seasonal periods "
                    f"or insufficient data for the model complexity."
                )
                self.optimization_history.append(
                    {
                        "error": error,
                        "trend": trend,
                        "seasonal": seasonal,
                        "llf": inf,
                        "aic": inf,
                        "bic": inf,
                        "hqic": inf,
                        "mse": inf,
                        "mae": inf,
                        "mse_rel": inf,
                        "mae_rel": inf,
                        "status": str(f"Fitting Failed: {str(e)[:100]}"),
                        "model": trained_model,
                    }
                )
                continue

        if not self.optimization_history:
            raise knext.ExecutionError("All model fitting attempts failed. Please check the input data and model configuration.")

        criterion = str(self.criterion).lower()
        self.optimization_history = pd.DataFrame(self.optimization_history)

        if criterion == "llf":
            best_idx = self.optimization_history[criterion].idxmax()
        else:
            best_idx = self.optimization_history[criterion].idxmin()

        return self.optimization_history.loc[best_idx].to_dict()

    def __get_model_combinations(self):
        """
        Generates all valid combinations of ETS model components based on user selections.
        """
        from itertools import product

        # initialize lists for each component's options for safety
        error_options = []
        trend_options = []
        seasonal_options = []

        if self.error_model == "Automatic":
            error_options = ["Additive", "Multiplicative"]
            if self.boxcox.enabled:
                error_options = ["Additive"]
        else:
            error_options = [self.error_model]

        if self.trend_model == "Automatic":
            if (not self.model_methods.allow_multiplicative_trend) or (self.boxcox.enabled):
                trend_options = ["None", "Additive", "Additive damped"]
            else:
                trend_options = ["None", "Additive", "Multiplicative", "Additive damped", "Multiplicative damped"]
        else:
            trend_options = [self.trend_model]

        if self.seasonal_model == "Automatic":
            seasonal_options = ["None", "Additive", "Multiplicative"]
            if self.boxcox.enabled:
                seasonal_options = ["None", "Additive"]
        else:
            seasonal_options = [self.seasonal_model]

        return list(product(error_options, trend_options, seasonal_options))

    def __train_model(
        self,
        target_col,
        error,
        trend,
        seasonal,
    ):
        """
        Workhorse function fitting the ETS model to the target column using specified model components.
        """
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        damped_trend = False
        if "damped" in trend:
            damped_trend = True
            trend = trend.replace(" damped", "")

        model = ETSModel(
            endog=target_col,
            error=error,
            trend=None if trend == "None" else trend,
            damped_trend=damped_trend,
            seasonal=None if seasonal == "None" else seasonal,
            seasonal_periods=self.model_methods.seasonality if seasonal != "None" else None,
        )

        return model.fit(disp=False)

    def __get_coeffs_and_stats(self, model, pd):
        """
        Compiles comprehensive model summary with parameters, coefficients, and fit statistics.

        Creates a detailed table containing the optimal ETS parameters, all model coefficients
        with their standard errors. Key goodness-of-fit metrics are obtained from utility get_model_stats.
        Each entry includes explanatory text to help interpret the results.

        Parameters:
        - model: statsmodels.tsa.exponential_smoothing.ets.ETSResults
            Fitted ETS model containing coefficients and statistics.

        Returns:
        - pd.DataFrame
            Model summary table with parameters, coefficients, and interpretive explanations.
        """
        # Create the data structure
        data = []

        # Model type
        data.append(
            {
                "Parameter": "Model Type",
                "Value": float("nan"),
                "Explanation": f"Type of the ETS model used for forecasting: {model.short_name}.",
            }
        )
        # Box-Cox lambda
        if self.boxcox.enabled:
            data.append(
                {
                    "Parameter": "Box-Cox Lambda",
                    "Value": float(self.boxcox.lambda_value),
                    "Explanation": "Estimated or given lambda parameter used for Box-Cox transformation of the target variable.",
                }
            )

        # Model coefficients and standard errors
        if hasattr(model, "params") and len(model.params) > 0:
            for i in range(len(model.params)):
                param_name = model.param_names[i]
                if "initial" in param_name.lower():
                    continue  # Skip initial state parameters
                coeff_val = model.params[i]
                data.append(
                    {
                        "Parameter": f"{param_name} (Coefficient)",
                        "Value": float(coeff_val),
                        "Explanation": "Model coefficient representing the relationship strength.",
                    }
                )
                # Add standard error if available
                if hasattr(model, "bse"):
                    sterr = model.bse[i]
                    data.append(
                        {
                            "Parameter": f"{param_name} (Std. Error)",
                            "Value": sterr,  # np.nan if np.isnan(float(model.bse[i])) else
                            "Explanation": "Standard error of the coefficient estimate.",
                        }
                    )

        return kutil.get_model_stats(model, data)

import logging
import knime.extension as knext
from util import utils as kutil
from .multivariate_cat import multivariate_analysis_category

LOGGER = logging.getLogger(__name__)


class RegressionType(knext.EnumParameterOptions):
    """Regression method options for the Linear Model Learner node."""

    OLS = (
        "OLS",
        "Ordinary Least Squares - standard unregularized linear regression",
    )
    RIDGE = (
        "Ridge",
        "Ridge (L2) regularization - shrinks coefficients towards zero. Optimal alpha selected via cross-validation.",
    )
    LASSO = (
        "Lasso",
        "Lasso (L1) regularization - performs variable selection by setting some coefficients to zero. Optimal alpha selected via cross-validation.",
    )
    PANEL = (
        "Panel Data",
        "Panel regression with entity and time fixed or random effects using linearmodels. Requires both entity and time index columns.",
    )


class PanelEffectType(knext.EnumParameterOptions):
    """Fixed or Random effects option for panel regression."""

    FIXED = ("Fixed", "Fixed Effects")
    RANDOM = ("Random", "Random Effects")


@knext.parameter_group(label="Model Settings")
class ModelSettings:
    """
    Regression model type, specification, and diagnostic computation settings.
    """

    regression_type = knext.EnumParameter(
        label="Regression Type",
        description="Choose the type of regression estimation. OLS provides full diagnostics; Ridge and Lasso add regularization for improved prediction or feature selection. Panel options for entity or time effects in longitudinal data.",
        default_value=RegressionType.OLS.name,
        enum=RegressionType,
    )

    include_intercept = knext.BoolParameter(
        label="Include Intercept",
        description="Include a constant term (intercept) in the regression model. Disable only if you have theoretical reasons to force the regression line through the origin. Not applicable for panel regression (entity/time effects serve as intercepts).",
        default_value=True,
    ).rule(knext.OneOf(regression_type, [RegressionType.PANEL.name]), knext.Effect.HIDE)

    log_transform_y = knext.BoolParameter(
        label="Logarithm of Y",
        description="Apply natural logarithm transformation to the target variable. Useful for modeling multiplicative relationships or when the target has exponential growth. Results will need to be back-transformed for interpretation.",
        default_value=False,
    )

    # Panel data settings (shown only for Panel regression)
    panel_effect = knext.EnumParameter(
        label="Effect Type",
        description="Choose between Fixed Effects (FE) or Random Effects (RE) for panel regression. Fixed Effects control for all time-invariant unobserved heterogeneity within each entity by including entity-specific intercepts (equivalent to entity dummy variables). Use FE when unobserved factors (e.g., firm culture, individual ability) are correlated with predictors. Random Effects assume unobserved factors are uncorrelated with predictors and models them as random variation, providing more efficient estimates when this assumption holds. FE is more robust but uses more degrees of freedom; RE is more efficient but requires stronger assumptions. Hausman test can help choose between them.",
        default_value=PanelEffectType.FIXED.name,
        enum=PanelEffectType,
    ).rule(knext.OneOf(regression_type, [RegressionType.PANEL.name]), knext.Effect.SHOW)

    panel_entity_index = knext.ColumnParameter(
        label="Entity Index Column",
        description="Column identifying the cross-sectional units (entities) in your panel data. Each unique value represents a distinct entity (e.g., individual person, company, country, store) tracked over time. Examples: customer_id, firm_id, country_code, patient_id. The model will account for systematic differences between entities by estimating entity-specific effects (FE) or treating them as random variation (RE). This controls for time-invariant characteristics of each entity that might confound the relationship between predictors and outcome.",
        port_index=0,
    ).rule(knext.OneOf(regression_type, [RegressionType.PANEL.name]), knext.Effect.SHOW)

    panel_time_index = knext.ColumnParameter(
        label="Time Index Column",
        description="Column identifying the time periods in your panel data. Each value represents a distinct time point when observations were recorded (e.g., year, quarter, month, date). Examples: year, observation_date, time_period, wave. The model will account for time-specific shocks or trends affecting all entities simultaneously by estimating time fixed effects. This controls for temporal factors (e.g., economic cycles, policy changes, seasonality) that might affect all entities and confound the analysis. Both entity and time indices are required to properly structure the panel data.",
        port_index=0,
    ).rule(knext.OneOf(regression_type, [RegressionType.PANEL.name]), knext.Effect.SHOW)

    compute_vif = knext.BoolParameter(
        label="Compute VIF (Variance Inflation Factor)",
        description="Calculate Variance Inflation Factors to detect multicollinearity among predictors. VIF > 10 indicates problematic multicollinearity; VIF > 5 suggests caution. Note: Requires at least 2 predictors. Only available for OLS.",
        default_value=True,
    ).rule(knext.OneOf(regression_type, [RegressionType.RIDGE.name, RegressionType.LASSO.name, RegressionType.PANEL.name]), knext.Effect.HIDE)

    compute_influence = knext.BoolParameter(
        label="Compute Influence Diagnostics",
        description="Calculate Cook's Distance and leverage values to identify influential observations that may disproportionately affect model estimates. Only available for OLS.",
        default_value=True,
    ).rule(knext.OneOf(regression_type, [RegressionType.RIDGE.name, RegressionType.LASSO.name, RegressionType.PANEL.name]), knext.Effect.HIDE)


@knext.node(
    name="Linear Model Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/LinearModelLearner.png",
    category=multivariate_analysis_category,
    keywords=[
        "Linear Regression",
        "OLS",
        "Ordinary Least Squares",
        "Multiple Regression",
        "Statistical Modeling",
        "Diagnostics",
        "VIF",
        "Cook's Distance",
        "Ridge",
        "Lasso",
        "Panel Data",
        "Fixed Effects",
        "Random Effects",
        "Longitudinal",
    ],
    id="linear_model_learner",
)
@knext.input_table(
    name="Input Data",
    description="Dataset for linear regression analysis. Must contain one numeric target variable and one or more predictor variables (numeric or categorical). Categorical variables are automatically converted to dummy variables. For panel data, must include both entity and time index columns. Missing values are handled via listwise deletion.",
)
@knext.output_table(
    name="Model Summary",
    description="Comprehensive regression results including R², Adjusted R², F-statistic, RMSE, MAE, and residual diagnostic tests (Jarque-Bera, Breusch-Pagan, Durbin-Watson, Omnibus).",
)
@knext.output_table(
    name="Coefficients",
    description="Detailed coefficient table with estimates, standard errors, t-statistics, p-values, confidence intervals, standardized beta coefficients, and VIF for multicollinearity assessment.",
)
@knext.output_table(
    name="Predictions and Residuals",
    description="Fitted values, residuals, standardized residuals, studentized residuals, Cook's Distance, leverage, and influence flags for each observation.",
)
@knext.output_image(
    name="Diagnostic Plots",
    description="Comprehensive 6-panel diagnostic visualization: (1) Residuals vs Fitted, (2) Q-Q Plot, (3) Scale-Location, (4) Cook's Distance, (5) Residual Histogram, (6) Leverage vs Residuals.",
)
@knext.output_binary(
    name="Model",
    description="Trained regression model object with all specifications and settings. Can be used with a Linear Model Predictor node to generate predictions on new data without retraining.",
    id="linear_regression.model",
)
class LinearModelLearner:
    """
    Fits linear regression models with four estimation methods: **OLS** (Ordinary Least Squares), **Ridge** (L2 regularization),
    **Lasso** (L1 regularization), and **Panel Data**. Provides comprehensive diagnostics including VIF for multicollinearity,
    Cook's Distance for influential observations, and residual tests for assumption validation. Automatically handles categorical
    variables through one-hot encoding.

    ## Regression Methods

    - **OLS**: Standard linear regression with full diagnostics (statistical tests, VIF, influence measures)
    - **Ridge**: L2 regularization that shrinks coefficients toward zero to handle multicollinearity and improve prediction stability. Optimal alpha (regularization strength) is automatically selected via 5-fold cross-validation from 20 logarithmically-spaced candidates ranging from 0.001 to 100.
    - **Lasso**: L1 regularization that performs automatic variable selection by shrinking some coefficients exactly to zero, effectively removing less important predictors. Optimal alpha is automatically selected via 5-fold cross-validation from 20 logarithmically-spaced candidates ranging from 0.001 to 100.
    - **Panel Data**: Panel regression with both entity and time effects. Controls for unobserved heterogeneity across entities (e.g., individuals, firms, countries) and time-specific factors. Requires both entity and time index columns.

    ## Panel Data Options

    - **Fixed Effects (FE)**: Controls for unobserved, time-invariant factors specific to each unit (e.g., firm culture).
    - **Random Effects (RE)**: Assumes unobserved factors are uncorrelated with independent variables.

    ## Features

    Model options include optional intercept (OLS/Ridge/Lasso only), log transformation of target variable, automatic dummy variable
    creation for categorical predictors (first category as reference), and automatic handling of missing values via listwise deletion.

    OLS provides full diagnostics including inference statistics (standard errors, t-statistics, p-values, 95% confidence intervals),
    VIF for detecting multicollinearity (VIF > 10 indicates high collinearity), Cook's Distance and leverage for identifying
    influential observations and outliers, and residual tests including Jarque-Bera and Omnibus (normality), Breusch-Pagan
    (heteroscedasticity), and Durbin-Watson (autocorrelation).

    Panel regression provides inference statistics (standard errors, t-statistics, p-values, confidence intervals) but VIF and
    influence diagnostics are not available as they are not meaningful in panel data context.

    Ridge and Lasso do not provide inference statistics because regularization biases coefficient estimates, invalidating
    standard error calculations and p-values derived from OLS theory. VIF is not computed for regularized models. Cook's Distance and leverage are not available
    because these diagnostics rely on the hat matrix from unbiased OLS fits.

    The regression methods provide fit statistics (R², Adjusted R², RMSE, MAE), standardized coefficients (beta weights for comparing
    predictor importance), and predictions with residuals (fitted values, raw and standardized residuals).

    ## Categorical Variables

    String/categorical predictors are automatically converted to dummy variables using one-hot encoding. The first category
    of each categorical variable serves as the reference category (dropped to avoid the dummy variable trap). Dummy variable
    names follow the format: `VariableName_CategoryValue`.

    ## Outputs

    1. **Model Summary**: Fit statistics, diagnostic test results (OLS only), and optimal regularization parameter (Ridge/Lasso)
    2. **Coefficients Table**: Estimates, inference statistics (standard errors, t/p-values, CIs for OLS/Panel), standardized betas, VIF (OLS only)
    3. **Predictions Table**: Row-level predictions, residuals, Cook's Distance (OLS only), leverage (OLS only), influence flags
    4. **Diagnostic Plots**: 6-panel visualization including residuals vs fitted, Q-Q plot, scale-location, Cook's Distance (OLS only),
       residual histogram, and leverage plot (OLS only)

    ## Key Assumptions

    **OLS/Ridge/Lasso:**
    - Linearity between predictors and target
    - Homoscedasticity (constant residual variance)
    - Normality of residuals
    - No perfect multicollinearity

    """

    target_column = knext.ColumnParameter(
        label="Target Variable (Y)",
        description="Numeric dependent variable to predict. Must be continuous and numeric. This is the outcome variable that the model will learn to predict from the predictor variables.",
        port_index=0,
        column_filter=kutil.is_numeric,
    )

    predictor_columns = knext.MultiColumnParameter(
        label="Predictor Variables (X)",
        description="One or more independent variables used to predict the target. Numeric variables are used as-is. Categorical (string) variables are automatically converted to dummy variables (one-hot encoding) with the first category as reference.",
        column_filter=kutil.boolean_or(kutil.is_numeric, kutil.is_string),
    )

    model_settings = ModelSettings()

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        """
        Validates configuration and defines output schemas.
        """
        # Validate column selection
        self.target_column = kutil.column_exists_or_preset(
            configure_context,
            self.target_column,
            input_schema,
            kutil.is_numeric,
        )

        # Model summary schema (includes diagnostics)
        model_summary_schema = knext.Schema(
            [knext.string(), knext.double(), knext.string()],
            ["Statistic", "Value", "Interpretation"],
        )

        # Coefficients schema (includes VIF and Standardized Beta)
        coef_columns = [
            "Variable",
            "Estimate",
            "Standardized Beta",
            "Std Error",
            "t-statistic",
            "p-value",
            "CI Lower",
            "CI Upper",
            "VIF",
            "VIF Assessment",
        ]
        coef_types = [knext.string()] + [knext.double()] * 8 + [knext.string()]

        coefficients_schema = knext.Schema(coef_types, coef_columns)

        # Predictions schema (includes Cook's Distance, Leverage, Flag)
        predictions_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double(), knext.string(), knext.double(), knext.double()],
            ["Actual", "Fitted", "Residual", "Flag", "Cook's Distance", "Leverage"],
        )

        binary_model_schema = knext.BinaryPortObjectSpec("linear_regression.model")

        return (
            model_summary_schema,
            coefficients_schema,
            predictions_schema,
            knext.ImagePortObjectSpec(knext.ImageFormat.SVG),
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        """
        Fits OLS regression model and generates comprehensive diagnostics.
        """
        # Import heavy dependencies only when needed
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        import warnings
        import pickle

        # Suppress convergence warnings for cleaner output
        warnings.filterwarnings("ignore")

        exec_context.set_progress(0.05, "Loading and preparing data...")

        # Load and prepare data
        df = input_table.to_pandas()
        df_clean, y, n_obs = self._prepare_and_clean_data(df, pd, np)

        # Handle categorical variables and create design matrix
        X, predictor_names_for_model = self._encode_categorical_variables(df_clean, pd)
        n_predictors = X.shape[1]

        # Check for perfect collinearity
        if np.linalg.matrix_rank(X) < n_predictors:
            raise knext.InvalidParametersError(
                "Perfect multicollinearity detected: One or more predictors are linear combinations of others. "
                "Please remove redundant predictors or check for constant columns."
            )

        exec_context.set_progress(0.15, "Fitting regression model...")

        # Get regression type
        reg_type = self.model_settings.regression_type

        # Fit model based on regression type
        if reg_type == RegressionType.OLS.name:
            import statsmodels.api as sm

            results, beta, se_beta, t_stats, p_values, y_pred, residuals, predictor_names, selected_alpha = self._fit_ols_model(
                X, y, predictor_names_for_model, sm, np
            )
            ci_lower, ci_upper = None, None  # Will compute below

        elif reg_type == RegressionType.PANEL.name:
            results, beta, se_beta, t_stats, p_values, ci_lower, ci_upper, y_pred, residuals, predictor_names, X, y, selected_alpha = (
                self._fit_panel_model(df, y, pd, np, exec_context)
            )
            n_obs = len(y)
            n_predictors = len(beta)  # Use actual number of coefficients, not X.shape[1]

        else:  # Ridge/Lasso
            model, beta, se_beta, t_stats, p_values, y_pred, residuals, predictor_names, X_design, selected_alpha = self._fit_regularized_model(
                X, y, predictor_names_for_model, np
            )
            ci_lower, ci_upper = None, None  # Will compute below

        exec_context.set_progress(0.30, "Computing model fit statistics...")

        # Compute model statistics
        stats_dict = self._compute_model_statistics(y, residuals, beta, n_obs, np, stats)

        # Compute confidence intervals (only if not already computed by panel model)
        if ci_lower is None or ci_upper is None:
            ci_lower, ci_upper = self._compute_confidence_intervals(beta, se_beta, stats_dict["df_residual"], stats, np)

        # Build model summary
        model_summary_data = self._build_model_summary(n_obs, stats_dict["n_predictors"], stats_dict, selected_alpha, np)

        exec_context.set_progress(0.45, "Computing standardized coefficients...")

        # Compute standardized coefficients
        standardized_beta = self._compute_standardized_coefficients(X, y, beta, n_obs, np)

        # Calculate VIF
        vif_values, vif_assessments = self._compute_vif(X, n_obs, n_predictors, np)

        # Build coefficients DataFrame with all statistics
        coef_data = {
            "Variable": predictor_names,
            "Estimate": beta,
            "Standardized Beta": standardized_beta,
            "Std Error": se_beta,
            "t-statistic": t_stats,
            "p-value": p_values,
            "CI Lower": ci_lower,
            "CI Upper": ci_upper,
            "VIF": vif_values,
            "VIF Assessment": vif_assessments,
        }
        coefficients_df = pd.DataFrame(coef_data)

        exec_context.set_progress(0.55, "Running residual diagnostics...")

        # Standardized residuals
        mse = stats_dict["mse"]
        standardized_residuals = residuals / np.sqrt(mse) if not np.isnan(mse) else residuals / np.std(residuals)

        # Studentized residuals and leverage (only for OLS)
        if reg_type == RegressionType.OLS.name:
            import statsmodels.api as sm

            if self.model_settings.include_intercept:
                X_sm = sm.add_constant(X)
            else:
                X_sm = X
            leverage = (X_sm * np.linalg.solve(X_sm.T @ X_sm, X_sm.T).T).sum(axis=1)
        else:
            leverage = np.full(n_obs, np.nan)

        # Add diagnostic tests to model summary
        diagnostic_results = self._compute_residual_diagnostics(residuals, X, mse, stats_dict, stats, np)
        model_summary_data.extend(diagnostic_results)

        model_summary_df = pd.DataFrame(model_summary_data, columns=["Statistic", "Value", "Interpretation"])

        exec_context.set_progress(0.75, "Computing influence diagnostics...")

        # Compute influence diagnostics
        cooks_d, leverage, flags = self._compute_influence_diagnostics(residuals, standardized_residuals, leverage, n_obs, beta, mse, np)

        # Ensure all are 1D arrays/lists
        cooks_d = np.asarray(cooks_d).ravel()
        leverage = np.asarray(leverage).ravel()
        y_1d = np.asarray(y).ravel()
        y_pred_1d = np.asarray(y_pred).ravel()
        residuals_1d = np.asarray(residuals).ravel()

        # Predictions and residuals table
        predictions_df = pd.DataFrame(
            {
                "Actual": y_1d if not self.model_settings.log_transform_y else np.exp(y_1d),
                "Fitted": y_pred_1d if not self.model_settings.log_transform_y else np.exp(y_pred_1d),
                "Residual": residuals_1d,
                "Flag": flags,
                "Cook's Distance": cooks_d,
                "Leverage": leverage,
            }
        )

        exec_context.set_progress(0.85, "Creating diagnostic plots...")

        # Create diagnostic plots
        buf = self._create_diagnostic_plots(y_pred, residuals, standardized_residuals, cooks_d, leverage, n_obs, plt, stats, np)

        exec_context.set_progress(0.95, "Preparing model for export...")

        # Create optimized model dictionary for predictor node (only essential data)
        model_dict = {
            # Model identification and type
            "regression_type": reg_type,
            "model_label": RegressionType[reg_type].value[0],
            # Model settings for prediction
            "include_intercept": self.model_settings.include_intercept if reg_type != RegressionType.PANEL.name else None,
            "log_transform_y": self.model_settings.log_transform_y,
            # Panel-specific settings (if applicable)
            "panel_effect": self.model_settings.panel_effect if reg_type == RegressionType.PANEL.name else None,
            "panel_entity_index": self.model_settings.panel_entity_index if reg_type == RegressionType.PANEL.name else None,
            "panel_time_index": self.model_settings.panel_time_index if reg_type == RegressionType.PANEL.name else None,
            # Feature information for prediction (before and after dummy transformation)
            "predictor_columns": list(self.predictor_columns),  # Original feature names before dummy encoding
            "predictor_names_for_model": predictor_names_for_model,  # Feature names after dummy encoding (for prediction)
            # Fitted model objects for prediction (minimal necessary objects)
            "model_object": model
            if reg_type in [RegressionType.RIDGE.name, RegressionType.LASSO.name]
            else None,  # sklearn fitted models (Ridge/Lasso only)
            "results_object": results
            if reg_type in [RegressionType.OLS.name, RegressionType.PANEL.name]
            else None,  # statsmodels/linearmodels fitted results (OLS/Panel only)
        }

        model_binary = pickle.dumps(model_dict)

        exec_context.set_progress(1.0, "Returning the output!")

        return (
            knext.Table.from_pandas(model_summary_df),
            knext.Table.from_pandas(coefficients_df),
            knext.Table.from_pandas(predictions_df),
            buf.getvalue(),
            model_binary,
        )

    ## Helper methods for linear regression computations

    def _prepare_and_clean_data(self, df, pd, np):
        """
        Prepare data by removing missing values and extracting target variable.

        Returns:
            tuple: (df_clean, y, n_obs) - cleaned dataframe, target values, number of observations
        """
        analysis_columns = [self.target_column] + list(self.predictor_columns)
        df_clean = df[analysis_columns].dropna()

        if df_clean.shape[0] < len(self.predictor_columns) + 2:
            raise knext.InvalidParametersError(
                f"Insufficient data: Only {df_clean.shape[0]} complete observations available. "
                f"Linear regression requires at least {len(self.predictor_columns) + 2} observations "
                f"for {len(self.predictor_columns) + 2} predictors."
            )

        y = df_clean[self.target_column].values

        # Apply log transformation to Y if requested
        if self.model_settings.log_transform_y:
            if np.any(y <= 0):
                raise knext.InvalidParametersError(
                    "Cannot apply logarithm transformation: Target variable contains zero or negative values. "
                    "Log transformation requires all positive values."
                )
            y = np.log(y)

        n_obs = len(y)
        return df_clean, y, n_obs

    def _encode_categorical_variables(self, df_clean, pd):
        """
        Handle categorical variables by creating dummy variables.

        Returns:
            tuple: (X, predictor_names_for_model) - design matrix and predictor names after encoding
        """
        numeric_cols = []
        categorical_cols = []
        for col in self.predictor_columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        if categorical_cols:
            df_dummies = pd.get_dummies(df_clean[self.predictor_columns], columns=categorical_cols, drop_first=True, dtype=float)
            X = df_dummies.values
            predictor_names_for_model = list(df_dummies.columns)
        else:
            X = df_clean[self.predictor_columns].values
            predictor_names_for_model = list(self.predictor_columns)

        return X, predictor_names_for_model

    def _fit_ols_model(self, X, y, predictor_names_for_model, sm, np):
        """
        Fit OLS regression model using statsmodels.

        Returns:
            tuple: (results, beta, se_beta, t_stats, p_values, y_pred, residuals, predictor_names)
        """
        if self.model_settings.include_intercept:
            X_design = sm.add_constant(X)
            predictor_names = ["const"] + predictor_names_for_model
        else:
            X_design = X
            predictor_names = predictor_names_for_model

        model = sm.OLS(y, X_design)
        results = model.fit()

        return (
            results,
            results.params,
            results.bse,
            results.tvalues,
            results.pvalues,
            results.fittedvalues,
            results.resid,
            predictor_names,
            np.nan,  # selected_alpha not applicable for OLS
        )

    def _fit_panel_model(self, df, y, pd, np, exec_context):
        """
        Fit panel regression model using linearmodels.

        Returns:
            tuple: (results, beta, se_beta, t_stats, p_values, ci_lower, ci_upper,
                   y_pred, residuals, predictor_names, X_panel, y_panel)
        """
        from linearmodels.panel import PanelOLS, RandomEffects

        entity_col = self.model_settings.panel_entity_index
        time_col = self.model_settings.panel_time_index
        predictors = [col for col in self.predictor_columns if col not in [entity_col, time_col]]

        if not predictors:
            raise knext.InvalidParametersError("Panel regression requires at least one predictor not used as entity or time index.")

        df_panel = df.copy()
        df_panel = df_panel.dropna(subset=[self.target_column] + predictors + [entity_col, time_col])

        # Handle categorical variables
        numeric_pred = [col for col in predictors if pd.api.types.is_numeric_dtype(df_panel[col])]
        categorical_pred = [col for col in predictors if col not in numeric_pred]

        if categorical_pred:
            df_dummies = pd.get_dummies(df_panel[predictors], columns=categorical_pred, drop_first=True, dtype=float)
            df_panel = pd.concat([df_panel[[entity_col, time_col, self.target_column]], df_dummies], axis=1)
            predictors_final = list(df_dummies.columns)
        else:
            predictors_final = predictors

        # Set 2-level MultiIndex for panel data
        df_panel = df_panel.set_index([entity_col, time_col])
        y_panel = df_panel[self.target_column]
        X_panel = df_panel[predictors_final]

        # Fit panel model
        if self.model_settings.panel_effect == PanelEffectType.FIXED.name:
            model = PanelOLS(y_panel, X_panel, entity_effects=True, time_effects=True, drop_absorbed=True)
        else:
            model = RandomEffects(y_panel, X_panel)

        results = model.fit()

        # Extract statistics
        beta = results.params.values
        se_beta = results.std_errors.values if hasattr(results, "std_errors") else np.full_like(beta, np.nan)
        t_stats = results.tstats.values if hasattr(results, "tstats") else np.full_like(beta, np.nan)
        p_values = results.pvalues.values if hasattr(results, "pvalues") else np.full_like(beta, np.nan)

        ci = results.conf_int() if hasattr(results, "conf_int") else None
        if ci is not None:
            ci_lower = ci.iloc[:, 0].values
            ci_upper = ci.iloc[:, 1].values
        else:
            ci_lower = np.full_like(beta, np.nan)
            ci_upper = np.full_like(beta, np.nan)

        y_pred = results.fitted_values.values
        residuals = results.resids.values
        predictor_names = list(results.params.index)

        # Check if any variables were dropped due to collinearity
        if len(predictor_names) < len(predictors_final):
            dropped_vars = set(predictors_final) - set(predictor_names)
            effect_type = "fixed effects" if self.model_settings.panel_effect == PanelEffectType.FIXED.name else "random effects"
            exec_context.set_warning(
                f"One or more variables were dropped because they are collinear with the {effect_type}: {', '.join(sorted(dropped_vars))}"
            )
            # Filter X_panel to only include non-dropped variables for standardized coefficient computation
            X_panel_filtered = X_panel[predictor_names]
        else:
            X_panel_filtered = X_panel

        return (
            results,
            beta,
            se_beta,
            t_stats,
            p_values,
            ci_lower,
            ci_upper,
            y_pred,
            residuals,
            predictor_names,
            X_panel_filtered.values,
            y_panel.values,
            np.nan,
        )

    def _fit_regularized_model(self, X, y, predictor_names_for_model, np):
        """
        Fit Ridge or Lasso regression with cross-validated regularization.

        Returns:
            tuple: (model, beta, y_pred, residuals, predictor_names, X_design, selected_alpha)
        """
        from sklearn.linear_model import RidgeCV, LassoCV

        alphas = np.logspace(-3, 2, 20)
        reg_type = self.model_settings.regression_type

        if reg_type == RegressionType.RIDGE.name:
            model = RidgeCV(alphas=alphas, fit_intercept=self.model_settings.include_intercept, cv=5)
        else:  # LASSO
            model = LassoCV(alphas=alphas, fit_intercept=self.model_settings.include_intercept, cv=5, max_iter=10000)

        model.fit(X, y)
        selected_alpha = model.alpha_
        n_obs = len(y)

        if self.model_settings.include_intercept:
            beta = np.concatenate([[model.intercept_], model.coef_])
            predictor_names = ["Intercept"] + predictor_names_for_model
            X_design = np.column_stack([np.ones(n_obs), X])
        else:
            beta = model.coef_
            predictor_names = predictor_names_for_model
            X_design = X

        y_pred = model.predict(X)
        residuals = y - y_pred

        # Regularized models don't provide inference statistics
        se_beta = np.full_like(beta, np.nan)
        t_stats = np.full_like(beta, np.nan)
        p_values = np.full_like(beta, np.nan)

        return (model, beta, se_beta, t_stats, p_values, y_pred, residuals, predictor_names, X_design, selected_alpha)

    def _compute_model_statistics(self, y, residuals, beta, n_obs, np, stats):
        """
        Compute model fit statistics (R², adjusted R², RMSE, MAE, F-statistic).

        Returns:
            dict: Dictionary containing all model statistics
        """
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(residuals**2)
        ss_regression = ss_total - ss_residual

        df_model = len(beta) - (1 if self.model_settings.include_intercept else 0)
        df_residual = n_obs - len(beta)
        df_total = n_obs - 1

        mse = ss_residual / df_residual if df_residual > 0 else np.nan
        rmse = np.sqrt(mse)

        r_squared = ss_regression / ss_total if ss_total > 0 else 0
        adj_r_squared = 1 - (ss_residual / df_residual) / (ss_total / df_total) if df_residual > 0 else np.nan

        reg_type = self.model_settings.regression_type
        if reg_type == RegressionType.OLS.name:
            f_statistic = (ss_regression / df_model) / mse if df_model > 0 else 0
            f_pvalue = 1 - stats.f.cdf(f_statistic, df_model, df_residual)
        else:
            f_statistic = np.nan
            f_pvalue = np.nan

        mae = np.mean(np.abs(residuals))

        return {
            "ss_total": ss_total,
            "ss_residual": ss_residual,
            "mse": mse,
            "rmse": rmse,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "f_statistic": f_statistic,
            "f_pvalue": f_pvalue,
            "mae": mae,
            "df_model": df_model,
            "df_residual": df_residual,
            "n_predictors": len(beta) - (1 if self.model_settings.include_intercept else 0),
        }

    def _compute_confidence_intervals(self, beta, se_beta, df_residual, stats, np):
        """
        Compute confidence intervals for coefficients (95% confidence level).

        Returns:
            tuple: (ci_lower, ci_upper)
        """
        reg_type = self.model_settings.regression_type
        confidence_level = 0.95

        if reg_type == RegressionType.OLS.name:
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df_residual)
            ci_lower = beta - t_critical * se_beta
            ci_upper = beta + t_critical * se_beta
        else:
            ci_lower = np.full_like(beta, np.nan)
            ci_upper = np.full_like(beta, np.nan)

        return ci_lower, ci_upper

    def _build_model_summary(self, n_obs, n_predictors, stats_dict, selected_alpha, np):
        """
        Build model summary data list with all statistics.

        Returns:
            list: Model summary data for DataFrame creation
        """
        reg_type = self.model_settings.regression_type
        reg_label = RegressionType[reg_type].value[0]

        model_summary_data = [
            ("Model Type", np.nan, f"Linear Regression - {reg_label}"),
            ("Observations", float(n_obs), f"{n_obs} complete cases used in analysis"),
            ("Predictors", float(n_predictors), f"{n_predictors} independent variable(s)"),
        ]

        if self.model_settings.log_transform_y:
            model_summary_data.append(("Target Transformation", np.nan, "Natural logarithm applied to Y"))

        model_summary_data.extend(
            [
                ("R²", stats_dict["r_squared"], f"{stats_dict['r_squared']:.4f} - Proportion of variance explained"),
                ("Adjusted R²", stats_dict["adj_r_squared"], f"{stats_dict['adj_r_squared']:.4f} - Adjusted for number of predictors"),
            ]
        )

        if reg_type == RegressionType.OLS.name:
            model_summary_data.extend(
                [
                    (
                        "F-statistic",
                        stats_dict["f_statistic"],
                        f"F({stats_dict['df_model']}, {stats_dict['df_residual']}) = {stats_dict['f_statistic']:.4f}",
                    ),
                    ("F p-value", stats_dict["f_pvalue"], "Significant" if stats_dict["f_pvalue"] < 0.05 else "Not significant"),
                ]
            )
        else:
            model_summary_data.append(
                ("Optimal Alpha (CV)", selected_alpha, f"Best regularization strength selected via 5-fold CV: {selected_alpha:.6f}")
            )

        model_summary_data.extend(
            [
                ("RMSE", stats_dict["rmse"], f"Root Mean Squared Error: {stats_dict['rmse']:.4f}"),
                ("MAE", stats_dict["mae"], f"Mean Absolute Error: {stats_dict['mae']:.4f}"),
            ]
        )

        return model_summary_data

    def _compute_standardized_coefficients(self, X, y, beta, n_obs, np):
        """
        Compute standardized regression coefficients.

        Returns:
            np.ndarray: Standardized beta coefficients
        """
        reg_type = self.model_settings.regression_type

        if reg_type == RegressionType.PANEL.name:
            try:
                sd_x = np.std(X, axis=0, ddof=1)
                sd_y = np.std(y, ddof=1)
                standardized_beta = beta * (sd_x / sd_y)
            except Exception:
                standardized_beta = np.full(len(beta), np.nan)
        else:
            y_std = (y - np.mean(y)) / np.std(y, ddof=1)
            X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

            if self.model_settings.include_intercept:
                X_std_design = np.column_stack([np.ones(n_obs), X_std])
                beta_std = np.linalg.solve(X_std_design.T @ X_std_design, X_std_design.T @ y_std)
                standardized_beta = np.concatenate([[np.nan], beta_std[1:]])
            else:
                beta_std = np.linalg.solve(X_std.T @ X_std, X_std.T @ y_std)
                standardized_beta = beta_std

        return standardized_beta

    def _compute_vif(self, X, n_obs, n_predictors, np):
        """
        Calculate Variance Inflation Factors for multicollinearity assessment.

        Returns:
            tuple: (vif_values, vif_assessments) - VIF values and their interpretations
        """
        reg_type = self.model_settings.regression_type
        vif_values = []
        vif_assessments = []

        if reg_type == RegressionType.PANEL.name:
            vif_values = [np.nan] * n_predictors
            vif_assessments = ["Not available for panel"] * n_predictors
            return vif_values, vif_assessments

        if self.model_settings.include_intercept:
            vif_values.append(np.nan)
            vif_assessments.append("N/A")

        if self.model_settings.compute_vif and n_predictors >= 2 and reg_type == RegressionType.OLS.name:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X_with_intercept = np.column_stack([np.ones(n_obs), X])

            for i in range(n_predictors):
                try:
                    vif = variance_inflation_factor(X_with_intercept, i + 1)

                    if np.isnan(vif) or np.isinf(vif):
                        vif_assessments.append("Could not compute")
                    elif vif < 5:
                        vif_assessments.append("Low ✓")
                    elif vif < 10:
                        vif_assessments.append("Moderate ⚠")
                    else:
                        vif_assessments.append("High ✗")

                    vif_values.append(vif)
                except Exception:
                    vif_values.append(np.nan)
                    vif_assessments.append("Could not compute")
        elif n_predictors < 2:
            for i in range(n_predictors):
                vif_values.append(np.nan)
                vif_assessments.append("Requires ≥2 predictors")
        else:
            for i in range(n_predictors):
                vif_values.append(np.nan)
                if reg_type != RegressionType.OLS.name:
                    vif_assessments.append("Not available for regularized models")
                else:
                    vif_assessments.append("Not computed")

        return vif_values, vif_assessments

    def _compute_residual_diagnostics(self, residuals, X, mse, stats_dict, stats, np):
        """
        Compute residual diagnostic tests (Jarque-Bera, Breusch-Pagan, Durbin-Watson, Omnibus).

        Returns:
            list: Diagnostic test results to append to model summary
        """
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        jb_interp = "Residuals are normally distributed (p > 0.05)" if jb_pvalue > 0.05 else "Residuals deviate from normality (p ≤ 0.05)"

        # Breusch-Pagan test for heteroscedasticity
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan

            if self.model_settings.include_intercept:
                import statsmodels.api as sm

                X_bp = sm.add_constant(X)
            else:
                X_bp = X
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_bp)
            bp_interp = "Homoscedasticity assumption met (p > 0.05)" if bp_pvalue > 0.05 else "Heteroscedasticity detected (p ≤ 0.05)"
        except Exception:
            bp_stat = np.nan
            bp_pvalue = np.nan
            bp_interp = "Could not compute"

        # Durbin-Watson test for autocorrelation
        dw_stat = np.sum(np.diff(residuals) ** 2) / stats_dict["ss_residual"]
        dw_interp = f"Value ≈ 2 suggests no autocorrelation (DW = {dw_stat:.3f})"

        # Omnibus test for normality
        omnibus_stat, omnibus_pvalue = stats.normaltest(residuals)
        omnibus_interp = "Residuals are normally distributed (p > 0.05)" if omnibus_pvalue > 0.05 else "Residuals deviate from normality (p ≤ 0.05)"

        return [
            ("--- Diagnostic Tests ---", np.nan, "Tests of regression assumptions"),
            ("Jarque-Bera Statistic", jb_stat, jb_interp),
            ("Jarque-Bera p-value", jb_pvalue, "Normality test"),
            ("Breusch-Pagan Statistic", bp_stat, bp_interp),
            ("Breusch-Pagan p-value", bp_pvalue, "Heteroscedasticity test"),
            ("Durbin-Watson Statistic", dw_stat, dw_interp),
            ("Omnibus Statistic", omnibus_stat, omnibus_interp),
            ("Omnibus p-value", omnibus_pvalue, "Normality test"),
        ]

    def _compute_influence_diagnostics(self, residuals, standardized_residuals, leverage, n_obs, beta, mse, np):
        """
        Compute Cook's Distance and influence flags for observations.

        Returns:
            tuple: (cooks_d, leverage, flags)
        """
        reg_type = self.model_settings.regression_type

        if self.model_settings.compute_influence and reg_type == RegressionType.OLS.name:
            studentized_residuals = residuals / (np.sqrt(mse * (1 - leverage)))
            cooks_d = (studentized_residuals**2 / len(beta)) * (leverage / (1 - leverage))

            cooks_threshold = 4 / n_obs
            leverage_threshold = 2 * len(beta) / n_obs
            flags = []
            for i in range(n_obs):
                flag_parts = []
                if cooks_d[i] > cooks_threshold:
                    flag_parts.append("High Cook's D")
                if leverage[i] > leverage_threshold:
                    flag_parts.append("High Leverage")
                if np.abs(standardized_residuals[i]) > 3:
                    flag_parts.append("Outlier")
                flags.append("; ".join(flag_parts) if flag_parts else "")
        else:
            cooks_d = np.full(n_obs, np.nan)
            leverage = np.full(n_obs, np.nan)
            if reg_type == RegressionType.PANEL.name:
                flags = ["Not available for panel"] * n_obs
            elif reg_type != RegressionType.OLS.name:
                flags = ["Not available for regularized models"] * n_obs
            else:
                flags = ["Not computed"] * n_obs

        return cooks_d, leverage, flags

    def _create_diagnostic_plots(self, y_pred, residuals, standardized_residuals, cooks_d, leverage, n_obs, plt, stats, np):
        """
        Create comprehensive 6-panel diagnostic visualization.

        Returns:
            BytesIO: Buffer containing the SVG plot
        """
        from io import BytesIO

        reg_type = self.model_settings.regression_type

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Linear Regression Diagnostic Plots", fontsize=16, fontweight="bold")

        # Plot 1: Residuals vs Fitted
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", s=50)
        ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax1.set_xlabel("Fitted Values", fontsize=11)
        ax1.set_ylabel("Residuals", fontsize=11)
        ax1.set_title("Residuals vs Fitted\n(Check linearity & homoscedasticity)", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        try:
            sorted_idx = np.argsort(y_pred)
            if len(sorted_idx) > 3:
                y_pred_sorted = y_pred[sorted_idx]
                residuals_sorted = residuals[sorted_idx]
                window = max(3, len(y_pred) // 20)
                residuals_smooth = np.convolve(residuals_sorted, np.ones(window) / window, mode="valid")
                y_pred_smooth = y_pred_sorted[(window - 1) // 2 : -(window - 1) // 2]
                ax1.plot(y_pred_smooth, residuals_smooth, "b-", linewidth=2, alpha=0.8)
        except Exception:
            pass

        # Plot 2: Q-Q Plot
        ax2 = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title("Normal Q-Q Plot\n(Check normality assumption)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.get_lines()[0].set_markerfacecolor("blue")
        ax2.get_lines()[0].set_markeredgecolor("k")
        ax2.get_lines()[0].set_markersize(6)
        ax2.get_lines()[0].set_alpha(0.6)

        # Plot 3: Scale-Location
        ax3 = axes[0, 2]
        sqrt_std_resid = np.sqrt(np.abs(standardized_residuals))
        ax3.scatter(y_pred, sqrt_std_resid, alpha=0.6, edgecolors="k", s=50)
        ax3.set_xlabel("Fitted Values", fontsize=11)
        ax3.set_ylabel("√|Standardized Residuals|", fontsize=11)
        ax3.set_title("Scale-Location Plot\n(Check homoscedasticity)", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        try:
            sorted_idx = np.argsort(y_pred)
            if len(sorted_idx) > 3:
                y_pred_sorted = y_pred[sorted_idx]
                sqrt_std_resid_sorted = sqrt_std_resid[sorted_idx]
                window = max(3, len(y_pred) // 20)
                sqrt_std_resid_smooth = np.convolve(sqrt_std_resid_sorted, np.ones(window) / window, mode="valid")
                y_pred_smooth = y_pred_sorted[(window - 1) // 2 : -(window - 1) // 2]
                ax3.plot(y_pred_smooth, sqrt_std_resid_smooth, "r-", linewidth=2, alpha=0.8)
        except Exception:
            pass

        # Plot 4: Cook's Distance
        ax4 = axes[1, 0]
        if self.model_settings.compute_influence and reg_type == RegressionType.OLS.name:
            cooks_threshold = 4 / n_obs
            colors = ["red" if c > cooks_threshold else "blue" for c in cooks_d]
            ax4.bar(range(n_obs), cooks_d, color=colors, alpha=0.7, edgecolor="k")
            ax4.axhline(y=cooks_threshold, color="r", linestyle="--", linewidth=2, label=f"Threshold: {cooks_threshold:.4f}")
            ax4.set_xlabel("Observation Index", fontsize=11)
            ax4.set_ylabel("Cook's Distance", fontsize=11)
            ax4.set_title("Cook's Distance\n(Identify influential observations)", fontsize=12, fontweight="bold")
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis="y")
        else:
            if reg_type == RegressionType.OLS.name:
                msg = "Influence diagnostics\nnot computed"
            elif reg_type == RegressionType.PANEL.name:
                msg = "Not available for\npanel models"
            else:
                msg = "Not available for\nregularized models"
            ax4.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

        # Plot 5: Residual Histogram
        ax5 = axes[1, 1]
        ax5.hist(standardized_residuals, bins=min(30, n_obs // 5), density=True, alpha=0.7, color="skyblue", edgecolor="black", label="Residuals")

        x_norm = np.linspace(standardized_residuals.min(), standardized_residuals.max(), 100)
        ax5.plot(x_norm, stats.norm.pdf(x_norm, 0, 1), "r-", linewidth=2, label="Normal Distribution")
        ax5.set_xlabel("Standardized Residuals", fontsize=11)
        ax5.set_ylabel("Density", fontsize=11)
        ax5.set_title("Residual Distribution\n(Check normality)", fontsize=12, fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis="y")

        # Plot 6: Leverage vs Standardized Residuals
        ax6 = axes[1, 2]
        if self.model_settings.compute_influence and reg_type == RegressionType.OLS.name:
            leverage_threshold = 2 * len(cooks_d) / n_obs
            colors = ["red" if (lev > leverage_threshold and abs(sr) > 2) else "blue" for lev, sr in zip(leverage, standardized_residuals)]
            ax6.scatter(leverage, standardized_residuals, c=colors, alpha=0.6, edgecolors="k", s=50)
            ax6.axhline(y=0, color="gray", linestyle="-", linewidth=1)
            ax6.axhline(y=2, color="r", linestyle="--", linewidth=1)
            ax6.axhline(y=-2, color="r", linestyle="--", linewidth=1)
            ax6.axvline(x=leverage_threshold, color="r", linestyle="--", linewidth=1)
            ax6.set_xlabel("Leverage", fontsize=11)
            ax6.set_ylabel("Standardized Residuals", fontsize=11)
            ax6.set_title("Leverage vs Residuals\n(Influential outliers in top-right)", fontsize=12, fontweight="bold")
            ax6.grid(True, alpha=0.3)
        else:
            if reg_type == RegressionType.OLS.name:
                msg = "Influence diagnostics\nnot computed"
            elif reg_type == RegressionType.PANEL.name:
                msg = "Not available for\npanel models"
            else:
                msg = "Not available for\nregularized models"
            ax6.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", dpi=100)
        buf.seek(0)
        plt.close(fig)

        return buf

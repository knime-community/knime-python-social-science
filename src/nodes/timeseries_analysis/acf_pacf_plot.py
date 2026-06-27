import logging
import knime.extension as knext
from util import utils as kutil
from .timeseries_cat import timeseries_analysis_category

LOGGER = logging.getLogger(__name__)


class PACFCalculationMethod(knext.EnumParameterOptions):
    YW = ("Yule-Walker Adj", "Yule-Walker with sample-size adjustment in denominator for autocovariances. Default method.")
    YWM = ("Yule-Walker No Adj", "Yule-Walker without sample-size adjustment in denominator for autocovariances")
    OLS = ("OLS", "Regression of time series on lags of it and on constant.")
    OLS_INEFFICIENT = ("OLS Inefficient", "Regression of time series on lags using a single common sample to estimate all pacf coefficients.")
    OLS_ADJUSTED = ("OLS Adjusted", "Regression of time series on lags with a bias adjustment.")
    LD = ("LD", "Levinson-Durbin recursion with bias correction.")
    LDB = ("LDB", "Levinson-Durbin recursion without bias correction.")


class WhichPLot(knext.EnumParameterOptions):
    BOTH = ("Both", "Generate both ACF and PACF plots.")
    ACF = ("ACF", "Autocorrelation Function plot only.")
    PACF = ("PACF", "Partial Autocorrelation Function plot only.")


@knext.node(
    name="(Partial) Autocorrelation Functions and Plots",
    node_type=knext.NodeType.VISUALIZER,
    icon_path="icons/AutocorrelationPlot.png",
    category=timeseries_analysis_category,
    id="acf_pacf_plot",
    keywords=[
        "ACF",
        "PACF",
        "Time Series",
        "Forecasting",
        "Seasonal",
    ],
)
@knext.input_table(
    name="Input Data",
    description="Time series data table. The selected target column must be numeric and contain no missing values.",
)
@knext.output_table(
    name="PACF and ACF Function Values and Confidence Intervals",
    description="Table containing the PACF and ACF function values with upper and lower confidence bounds based on the specified confidence level. Length of the table corresponds to the number of lags specified + 1.",
)
@knext.output_table(
    name="Ljung-Box Q-statistic",
    description="Table containing the Ljung-Box Q-statistic for autocorrelation parameters and p-values. Length of the table corresponds to the number of lags specified.",
)
@knext.output_image(
    name="PACF/ACF Plot",
    description="Partial Autocorrelation Function (PACF) and/or Autocorrelation Function (ACF) plot(s) showing the correlation structure of the selected time series data.",
)
class AcfPacfPlot:
    """
    Computes and visualizes autocorrelation diagnostics for a univariate time series.

    **ACF (Autocorrelation Function)** measures correlation between the series and its lagged values.
    **PACF (Partial Autocorrelation Function)** measures correlation at a lag after removing the effect
    of intermediate lags. These plots are commonly used for identifying AR/MA structure and for general
    time-series diagnostics.

    ## Inputs
    - **Input Data**: A table containing at least one numeric column.

    ## Key constraints
    - The selected **Target Column** must contain **no missing values**.
    - The **Number of Lags** must be less than the number of observations (`nlags <= n-1`).

    ## Parameters
    - **Target Column**: Numeric column used to compute ACF/PACF.
    - **Number of Lags**: Maximum lag to compute. For seasonal data, lags that are multiples of the
    seasonal period are often informative.
    - **Which Plot to Generate**: ACF only, PACF only, or both.
    - **Confidence Level**: Confidence level for the correlation bounds (e.g., 0.95 → 95% intervals).
    - **Adjusted Autocovariance**: If enabled, uses `n - lag` in the autocovariance denominator;
    otherwise uses `n`.
    - **PACF Calculation Method**: Estimators include varions of Yule-Walker, OLS and
    Levinson-Durbin).

    ## Outputs
    1. **(P)ACF Values + Confidence Bounds**: Table containing ACF and PACF values and their
    lower/upper confidence bounds for lags `0..nlags`.
    2. **Ljung-Box Q-statistic**: Table containing Ljung-Box Q-statistics and p-values (computed from ACF),
    useful for testing residual autocorrelation.
    3. **PACF/ACF Plot**: SVG plot of ACF, PACF, or both.

    ## Notes
    - Interpretation is context dependent: significant spikes may indicate lag structure, but
    non-stationarity and seasonality can also drive large correlations.
    """

    input_column = knext.ColumnParameter(
        label="Target Column",
        description="Numeric time series column used to compute ACF and/or PACF. Must contain no missing values.",
        column_filter=kutil.is_numeric,
    )
    number_of_lags = knext.IntParameter(
        label="Number of Lags",
        description="Number of lags to include in the ACF and PACF plots. For non-seasonal data, 20 to 45 lags are typically sufficient, provided the number of observations is large enough. For seasonal data, consider using a number of lags that is a multiple of the seasonal period (2 or 3 times the seasonal period is usually enough).",
        default_value=20,
        min_value=1,
    )
    which_plot = knext.EnumParameter(
        label="Which Plot to Generate",
        description="Select which plot(s) to generate: ACF, PACF, or both (default).",
        default_value=WhichPLot.BOTH.name,
        enum=WhichPLot,
    )
    plot_title = knext.StringParameter(
        label="Plot Title",
        description="Title prefix for the plot(s). The node appends 'ACF - x lags' and/or 'PACF - x lags' depending on the plot selection.",
        default_value="Plot",
    )
    confidence_level = knext.DoubleParameter(
        label="Confidence Level",
        description="Confidence level for the ACF/PACF confidence bounds (e.g., 0.95 produces 95% bounds). Higher values produce wider bounds.",
        default_value=0.95,
        min_value=0.01,
        max_value=0.99,
        is_advanced=True,
    )
    adjusted = knext.BoolParameter(
        label="Adjusted Autocovariance",
        description="If True, then denominators for autocovariance are number of observations - lags, otherwise number of observations. Default is False.",
        default_value=False,
        is_advanced=True,
    )
    pacf_method = knext.EnumParameter(
        label="PACF Calculation Method",
        description="Method to use for PACF calculation. Options include Yule-Walker, OLS, and Levinson-Durbin variants. Default is Yule-Walker with sample-size adjustment.",
        default_value=PACFCalculationMethod.YW.name,
        enum=PACFCalculationMethod,
        is_advanced=True,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema) -> knext.Schema:
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.input_column = kutil.column_exists_or_preset(
            configure_context, self.input_column, input_schema, kutil.is_numeric, "Please select a numeric column for PACF/ACF plotting."
        )

        function_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double(), knext.double(), knext.double(), knext.double()],
            ["ACF Function Values", "ACF Lower CI", "ACF Upper CI", "PACF Function Values", "PACF Lower CI", "PACF Upper CI"],
        )
        qstat_schema = knext.Schema(
            [knext.double(), knext.double()],
            ["Q-Statistic", "p-Value"],
        )

        return (
            function_schema,
            qstat_schema,
            knext.ImagePortObjectSpec(knext.ImageFormat.SVG),
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        # Import heavy dependencies
        import pandas as pd
        from statsmodels.tsa.stattools import acf, pacf
        import matplotlib.pyplot as plt
        from io import BytesIO

        exec_context.set_progress(0.1)

        df = input_table.to_pandas()
        target_col = df[self.input_column]

        exec_context.set_progress(0.1)

        # Check for missing values
        kutil.validate_missing_values(target_col)

        # check if the number of lags falls within acceptable range
        n = len(target_col)
        if self.number_of_lags > n - 1:
            raise knext.InvalidParametersError(
                f"The number of observations in the target column ({n}) must be greater than the number of lags ({self.number_of_lags})."
            )

        exec_context.set_progress(0.2)

        # Calculate ACF
        acf_values, confidence_intervals, qstat_values, pvalues = acf(
            target_col, nlags=self.number_of_lags, alpha=1 - self.confidence_level, adjusted=self.adjusted, qstat=True
        )

        exec_context.set_progress(0.3)

        # Format correctly PACF calculation method string
        pacf_method = str(self.pacf_method).lower().replace("_", "-")

        # Calculate PACF
        pacf_values, pacf_conf_intervals = pacf(target_col, nlags=self.number_of_lags, alpha=1 - self.confidence_level, method=pacf_method)

        exec_context.set_progress(0.4)

        # Prepare output DataFrame for ACF values
        acf_values_df = pd.DataFrame(
            {
                "ACF Function Values": acf_values,
                "ACF Lower CI": confidence_intervals[:, 0],
                "ACF Upper CI": confidence_intervals[:, 1],
                "PACF Function Values": pacf_values,
                "PACF Lower CI": pacf_conf_intervals[:, 0],
                "PACF Upper CI": pacf_conf_intervals[:, 1],
            }
        )
        acf_values_df = acf_values_df.astype("float64")
        qstat_df = pd.DataFrame(
            {
                "Q-Statistic": qstat_values,
                "p-Value": pvalues,
            }
        )
        qstat_df = qstat_df.astype("float64")

        exec_context.set_progress(0.5)

        # Plots
        fig = self.__generate_plots(exec_context, target_col, pacf_method, plt)

        exec_context.set_progress(0.7)

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        exec_context.set_progress(1.0)

        return (
            knext.Table.from_pandas(acf_values_df),
            knext.Table.from_pandas(qstat_df),
            buf.getvalue(),
        )

    def __generate_plots(self, exec_context, target_col, pacf_method, plt):
        """
        Generates ACF and/or PACF plots based on user selection.

        Parameters:
        - target_col: pd.Series
            The time series data for which to generate the plots.
        - pacf_method: str
            Method for PACF calculation.
        - plt: module
            Matplotlib pyplot module for plotting.

        Returns:
        - fig: matplotlib.figure.Figure
            The generated plot figure.
        """

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        if self.which_plot != WhichPLot.BOTH.name:
            fig, ax = plt.subplots(figsize=(12, 6))
            if self.which_plot == WhichPLot.ACF.name:
                # ACF only
                from statsmodels.graphics.tsaplots import plot_acf

                plot_acf(
                    target_col,
                    lags=self.number_of_lags,
                    ax=ax,
                    alpha=1 - self.confidence_level,
                    adjusted=self.adjusted,
                )
                ax.set_title(f"{self.plot_title} ACF - {self.number_of_lags} lags")
            else:
                # PACF only
                from statsmodels.graphics.tsaplots import plot_pacf

                plot_pacf(
                    target_col,
                    lags=self.number_of_lags,
                    ax=ax,
                    alpha=1 - self.confidence_level,
                    method=pacf_method,
                )
                ax.set_title(f"{self.plot_title} PACF - {self.number_of_lags} lags")
            return fig

        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        plot_acf(
            target_col,
            lags=self.number_of_lags,
            ax=axes[0],
            alpha=1 - self.confidence_level,
            adjusted=self.adjusted,
        )
        axes[0].set_title(f"{self.plot_title} ACF - {self.number_of_lags} lags")

        exec_context.set_progress(0.6)

        plot_pacf(
            target_col,
            lags=self.number_of_lags,
            ax=axes[1],
            alpha=1 - self.confidence_level,
            method=pacf_method,
        )
        axes[1].set_title(f"{self.plot_title} PACF - {self.number_of_lags} lags")

        return fig

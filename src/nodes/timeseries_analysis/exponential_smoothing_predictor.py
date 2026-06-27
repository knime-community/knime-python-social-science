import logging
import knime.extension as knext
from .timeseries_cat import timeseries_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="ETS Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/TimeSeriesPredictor.png",
    category=timeseries_analysis_category,
    id="ets_predictor",
    keywords=[
        "Exponential Smoothing",
        "ETS",
        "State Space",
        "Time Series",
        "Forecasting",
        "Seasonal",
    ],
)
@knext.input_binary(
    name="Model Input",
    description="A pickled ETS results object output from the ETS Learner node (statsmodels ETSResults). This object contains all information necessary to generate forecasts.",
    id="est.model",
)
@knext.output_table(
    name="Forecasts",
    description="Table containing the generated forecast values with confidence intervals. The forecast starts one period after the end of the data used to train the input model. Includes point forecasts and upper/lower confidence bounds based on the specified confidence level.",
)
@knext.output_image(
    name="Forecast Plot",
    description="Time series plot showing historical data, forecasts, and confidence intervals. Historical data displayed as a solid line, forecasts as points, and confidence intervals as shaded area.",
)
class ExponentialSmoothingPredictor:
    """
    Generates out-of-sample forecasts from a fitted ETS (Error-Trend-Seasonal) model.

    This predictor consumes the pickled ETS model produced by **ETS Learner**
    and produces point forecasts plus prediction intervals (uncertainty bounds).

    ## Inputs
    - **Model Input**: Pickled ETS results object from the learner node.

    ## Parameters
    - **Forecasts**: Number of future steps to predict (minimum 1).
    - **Confidence Level**: Coverage for prediction intervals (e.g., 0.95 → 95% intervals).
    - **Plot Title**: Title shown on the generated forecast plot.

    ## Outputs
    1. **Forecasts**: Table with:
    - point forecasts,
    - lower prediction bound,
    - upper prediction bound,
    for the requested number of steps.
    2. **Forecast Plot**: SVG plot showing:
    - the historical series (from the fitted model),
    - forecast points,
    - shaded prediction interval band.

    ## Notes
    - Forecasts start immediately after the final observation used to fit the model.
    - Prediction intervals reflect the model's estimated error structure (additive/multiplicative).
    """

    number_of_forecasts = knext.IntParameter(
        label="Forecasts",
        description="Specifies the number of future time periods for which to generate forecasts. For example, if the training data ended at time T, a value of 12 here would forecast for T+1, T+2, ..., T+12.",
        default_value=1,
        min_value=1,
    )
    boxcox = knext.BoolParameter(
        label="Reverse Box-Cox transformation",
        description="Select this option if the original data was Box-Cox transformed within the ETS Learner node during model training. This will apply a revese Box-Cox transformation with the same lambda parameter to the forecasted values to revert them to their original scale. Ensure that the lambda parameter is the same used in the Learner node (reported in the Coefficients and Statistics output table if Box-Cox was enabled).",
        default_value=False,
    )
    lambda_value = knext.DoubleParameter(
        label="Box-Cox lambda",
        description="Enter the lambda parameter used for Box-Cox transformation in the ETS Learner node. Regardless of whether the lambda was estimated or user-defined in the Learner node, the same value is required. It is found in the Coefficients and Statistics output table if Box-Cox was enabled.",
        default_value=0.0,
    ).rule(
        knext.OneOf(boxcox, [False]),
        knext.Effect.HIDE,
    )
    confidence_level = knext.DoubleParameter(
        label="Confidence Level",
        description="The confidence level for prediction intervals, expressed as a decimal between 0.01 and 0.99. For example, 0.95 produces 95% confidence intervals. Higher values create wider intervals, while lower values create narrower intervals.",
        default_value=0.95,
        min_value=0.01,
        max_value=0.99,
    )
    plot_title = knext.StringParameter(
        label="Plot Title",
        description="Title for the forecast plot, set automatically based on the fitted model: (ETS(<short_name>) Forecast with Confidence Intervals).",
        default_value="ETS Forecast Plot with Confidence Intervals",
    )

    def configure(self, _configure_context, _input_model) -> knext.Schema:
        # Create schema with forecasts and confidence intervals
        forecast_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double()],
            ["Forecasts", "Lower_CI", "Upper_CI"],
        )

        return (
            forecast_schema,
            knext.ImagePortObjectSpec(knext.ImageFormat.SVG),
        )

    def execute(self, exec_context: knext.ExecutionContext, input_model):
        # Import heavy dependencies
        import pickle
        import pandas as pd
        import matplotlib.pyplot as plt
        from io import BytesIO
        from util.utils import inv_box_cox_transform

        exec_context.set_progress(0.1)

        trained_model = pickle.loads(input_model)

        exec_context.set_progress(0.3)

        # Extract point forecasts and confidence intervals
        point_forecast = trained_model.forecast(steps=self.number_of_forecasts)

        confidence_intervals = trained_model.get_prediction(start=point_forecast.index[0], end=point_forecast.index[-1])

        exec_context.set_progress(0.6)

        # Create output DataFrame
        output_df = pd.DataFrame(
            {
                "Forecasts": point_forecast.values,
                "Lower_CI": confidence_intervals.pred_int(alpha=1 - self.confidence_level).iloc[:, 0].values,
                "Upper_CI": confidence_intervals.pred_int(alpha=1 - self.confidence_level).iloc[:, 1].values,
            }
        )

        # Reverse box-cox transformation for forecasts and confidence intervals
        if self.boxcox:
            output_df["Forecasts"] = inv_box_cox_transform(output_df["Forecasts"], self.lambda_value)
            output_df["Lower_CI"] = inv_box_cox_transform(output_df["Lower_CI"], self.lambda_value)
            output_df["Upper_CI"] = inv_box_cox_transform(output_df["Upper_CI"], self.lambda_value)

        # Ensure all columns are float64 to match schema
        output_df = output_df.astype("float64")

        exec_context.set_progress(0.8)

        # Create forecast plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get real historical data from the trained model
        historical_data = trained_model.data.endog
        if self.boxcox:
            historical_data = inv_box_cox_transform(historical_data, self.lambda_value)

        # Create time index starting from 0
        historical_index = range(len(historical_data))
        forecast_index = range(len(historical_data), len(historical_data) + self.number_of_forecasts)

        # Plot historical data
        ax.plot(historical_index, historical_data, "b-", label="Historical Data", linewidth=1.5)

        # Plot forecasts
        ax.plot(forecast_index, output_df["Forecasts"], "ro-", label="Forecasts", markersize=4)

        # Plot confidence intervals as shaded area
        confidence_pct = int(self.confidence_level * 100)
        ax.fill_between(
            forecast_index, output_df["Lower_CI"], output_df["Upper_CI"], alpha=0.3, color="red", label=f"{confidence_pct}% Confidence Interval"
        )

        title_str = self.plot_title if self.plot_title else f"ETS({trained_model.model.short_name}) Forecast with Confidence Intervals"

        ax.set_title(title_str)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        exec_context.set_progress(1.0)

        return (
            knext.Table.from_pandas(output_df),
            buf.getvalue(),
        )

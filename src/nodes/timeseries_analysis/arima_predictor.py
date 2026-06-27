import logging
import knime.extension as knext
from .timeseries_cat import timeseries_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Auto-SARIMA Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/TimeSeriesPredictor.png",
    category=timeseries_analysis_category,
    id="auto_sarima_predictor",
    keywords=[
        "ARIMA",
        "SARIMA",
        "Time Series",
        "Forecasting",
        "Seasonal",
        "Autoregressive",
    ],
)
@knext.input_binary(
    name="Model Input",
    description="A pickled model, output from the Auto-SARIMA Learner node. This model contains all the information necessary to generate forecasts.",
    id="auto_sarima.model",
)
@knext.output_table(
    name="Forecasts",
    description="Table containing the generated forecast values with confidence intervals. The forecast starts one period after the end of the data used to train the input model. Includes point forecasts and upper/lower confidence bounds based on the specified confidence level.",
)
@knext.output_image(
    name="Forecast Plot",
    description="Time series plot showing historical data, forecasts, and confidence intervals. Historical data displayed as a solid line, forecasts as points, and confidence intervals as shaded area.",
)
class SarimaForcaster:
    """
    Generates future forecasts with confidence intervals using a pre-trained SARIMA model, output of the node Auto-SARIMA Learner.

    This node takes the pre-trained SARIMA model and produces out-of-sample forecasts for a specified number of future periods.
    The forecasts are generated directly from the end of the training data used to fit the input model, with confidence intervals providing uncertainty bounds around the point predictions.

    **Parameters & Behavior:**

    -   `Model Input` is the pickled `SARIMAX` model produced by the Auto-SARIMA Learner node. This model is used to generate the forecasts.

    -   `Forecast` allows to chose the number of forecast periods, minimum is 1.

    -   `Confidence Level` sets the confidence level for prediction intervals (0.01 to 0.99). Higher values create wider intervals.

    -   If a log transformation was applied during training (in the Learner node), the "Reverse Log" option must be checked here to ensure forecasts and confidence intervals are on the original scale.

    -   `Plot Title` allows to set the title of the forecast plot.

    **Outputs:**

    1.  Table with the forecasted values, lower confidence bounds, and upper confidence bounds.

    """

    number_of_forecasts = knext.IntParameter(
        label="Forecasts",
        description="Specifies the number of future time periods for which to generate forecasts. For example, if the training data ended at time T, a value of 12 here would forecast for T+1, T+2, ..., T+12.",
        default_value=1,
        min_value=1,
    )
    natural_log = knext.BoolParameter(
        label="Reverse Log",
        description="Select this option if the original data was log-transformed within the Auto-SARIMA Learner node during model training. This will apply an exponential function (np.exp) to the forecasted values to revert them to their original scale. Ensure this matches the transformation setting used in the Learner node.",
        default_value=False,
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
        description="Title for the forecast plot.",
        default_value="SARIMA Forecast Plot with Confidence Intervals",
    )

    def configure(self, _configure_context, _input_model):
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
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from io import BytesIO

        exec_context.set_progress(0.1)

        trained_model = pickle.loads(input_model)

        exec_context.set_progress(0.3)

        # Generate forecasts with confidence intervals using get_forecast
        forecast_result = trained_model.get_forecast(steps=self.number_of_forecasts)

        # Extract point forecasts and confidence intervals
        point_forecasts = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int(alpha=1 - self.confidence_level)

        exec_context.set_progress(0.6)

        # Create output DataFrame
        output_df = pd.DataFrame(
            {
                "Forecasts": point_forecasts.values,
                "Lower_CI": confidence_intervals.iloc[:, 0].values,
                "Upper_CI": confidence_intervals.iloc[:, 1].values,
            }
        )

        # Reverse log transformation for forecasts and confidence intervals
        if self.natural_log:
            output_df["Forecasts"] = np.exp(output_df["Forecasts"])
            output_df["Lower_CI"] = np.exp(output_df["Lower_CI"])
            output_df["Upper_CI"] = np.exp(output_df["Upper_CI"])

        # Ensure all columns are float64 to match schema
        output_df = output_df.astype("float64")

        exec_context.set_progress(0.8)

        # Create forecast plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get real historical data from the trained model
        historical_data = trained_model.data.endog
        if self.natural_log:
            historical_data = np.exp(historical_data)

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

        ax.set_title(f"{self.plot_title}")
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

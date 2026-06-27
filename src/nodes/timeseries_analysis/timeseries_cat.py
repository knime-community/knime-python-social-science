import knime.extension as knext

# NOTE: Categories currently don't work in modern UI.
# This is a bug that needs to be fixed in the KNIME Analytics Platform.

timeseries_analysis_category = knext.category(
    path="/community/socialscience",
    name="Time Series Analysis",
    level_id="timeseries_analysis",
    description="Nodes for time series analysis and forecasting",
    icon="icons/TimeSeriesLearner.png",
)

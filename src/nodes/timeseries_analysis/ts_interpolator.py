import logging
import knime.extension as knext
from util import utils as kutil
from .timeseries_cat import timeseries_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.parameter_group(label="Box-Cox settings")
class BoxCoxSettings:
    """
    Grouped parameters for Box-Cox transformation settings.
    """

    enabled = knext.BoolParameter(
        label="Use Box-Cox transformation",
        description="Apply Box-Cox transformation before interpolation. Requires all observed values to be positive.",
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
    name="Time Series Interpolator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/TimeSeriesInterpolator.png",
    category=timeseries_analysis_category,
    id="time_series_interpolator",
    keywords=[
        "Interpolation",
        "Missing Values",
        "Time Series",
        "Forecasting",
        "Seasonal",
    ],
)
@knext.input_table(
    name="Input Data",
    description="Table containing a numeric time series column with missing values to interpolate.",
)
@knext.output_table(
    name="Output Data",
    description="Single-column output table containing the interpolated series '<Target Column> interpolated'. If the selected column has no missing values, the input table is returned unchanged.",
)
class TimeSeriesInterpolator:
    """
    Interpolates missing values in a univariate time series, following the structure of R forecast::na.interp.

    The node provides two main interpolation behaviors:
    - **Non-seasonal** (seasonality = 1): linear interpolation with endpoint extension.
    - **Seasonal** (seasonality > 1): robust STL decomposition after an initial regression-based prefill,
    then interpolation on the seasonally adjusted series, and finally re-adding seasonality.

    An optional Box-Cox transformation can be applied before interpolation to stabilize variance.

    ## Inputs
    - **Input Data**: A table containing a numeric time series column (may contain missing values).

    ## Parameters
    - **Target Column**: Numeric series to interpolate (missing values allowed).
    - **Seasonality**:
    - 1 means “treat as non-seasonal” (pure linear interpolation),
    - values > 1 enable the seasonal STL-based path.
    - **Box-Cox settings** (optional):
    - Can estimate lambda automatically from observed values,
    - or use a user-provided lambda.
    - Requires observed values to be strictly positive.

    ## Algorithm overview (seasonal case)
    1. **Prefill** missing values using **Fourier terms + orthogonal polynomial regression**
    (to provide sensible starting values before decomposition).
    2. Run **robust STL** (Seasonal-Trend decomposition using Loess) on the prefilled series.
    3. Remove the estimated seasonal component, interpolate remaining signal linearly,
    then add seasonality back.
    4. Optional **stability fallback**:
    if interpolated values exceed a conservative range based on the original data,
    fall back to non-seasonal linear interpolation.

    ## Outputs
    - **Output Data**: A table containing the interpolated series (one column named
    “<Target Column> interpolated”).

    ## Notes
    - If the selected column has no missing values, the node returns the input unchanged and emits a warning.
    - If all values are missing, interpolation is not possible and the node raises an error.
    - If seasonality is large relative to the number of observations (e.g., < 2 seasonal cycles),
    the node automatically falls back to the non-seasonal interpolation path.
    """

    # Parameters
    input_column = knext.ColumnParameter(
        label="Target Column",
        description="Numeric time series column to interpolate. Must contain at least one non-missing value.",
        port_index=0,
        column_filter=kutil.is_numeric,
    )
    seasonality = knext.IntParameter(
        label="Seasonality",
        description="Specify the length of the seasonal period for your time series data. Seasonality of 1 (default) indicates no seasonality.",
        default_value=1,
        min_value=1,
    )

    boxcox = BoxCoxSettings()

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema) -> knext.Schema:
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.input_column,
            input_schema,
            kutil.is_numeric,
        )

        interpolation_schema = knext.Schema(
            [knext.double()],
            [f"{self.input_column} interpolated"],
        )

        return interpolation_schema

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        # Import heavy dependencies
        import pandas as pd
        import numpy as np

        df = input_table.to_pandas()
        input_column = df[self.input_column]
        input_column_series = pd.Series(input_column, dtype=np.float64)

        if not kutil.check_missing_values(input_column_series):
            exec_context.set_warning("The selected column has no missing values; no interpolation needed.")
            return knext.Table.from_pandas(df, row_ids="keep")

        if kutil.count_missing_values(input_column_series) == len(input_column_series):
            raise knext.InvalidParametersError("The selected column has only missing values; interpolation is not possible.")

        if len(input_column_series) < self.seasonality:
            raise knext.InvalidParametersError(
                f"The number of observations in the target column ({len(input_column_series)}) is lower than the seasonal period ({self.seasonality})."
            )

        use_boxcox = self.boxcox.enabled
        lambda_ = None
        if use_boxcox and not self.boxcox.estimate_lambda:
            lambda_ = self.boxcox.lambda_value

        exec_context.set_progress(0.1)

        interpolated_series = self.__interpolate_series(
            input_column_series,
            seasonality=self.seasonality,
            use_boxcox=use_boxcox,
            lambda_=lambda_,
            stability_check=True,
            np=np,
            pd=pd,
        )

        exec_context.set_progress(0.9)

        output_df = pd.DataFrame({f"{self.input_column} interpolated": interpolated_series})
        # Ensure all columns are float64 to match schema
        output_df = output_df.astype("float64")

        return knext.Table.from_pandas(output_df, row_ids="keep")

    def __linear_interp(self, x, missing, np):
        """
        Linear interpolation with endpoint extension.
        Equivalent to R approx(..., rule=2) on positions 1..n.
        Returns interpolated values at missing positions.
        """
        n = x.size
        idx = np.arange(n, dtype=float)
        known = ~missing
        return np.interp(idx[missing], idx[known], x[known])

    def __fourier_matrix(self, n: int, period: int, K: int, np):
        """
        Approximation of forecast::fourier(x, K) for a single seasonal period.
        Returns shape (n, 2K): [sin(2πkt/m), cos(2πkt/m)] for k=1..K.
        """
        if K <= 0:
            return np.empty((n, 0), dtype=float)

        t = np.arange(1, n + 1, dtype=float)  # R uses tt <- 1:n
        cols = []
        for k in range(1, K + 1):
            cols.append(np.sin(2.0 * np.pi * k * t / period))
            cols.append(np.cos(2.0 * np.pi * k * t / period))
        return np.column_stack(cols).astype(float)

    def __orthogonal_poly_matrix(self, n: int, degree: int, np):
        """
        Approximate stats::poly(tt, degree=...) (orthogonal polynomials).
        R's poly() uses an orthogonal polynomial basis; here we:
        - scale t to roughly improve conditioning
        - build raw powers
        - orthonormalize columns via QR
        """
        degree = int(degree)
        if degree <= 0:
            return np.empty((n, 0), dtype=float)

        t = np.arange(1, n + 1, dtype=float)
        t = (t - t.mean()) / (t.std() if t.std() > 0 else 1.0)

        V = np.column_stack([t**p for p in range(1, degree + 1)])  # (n, degree)
        Q, _ = np.linalg.qr(V)  # Q has orthonormal columns
        return Q[:, :degree].astype(float)

    def __fourier_poly_prefill(self, x, missing, seasonality: int, np):
        """
        Replicates the 'Fourier + poly regression' prefill step from forecast::na.interp
        to obtain reasonable starting values before STL.
        x: array with NaNs at missing positions (possibly Box-Cox transformed already)
        missing: boolean mask of original missing values
        seasonality: seasonal period length
        """
        from statsmodels.tools.tools import add_constant
        from statsmodels.regression.linear_model import OLS

        n = x.size
        freq = int(seasonality)

        # Mirror R choices for single-seasonal ts:
        K = min(freq // 2, 5)  # K <- min(trunc(freq/2), 5)
        degree = min(max(n // 10, 1), 6)  # degree <- pmin(pmax(trunc(n/10),1),6)

        F = self.__fourier_matrix(n=n, period=freq, K=K, np=np)
        P = self.__orthogonal_poly_matrix(n=n, degree=degree, np=np)
        X = np.column_stack([F, P])  # (n, p)

        # Add intercept like lm() does
        Xc = add_constant(X, has_constant="add")

        y_obs = x[~missing]
        X_obs = Xc[~missing, :]

        # If something goes wrong (rare), fall back to linear prefill
        try:
            fit = OLS(y_obs, X_obs).fit()
            pred = fit.predict(Xc)
            out = x.copy()
            out[missing] = pred[missing]
            return out
        except Exception:
            out = x.copy()
            out[missing] = self.__linear_interp(out, missing, np)
            return out

    def __interpolate_series(
        self,
        s,
        seasonality: int,
        use_boxcox: bool = False,
        lambda_: float | None = None,
        stability_check: bool = True,
        np=None,
        pd=None,
    ):
        """
        Interpolate missing values in a pandas Series.

        - If seasonal=False: linear interpolation (with endpoint extension).
        - If seasonal=True: robust STL -> interpolate seasonally-adjusted series -> add seasonal back.

        Notes:
        - STL cannot run with NaNs, so an initial Fourier + poly regression prefill before robust STL decomposition is used.
        - Box-Cox (lambda_) is optional; you should only use it if observed values are > 0.
        """
        from statsmodels.tsa.seasonal import STL

        idx = s.index
        name = s.name

        x = s.to_numpy(dtype=float)  # series turned into numpy array
        miss = np.isnan(x)  # Boolean mask of missing values
        origx = x.copy()  # Create a copy of the original array

        # Save original scale range for optional stability fallback
        minx = np.nanmin(x)  # The minimum value of an array along a given axis, ignoring any NaNs.
        maxx = np.nanmax(x)  # The maximum value of an array along a given axis, ignoring any NaNs.
        drange = maxx - minx  # The range of the data, ignoring any NaNs.

        # Optional Box-Cox transform
        if use_boxcox:
            x[~miss], lambda_ = kutil.box_cox_transform(x[~miss], LOGGER, lambda_)

        # Non-seasonal path: pure linear interpolation when seasonality == 1 or too few non-missing values
        if (seasonality == 1) or (sum(~miss) <= 2 * seasonality):
            x[miss] = self.__linear_interp(x, miss, np)
            if use_boxcox:
                origx[miss] = kutil.inv_box_cox_transform(x[miss], lambda_)
                return pd.Series(origx, index=idx, name=name)
            return pd.Series(x, index=idx, name=name)

        # Seasonal path: STL on initially-filled data
        # Fourier + poly regression prefill before robust STL
        x = self.__fourier_poly_prefill(x, miss, seasonality=seasonality, np=np)
        res = STL(x, period=seasonality, robust=True).fit()
        seas = res.seasonal  # seasonal component

        sa = x - seas  # seasonally adjusted (remove seasonal component, sa := trend + remainder)
        sa[miss] = self.__linear_interp(sa, miss, np)
        origx[miss] = sa[miss] + seas[miss]

        # Backtransform if needed
        if use_boxcox:
            origx[miss] = kutil.inv_box_cox_transform(x[miss], lambda_)

        # Optional stability fallback (same spirit as forecast::na.interp)
        if stability_check and drange > 0:
            if np.nanmax(origx) > maxx + 0.5 * drange or np.nanmin(origx) < minx - 0.5 * drange:
                # fall back to linear on original input
                return self.__interpolate_series(s, seasonality=1, use_boxcox=use_boxcox, lambda_=lambda_, stability_check=False, np=np, pd=pd)

        return pd.Series(origx, index=idx, name=name)

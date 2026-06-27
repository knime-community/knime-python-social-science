import logging
import knime.extension as knext
from util import utils as kutil
from .multivariate_cat import multivariate_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.parameter_group(
    label="Rotation Settings",
)
class RotationSettings:
    """
    Rotation methods to enhance component/factor interpretability. Rotations redistribute variance among components to achieve simpler structure with clearer variable-component relationships.
    """

    class RotationMethods(knext.EnumParameterOptions):
        NO_ROTATION = ("None", "No rotation applied - use raw unrotated loadings. Components remain orthogonal but may be difficult to interpret.")
        VARIMAX = (
            "Varimax",
            "Orthogonal rotation maximizing variance of squared loadings within components. Creates simple structure where each variable loads highly on few components. Most popular choice.",
        )
        PROMAX = (
            "Promax",
            "Oblique rotation allowing correlated components. First applies Varimax, then relaxes orthogonality constraint. Use when components are expected to correlate.",
        )
        QUARTIMAX = (
            "Quartimax",
            "Orthogonal rotation maximizing variance of squared loadings across variables. Emphasizes general factors. Less commonly used than Varimax.",
        )

    rotation_method = knext.EnumParameter(
        label="Select a rotation method",
        description="Choose a rotation method to apply to the analysis results.",
        default_value=RotationMethods.NO_ROTATION.name,
        enum=RotationMethods,
    )


class AnalysisMethod(knext.EnumParameterOptions):
    STANDARD = ("Standard PCA", "Standard Principal Component Analysis.")
    INCREMENTAL = ("Incremental PCA", "Incremental Principal Component Analysis (IncrementalPCA) for large datasets.")
    FACTOR_ANALYSIS = ("Exploratory Factor Analysis", "Exploratory Factor Analysis using maximum likelihood estimation to find latent factors.")


@knext.node(
    name="Factor Analyzer",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/FactorAnalyzer.png",
    category=multivariate_analysis_category,
    keywords=[
        "Factor Analysis",
        "PCA",
        "Principal Component",
        "Dimensionality Reduction",
        "Varimax",
        "Exploratory Factor Analysis",
    ],
    id="factor_analysis",
)
@knext.input_table(
    name="Input Data",
    description="Numeric data table for dimensionality reduction or factor analysis. Must contain at least two numeric columns with sufficient observations. Missing values are automatically handled by row-wise deletion.",
)
@knext.output_table(
    name="Model Fit",
    description="Model performance metrics and variance decomposition: eigenvalues and explained variance ratios (PCA methods), log-likelihood values (Factor Analysis). Used to assess model quality and determine optimal component count.",
)
@knext.output_table(
    name="Component Loadings",
    description="Variable-component relationships matrix showing rotated loadings, communalities (shared variance), and noise variance (unique variance). Essential for interpreting which variables define each component/factor.",
)
@knext.output_binary(
    name="Model",
    description="Trained dimensionality reduction model object with rotation matrices and preprocessing parameters. Ready for transforming new data with predictor nodes.",
    id="factor_analysis.model",
)
class FactorAnalysisNode:
    """
    ## Advanced Dimensionality Reduction and Factor Analysis

    Performs Principal Component Analysis (PCA) and Exploratory Factor Analysis (EFA) to extract latent structure from multivariate datasets. Reduces dimensionality while preserving interpretable patterns through advanced rotation methods.

    ## Analysis Methods

    - **PCA**: Orthogonal decomposition finding components that maximize variance
    - **Incremental PCA**: Memory-efficient streaming PCA for large datasets
    - **Exploratory Factor Analysis**: Maximum likelihood estimation of latent factors
    - **Rotation Options**: Varimax, Promax, Quartimax for enhanced interpretability

    ## Configuration Options

    - **Analysis Method**: Choose between PCA variants and factor analysis
    - **Component Count**: Manual specification or automatic selection
    - **Variable Selection**: Multi-column picker with numeric filtering
    - **Standardization**: Optional z-score normalization
    - **Rotation Method**: Orthogonal or oblique rotation for interpretability

    ## Output Components

    **Model Fit Statistics**:
    - Eigenvalues and explained variance ratios (PCA methods)
    - Log-likelihood values (Factor Analysis)
    - Component selection diagnostics

    **Component Loadings Matrix**:
    - Variable-component relationship coefficients (rotated or unrotated loadings)
    - Communalities (captured variance of each variable)
    - Noise variance (unexplained variance of each variable)

    **Model Object**:
    - Trained model with transformation parameters

    ## Model Selection Criteria:
    - **Scree Plot**: Look for "elbow" in eigenvalue decline
    - **Cumulative Variance**: Aim for 60-80% in most applications
    - **Interpretability**: Factors should have clear conceptual meaning

    ## Use Cases

    - **Data Reduction**: Compress high-dimensional datasets while preserving structure
    - **Pattern Discovery**: Identify underlying factors in survey and behavioral data
    - **Feature Engineering**: Create meaningful composite variables from raw measurements
    - **Exploratory Analysis**: Understand correlation patterns and variable clustering
    - **Scale Validation**: Test theoretical factor structures in psychometric research

    ## References

    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer
    - Fabrigar, L. R., & Wegener, D. T. (2012). *Exploratory Factor Analysis*. Oxford University Press
    - Bartholomew, D. J., et al. (2011). *Analysis of Multivariate Social Science Data* (2nd ed.). CRC Press
    """

    analysis_method = knext.EnumParameter(
        label="Analysis Method",
        description="Select the dimensionality reduction or factor analysis algorithm to use.",
        default_value=AnalysisMethod.STANDARD.name,
        enum=AnalysisMethod,
    )
    n_components = knext.IntParameter(
        label="Number of Components",
        description="Specify how many principal components to compute.",
        default_value=2,
        min_value=1,
        max_value=1000,
    )

    rotation_settings = RotationSettings()

    standardize_column = knext.BoolParameter(
        label="Standardize input data",
        description="Optionally standardize the input data before applying factor analysis.",
        default_value=True,
    )
    features_cols = knext.MultiColumnParameter(
        label="Numeric Input Columns",
        description="Select two or more numeric columns to include in the factor analysis.",
        column_filter=kutil.is_numeric,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        num_cols = len(self.features_cols)
        max_dims = min(self.n_components, num_cols)

        variance_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double(), knext.double(), knext.double()],
            ["Dimension", "Eigenvalue", "Explained Variance Ratio", "Cumulative Explained Variance", "Log-Likelihood"],
        )

        loadings_schema = knext.Schema(
            [knext.string(), knext.double(), knext.double()] + [knext.double()] * max_dims,
            ["Variable", "Communalities", "Noise Variance"] + [f"Loading (PC{i + 1})" for i in range(max_dims)],
        )
        # Define the binary model output port schema
        binary_model_schema = knext.BinaryPortObjectSpec("factor_analysis.model")

        return (
            variance_schema,
            loadings_schema,
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        # Import heavy dependencies only when needed
        import pickle
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from factor_analyzer.rotator import Rotator

        df = input_table.to_pandas()
        X = df[self.features_cols].dropna()

        if X.shape[0] < 2 or X.shape[1] < 2:
            raise ValueError("Factor analysis requires at least two rows and two numeric columns.")

        max_dims = min(self.n_components, X.shape[1])

        # Select analysis method
        method = self.analysis_method
        if method == AnalysisMethod.STANDARD.name:
            factor_model = PCA(n_components=max_dims)
            if self.standardize_column:
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(X)
            else:
                scaler = None
                x_transformed = X.values
        elif method == AnalysisMethod.INCREMENTAL.name:
            from sklearn.decomposition import IncrementalPCA

            factor_model = IncrementalPCA(n_components=max_dims)
            if self.standardize_column:
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(X)
            else:
                scaler = None
                x_transformed = X.values
        elif method == AnalysisMethod.FACTOR_ANALYSIS.name:
            from sklearn.decomposition import FactorAnalysis

            factor_model = FactorAnalysis(n_components=max_dims, svd_method="lapack")
            if self.standardize_column:
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(X)
            else:
                scaler = None
                x_transformed = X.values
        else:
            raise ValueError("Unknown analysis method selected.")

        # Fit the analysis model
        factor_model.fit(x_transformed)

        # Calculate loadings based on method
        if method == AnalysisMethod.FACTOR_ANALYSIS.name:
            # For Factor Analysis, use the components directly as loadings
            loadings = factor_model.components_.T
        else:
            # For PCA methods, scale loadings by sqrt of explained variance
            loadings = factor_model.components_.T * np.sqrt(factor_model.explained_variance_)

        phi = loadings[:, :max_dims]

        # Apply rotation if selected
        rotation_method = self.rotation_settings.rotation_method
        print(f"Applying rotation method: {rotation_method}")

        # Get the actual enum value (display name)
        rotation_value = RotationSettings.RotationMethods[rotation_method].value[0]
        print(f"Rotation value: {rotation_value}")

        if rotation_value == "None":
            rotated_loadings = phi
            rotation_matrix = np.eye(phi.shape[1])
        else:
            # Use factor_analyzer's Rotator - convert to lowercase for the library
            rotation_method_lower = rotation_value.lower()
            rotator = Rotator(method=rotation_method_lower)
            rotated_loadings = rotator.fit_transform(phi)
            rotation_matrix = rotator.rotation_

        # Flip sign of loadings if the sum over the same dimension is < 1
        # (for each component/dimension)
        for i in range(rotated_loadings.shape[1]):
            col_sum = np.sum(rotated_loadings[:, i])
            if col_sum < 1:
                rotated_loadings[:, i] *= -1
                # If rotation_matrix exists and is square, flip its sign for the same component
                if rotation_matrix.shape[0] == rotation_matrix.shape[1]:
                    rotation_matrix[:, i] *= -1

        # Define eigenvalues, explained variance ratio, and cumulative explained variance
        if method == AnalysisMethod.FACTOR_ANALYSIS.name:
            # For Exploratory Factor Analysis (EFA), leave eigenvalue table empty
            # Factor Analysis uses maximum likelihood estimation and eigenvalues are not directly meaningful
            eigenvalues = np.array([])
            var_ratio = np.array([])
            cum_var = np.array([])
        elif hasattr(factor_model, "explained_variance_"):
            # For PCA methods: use true eigenvalues from the covariance matrix decomposition
            eigenvalues = factor_model.explained_variance_[:max_dims]
            var_ratio = factor_model.explained_variance_ratio_[:max_dims]
            cum_var = np.cumsum(var_ratio)
        else:
            # Fallback: compute eigenvalues from transformed data variance
            eigenvalues = np.var(factor_model.transform(x_transformed), axis=0, ddof=1)[:max_dims]
            var_ratio = eigenvalues / np.sum(eigenvalues)
            cum_var = np.cumsum(var_ratio)

        # Create the variance DataFrame
        if len(eigenvalues) > 0:
            # For PCA methods: create dimension counter from 1 to max_dims
            dimension_col = np.arange(1, len(eigenvalues) + 1, dtype=np.float64)
            log_likelihood_col = np.zeros(len(eigenvalues), dtype=np.float64)  # Set to 0 for PCA
            variance_df = pd.DataFrame(
                {
                    "Dimension": dimension_col,
                    "Eigenvalue": eigenvalues.astype(np.float64),
                    "Explained Variance Ratio": var_ratio.astype(np.float64),
                    "Cumulative Explained Variance": cum_var.astype(np.float64),
                    "Log-Likelihood": log_likelihood_col,
                }
            )
        else:
            # For Factor Analysis: single row with selected dimensions and log-likelihood
            # Get log-likelihood if available, otherwise use 0
            if hasattr(factor_model, "loglike_"):
                # Handle both scalar and array cases
                loglike_raw = factor_model.loglike_
                if hasattr(loglike_raw, "__len__"):  # It's an array/list
                    log_likelihood_value = float(np.max(loglike_raw))
                else:  # It's a scalar
                    log_likelihood_value = float(loglike_raw)
            else:
                log_likelihood_value = 0.0

            variance_df = pd.DataFrame(
                {
                    "Dimension": np.array([max_dims], dtype=np.float64),  # Single value: selected number of components
                    "Eigenvalue": np.array([0.0], dtype=np.float64),  # Set to 0 for Factor Analysis
                    "Explained Variance Ratio": np.array([0.0], dtype=np.float64),  # Set to 0 for Factor Analysis
                    "Cumulative Explained Variance": np.array([0.0], dtype=np.float64),  # Set to 0 for Factor Analysis
                    "Log-Likelihood": np.array([log_likelihood_value], dtype=np.float64),  # Single scalar log-likelihood
                }
            )

        # Calculate communalities (sum of squared loadings for each variable)
        communalities = np.sum(rotated_loadings[:, :max_dims] ** 2, axis=1)

        # Calculate noise variance (unique variance not explained by factors)
        noise_variance = 1.0 - communalities

        # Prepare loadings table with communalities and noise variance
        loadings_data = pd.DataFrame(
            rotated_loadings[:, :max_dims], index=self.features_cols, columns=[f"Loading (PC{i + 1})" for i in range(max_dims)]
        )

        # Add communalities and noise variance columns
        loadings_data.insert(0, "Communalities", communalities)
        loadings_data.insert(1, "Noise Variance", noise_variance)

        # Reset index to make Variable a column
        loadings_df = loadings_data.reset_index().rename(columns={"index": "Variable"})

        # Save the trained factor analysis model to the binary output port for scoring/prediction
        # This model object contains all necessary information for transforming new data
        if self.standardize_column:
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_
        else:
            scaler_mean = None
            scaler_scale = None

        # Create comprehensive model dictionary for scoring node
        model_dict = {
            # Core model components
            "model": factor_model,  # The fitted sklearn model (PCA/IncrementalPCA/FactorAnalysis)
            "analysis_method": method,  # Method identifier for proper reconstruction
            "n_components": max_dims,  # Number of components/factors
            # Loadings and rotation information
            "loadings": rotated_loadings,  # Final rotated loadings matrix
            "rotation_matrix": rotation_matrix,  # Rotation transformation matrix
            "unrotated_components": phi,  # Original unrotated components
            "rotation_method": rotation_value,  # Rotation method used
            # Preprocessing information
            "scaler_mean": scaler_mean,  # Feature means (if standardized)
            "scaler_scale": scaler_scale,  # Feature scales (if standardized)
            "standardize_column": self.standardize_column,  # Whether standardization was applied
            "features_cols": self.features_cols,  # Original feature column names
        }

        model_binary = pickle.dumps(model_dict)

        return (
            knext.Table.from_pandas(variance_df),
            knext.Table.from_pandas(loadings_df),
            model_binary,
        )

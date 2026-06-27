import logging
import knime.extension as knext
from util import utils as kutil
from .multivariate_cat import multivariate_analysis_category

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Correspondence Analyzer",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/CorrespondenceAnalyzer.png",
    category=multivariate_analysis_category,
    keywords=[
        "Correspondence Analysis",
        "CA",
        "MCA",
        "Categorical Data",
        "Association Analysis",
        "Dimensionality Reduction",
    ],
    id="correspondence_analysis",
)
@knext.input_table(
    name="Input Data",
    description="Categorical data table for association analysis. Must contain at least two categorical (string) columns with minimum 2 unique values each. Automatically selects CA (2 variables) or MCA (3+ variables).",
)
@knext.output_table(
    name="Variance Explained",
    description="Eigenvalue decomposition results showing dimension importance: eigenvalues, explained variance ratios, and cumulative variance. Used to assess dimensionality reduction quality and determine optimal number of dimensions.",
)
@knext.output_table(
    name="Model Summary",
    description="Complete modality-level statistics including mass (frequency), point inertia (χ² contribution), coordinates (position), contributions (dimension importance), and cos² (representation quality). Essential for interpreting category relationships.",
)
@knext.output_image(
    name="Factor Map",
    description="2D biplot visualization displaying category positions in the first two factorial dimensions. Point size reflects category frequency, distances indicate similarity, and positioning reveals association patterns.",
)
class CorrespondenceAnalysisNode:
    """
    Reveals hidden associations and patterns in categorical data through geometric data analysis, automatically selecting between Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA) based on input complexity.

    ## Analysis Methods

    **Correspondence Analysis (CA)** - Applied to exactly two categorical variables:
    - Decomposes contingency tables using chi-square distance metrics
    - Reveals association patterns between categories of two variables
    - Optimal for exploring relationships in cross-tabulated data

    **Multiple Correspondence Analysis (MCA)** - Applied to three or more categorical variables:
    - Uses complete disjunctive coding and generalized singular value decomposition
    - Extends CA to multivariate categorical datasets
    - Applies Benzécri correction to adjust for eigenvalue inflation in MCA, improving interpretability of inertia and variance explained.

    ## Configuration Options

    - **Input Columns**: Two or more categorical variables (string type)
    - **Output Dimensions**: Number of principal axes to compute (1-100)
    - **Automatic Method Selection**: CA for 2 variables, MCA for 3+ variables
    - **Missing Value Handling**: Automatically treated as "missing" category

    ## Analysis Outputs

    1. **Variance Explained Table**: Eigenvalues, proportion and cumulative variance by dimension
    2. **Model Summary Table**: Complete modality statistics with coordinates and quality measures
    3. **Factor Map Visualization**: 2D biplot showing category relationships and groupings

    ## Model Summary Statistics

    - **Mass**: Marginal relative frequency in the dataset
    - **Point Inertia**: Absolute contribution to total inertia (χ² distance from independence)
    - **Contribution**: Relative contribution percentage to each dimension's variance
    - **Coordinate**: Principal coordinate position along each extracted dimension
    - **cos² (Quality)**: Representation quality - squared correlation between modality and dimension

    ## Factor Map Reading:

    - **Distance**: Closer categories are more similar/associated and tend to co-occur
    - **Origin Proximity**: Categories near origin are average/neutral, less distinctive

    ## Quality Assessment:

    - **High cos²** (>0.5): Category well-represented by dimension
    - **High Contribution** (>average): Category defines the dimension meaning
    - **Cumulative Variance**: Percentage of associations explained by retained dimensions

    ## Use Cases

    - **Market Research**: Customer segmentation, brand perception, purchase behavior analysis
    - **Social Sciences**: Survey analysis, demographic patterns, attitude research
    - **Healthcare**: Patient profiling, treatment outcomes, risk factor analysis
    - **Education**: Student performance patterns, curriculum effectiveness assessment
    - **Quality Control**: Defect pattern analysis, process improvement identification
    - **Text Mining**: Document classification, topic modeling, content analysis

    **Note:** Method automatically determined by input dimensionality. Requires minimum 2 categories per variable.
    """

    n_components = knext.IntParameter(
        label="Number of Output Dimensions",
        description="Principal dimensions to extract from the analysis. More dimensions capture additional variance but increase complexity. Typical choices: 2-3 for visualization, 5-10 for detailed analysis. Cannot exceed (min(categories)-1) due to mathematical rank constraints.",
        default_value=2,
        min_value=1,
        max_value=100,
    )

    features_cols = knext.MultiColumnParameter(
        label="Categorical Input Columns",
        description="Categorical variables for association analysis. Minimum 2 columns required, each with at least 2 unique values. Missing values automatically treated as 'missing' category. Method selection: CA (exactly 2 columns) or MCA (3+ columns).",
        column_filter=kutil.is_string,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        max_dims = self.n_components  # number of dimensions to compute defines the output schema

        variance_explained_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double()],
            ["Eigenvalue", "Explained Variance Ratio", "Cumulative Explained Variance"],
        )

        contrib_score_schema = knext.Schema(
            [knext.string(), knext.string(), knext.double(), knext.double()] + [knext.double()] * (3 * max_dims),
            ["Column", "Modality", "Mass", "Point Inertia"]
            + [f"Contribution (Dim {i + 1})" for i in range(max_dims)]
            + [f"Coordinate (Dim {i + 1})" for i in range(max_dims)]
            + [f"cos² (Dim {i + 1})" for i in range(max_dims)],
        )

        return (
            variance_explained_schema,
            contrib_score_schema,
            knext.ImagePortObjectSpec(knext.ImageFormat.SVG),
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        # Import heavy dependencies only when needed
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from io import BytesIO

        df = input_table.to_pandas()
        dimensions = self.features_cols
        dimension_df = df[dimensions].fillna("missing").astype(str)
        max_dims = self.n_components

        if df.size == 0:
            raise knext.InvalidParametersError(
                "Input table is empty. Please provide categorical data with at least 2 rows and 2 columns for correspondence analysis."
            )

        if len(dimensions) < 2:
            raise knext.InvalidParametersError(
                "Correspondence analysis requires at least 2 categorical columns. "
                "Please select additional categorical variables to analyze associations between categories."
            )

        if len(dimensions) < max_dims:
            raise knext.InvalidParametersError(
                f"Cannot extract {max_dims} dimensions from {len(dimensions)} variables. "
                f"Maximum extractable dimensions ≤ min(categories per variable) - 1. "
                f"Either reduce 'Number of Output Dimensions' or add more categorical columns."
            )

        # Check category counts and issue performance warnings
        total_categories = 0
        high_cardinality_cols = []

        for col in dimensions:
            unique_count = dimension_df[col].nunique()
            if unique_count < 2:
                raise knext.InvalidParametersError(
                    f"Column '{col}' has only {unique_count} unique value(s). "
                    f"Each categorical variable must have at least 2 different categories for meaningful analysis. "
                    f"Consider removing this column or combining rare categories."
                )

            total_categories += unique_count
            if unique_count > 50:
                high_cardinality_cols.append(f"{col} ({unique_count} categories)")

        # Performance warning for high-cardinality categorical data
        if total_categories > 200 or high_cardinality_cols:
            exec_context.set_warning(
                f"⚠️ PERFORMANCE WARNING: High categorical complexity detected.\n"
                f"Total categories across all variables: {total_categories}\n"
                f"High-cardinality columns: {', '.join(high_cardinality_cols) if high_cardinality_cols else 'None'}\n\n"
                f"This may cause:\n"
                f"• Slow computation and high memory usage\n"
                f"• Overcrowded visualizations difficult to interpret\n"
                f"• Numerical instability in eigenvalue decomposition\n\n"
                f"RECOMMENDED SOLUTIONS:\n"
                f"1. Group rare categories into 'Other' category (keep top 10-20 per variable)\n"
                f"2. Focus on most important categorical variables\n"
                f"3. Use hierarchical coding (e.g., Country→Region)\n"
                f"4. Consider reducing 'Number of Output Dimensions'\n"
                f"5. Filter data to most relevant category subsets"
            )
            LOGGER.warning(f"High cardinality warning: {total_categories} total categories, high-cardinality columns: {high_cardinality_cols}")

        # The CA (Correspondence analysis) computation follows the standard mathematical formulation as described in:
        # Greenacre, M. (2017). Correspondence analysis in practice (3rd ed.). Chapman and Hall/CRC.
        # The computational details are available on Wikipedia: https://en.wikipedia.org/wiki/Correspondence_analysis

        if len(dimensions) == 2:
            # Step 1: Compute contingency table
            contingency = pd.crosstab(dimension_df[dimensions[0]], dimension_df[dimensions[1]])
            observed = contingency.to_numpy()

            # === Step 2: Normalize counts to get correspondence matrix ===
            X = self._normalize_to_correspondence_matrix(pd.DataFrame(observed), pd, np)

            # === Step 3: Compute marginal distributions (masses) ===
            r, c = self._compute_masses(X, np)

            # === Step 4: Compute standardized residuals matrix ===
            S = self._standardized_residual_matrix(X, r, c, np)

            # Step 5 and 6 : Apply SVD and Filter small eigenvalues, slice the matrices and compute total inertia and explained ratio
            U, singular_vals, VT, eigenvals, all_eigenvals = self._compute_svd_filtered(S, np)

            if len(eigenvals) < max_dims:
                raise knext.InvalidParametersError(
                    f"Insufficient data complexity for {max_dims} dimensions. Only {len(eigenvals)} meaningful components available after filtering near-zero eigenvalues. "
                    f"This indicates low categorical associations or insufficient variability. "
                    f"Solutions: (1) Reduce 'Number of Output Dimensions' to {len(eigenvals)} or less, "
                    f"(2) Add more categorical columns with diverse categories, "
                    f"(3) Check for columns with mostly uniform distributions."
                )

            total_inertia = np.sum(all_eigenvals)
            explained_ratio = eigenvals / total_inertia

            # === Step 7 - 8 : Compute coordinates and contributions to dimensions and rows and columns masses ===
            (
                row_coords,
                col_coords,
                row_contrib,
                col_contrib,
            ) = self._compute_coordinates_and_contributions(U, VT, singular_vals, r, c, eigenvals, np)

            # === Step 9: Representation quality with respect to the dimensions via cosine ===
            cos2_row = (row_coords**2) / np.sum(row_coords**2, axis=1, keepdims=True)
            cos2_col = (col_coords**2) / np.sum(col_coords**2, axis=1, keepdims=True)

            row_labels = contingency.index.astype(str)
            col_labels = contingency.columns.astype(str)
            masses = np.concatenate([r, c])

            # === Step 10: Combine results ===
            scores_matrix = np.vstack([row_coords, col_coords])
            modality_labels = list(row_labels) + list(col_labels)
            contrib_matrix = np.vstack([row_contrib, col_contrib])
            cos_matrix = np.vstack([cos2_row, cos2_col])

        else:
            # MCA (Multiple Correspondence Analysis):
            # Reference: Abdi, H., & Valentin, D. (2007). *Multiple Correspondence Analysis*. In N. Salkind (Ed.),
            # https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
            # Wikipedia: https://en.wikipedia.org/wiki/Multiple_correspondence_analysis

            # Step 1: Create indicator matrix using pandas
            z_df = pd.get_dummies(dimension_df, columns=dimension_df.columns)  # DataFrame with .columns

            if z_df.shape[1] == 0:
                raise knext.InvalidParametersError(
                    "MCA encoding failed: No categorical features could be converted to indicator variables. "
                    "This typically occurs when all selected columns are empty or contain only missing values. "
                    "Please verify your categorical columns contain valid category data."
                )

            K = len(dimensions)  # number of original categorical variables

            # Step 2: Normalize to correspondence matrix
            Z = self._normalize_to_correspondence_matrix(z_df, pd, np)

            # Step 3: Row and column masses
            r, c = self._compute_masses(Z, np)

            # Step 4: Standardized residual matrix
            S = self._standardized_residual_matrix(Z, r, c, np)

            # Step 5 and 6 : Apply SVD and filter small eigenvalues, slice the matrices, apply correction, compute total inertia and explained ratio
            U, singular_vals, VT, eigenvals, all_eigenvals = self._compute_svd_filtered(S, np)

            if len(eigenvals) < self.n_components:
                raise Warning(
                    f"Only {len(eigenvals)} components could be computed after filtering the eigenvalues close to zero. "
                    f"Requested {max_dims}, but data rank is lower. Your data has low variance. Try adding another categorical column or using columns with more unique values."
                )

            # Benzécri correction improves interpretability by adjusting inflated eigenvalues in MCA.
            # PDF: https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf (see Chapter 5)
            """
            Benzécri correction adjusts inflated eigenvalues in MCA.

            If K = number of original categorical variables, and λ is an uncorrected eigenvalue:

            If λ > 1/K:
                λ_corrected = (K / (K - 1))² * (λ - 1/K)²
            Else:
                λ_corrected = 0

            This makes the interpretation of inertia more reliable in MCA.
            """
            if K > 1:
                corrected_eigenvals = np.array([(K / (K - 1) * (eig - 1 / K)) ** 2 if eig > 1 / K else 0 for eig in eigenvals])
                eigenvals = corrected_eigenvals

            total_inertia = np.sum(all_eigenvals)
            explained_ratio = eigenvals / total_inertia

            # Step 7: Coordinates, contributions, representation quality respective to dimensions

            (
                _,
                col_coords,
                _,
                col_contrib,
            ) = self._compute_coordinates_and_contributions(U, VT, singular_vals, r, c, eigenvals, np)

            # === Step 8: Representation quality with respect to the dimensions via cosine ===
            cos2_col = (col_coords**2) / np.sum(col_coords**2, axis=1, keepdims=True)
            col_cos2 = cos2_col  # MCA uses only column modalities

            # Step 9: Combine outputs
            modality_labels = list(z_df.columns.astype(str))
            scores_matrix = col_coords
            contrib_matrix = col_contrib
            cos_matrix = col_cos2
            # MCA uses only column categories
            masses = c

        # === Create  factor map ===
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(0, color="gray", lw=1)
        ax.axvline(0, color="gray", lw=1)
        ax.grid(True, linestyle="--", alpha=0.5)

        if scores_matrix.shape[1] < 2:
            raise ValueError("Not enough components to plot factor map. Need at least 2 dimensions.")

        x = scores_matrix[:, 0]
        y = scores_matrix[:, 1]

        # Compute base axis ranges
        x_min_raw, x_max_raw = np.min(x), np.max(x)
        y_min_raw, y_max_raw = np.min(y), np.max(y)

        x_range = x_max_raw - x_min_raw
        y_range = y_max_raw - y_min_raw

        # Ensure min range is at least 60% of the max range
        min_ratio = 0.6
        max_range = max(x_range, y_range)
        min_range = max_range * min_ratio

        if x_range < min_range:
            x_center = (x_max_raw + x_min_raw) / 2
            x_range = min_range
            x_min_raw = x_center - x_range / 2
            x_max_raw = x_center + x_range / 2

        if y_range < min_range:
            y_center = (y_max_raw + y_min_raw) / 2
            y_range = min_range
            y_min_raw = y_center - y_range / 2
            y_max_raw = y_center + y_range / 2

        # Add margin for padding
        x_margin = 0.25
        y_margin = 0.25
        xmin, xmax = x_min_raw - x_margin, x_max_raw + x_margin
        ymin, ymax = y_min_raw - y_margin, y_max_raw + y_margin

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # === Assign styles by column ===
        if len(dimensions) == 2:
            modality_columns = [dimensions[0]] * len(row_labels) + [dimensions[1]] * len(col_labels)
        else:
            modality_columns = [col.split("_")[0] for col in modality_labels]

        column_names = sorted(set(modality_columns))
        base_colors = plt.get_cmap("tab10").colors
        base_markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

        column_style_map = {
            col: {
                "color": base_colors[i % len(base_colors)],
                "marker": base_markers[i % len(base_markers)],
            }
            for i, col in enumerate(column_names)
        }

        # === Plot points and prepare legend handles ===
        legend_handles = {}
        for i, label in enumerate(modality_labels):
            col = modality_columns[i]
            style = column_style_map[col]

            ax.scatter(
                x[i],
                y[i],
                color=style["color"],
                marker=style["marker"],
                edgecolor="black",
                s=70,
                alpha=0.9,
                zorder=3,
            )

            if len(column_names) <= 4 and col not in legend_handles:
                legend_handles[col] = ax.scatter([], [], color=style["color"], marker=style["marker"], label=col)

        # === Smart label placement with adaptive offset after first cycle ===
        base_offset = 0.015 * max(np.ptp(x), np.ptp(y))
        max_cycles = 5  # 1st cycle fixed offset, next ones increase radius

        # Unit direction vectors
        directions = [
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (0, 1),
            (0, -1),
        ]

        label_positions = []

        # === Improved overlap threshold scaling ===
        range_scale = max(np.ptp(x), np.ptp(y))  # Max range across dimensions
        dynamic_threshold = 0.2 * range_scale  # Dynamically adjust based on plot extent

        def is_too_close(p1, p2, threshold=dynamic_threshold):
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) < threshold

        for i, label in enumerate(modality_labels):
            xi, yi = x[i], y[i]
            col = modality_columns[i]
            short_label = label.split("_")[-1] if "_" in label else label

            # Try increasing offset after first full cycle
            for cycle_i in range(max_cycles):
                scale = 1.0 if cycle_i == 0 else 1.1 * cycle_i  # Adaptive growth after first round
                for dx, dy in directions:
                    candidate_pos = (
                        xi + dx * base_offset * scale,
                        yi + dy * base_offset * scale,
                    )
                    if not any(is_too_close(candidate_pos, pos) for pos in label_positions):
                        break  # Found a good position
                else:
                    continue  # Try next cycle
                break  # Success: break outer loop
            else:
                LOGGER.warning(f"Label '{label}' placed at fallback due to crowding.")

            label_positions.append(candidate_pos)
            ax.text(
                candidate_pos[0],
                candidate_pos[1],
                short_label,
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
            )

        # === Axis labels and final layout ===
        dim1_variance = explained_ratio[0] * 100  # Convert to percentage
        dim2_variance = explained_ratio[1] * 100  # Convert to percentage
        ax.set_xlabel(f"Dimension 1 ({dim1_variance:.1f}%)", fontsize=12)
        ax.set_ylabel(f"Dimension 2 ({dim2_variance:.1f}%)", fontsize=12)
        ax.set_title("Correspondence Analysis – Factor Map", fontsize=14, weight="bold")
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(column_names) <= 4:
            ax.legend(
                handles=list(legend_handles.values()),
                loc="best",
                fontsize=9,
                title="Variable",
            )

        # Save as SVG to in-memory buffer
        buf = BytesIO()
        fig.savefig(buf, format="svg")
        buf.seek(0)

        # Variance explained output
        result_df = pd.DataFrame(
            {
                "Eigenvalue": eigenvals[:max_dims],
                "Explained Variance Ratio": explained_ratio[:max_dims],
                "Cumulative Explained Variance": explained_ratio[:max_dims].cumsum(),
            }
        )

        # Pad contributions matrix if needed
        if contrib_matrix.shape[1] < max_dims:
            pad_width = max_dims - contrib_matrix.shape[1]
            contrib_matrix = np.hstack([contrib_matrix, np.zeros((contrib_matrix.shape[0], pad_width))])

        # Pad scores matrix if needed
        if scores_matrix.shape[1] < max_dims:
            pad_width = max_dims - scores_matrix.shape[1]
            scores_matrix = np.hstack([scores_matrix, np.zeros((scores_matrix.shape[0], pad_width))])

        # Pad scores matrix if needed
        if cos_matrix.shape[1] < max_dims:
            pad_width = max_dims - cos_matrix.shape[1]
            cos_matrix = np.hstack([cos_matrix, np.zeros((cos_matrix.shape[0], pad_width))])

        # Compute Point Inertia: mass * squared Euclidean norm across components
        scores_used = scores_matrix[:, :max_dims]
        squared_norms = np.sum(scores_used**2, axis=1)
        point_inertia = masses * squared_norms

        # Now safely build DataFrames
        contrib_df = pd.DataFrame(
            contrib_matrix[:, :max_dims],
            columns=[f"Contribution (Dim {i + 1})" for i in range(max_dims)],
        )
        modality_labels_clean = [label.replace(f"{col}_", "") for label, col in zip(modality_labels, modality_columns)]

        contrib_df.insert(0, "Column", modality_columns)
        contrib_df.insert(1, "Modality", modality_labels_clean)

        score_df = pd.DataFrame(
            scores_matrix[:, :max_dims],
            columns=[f"Coordinate (Dim {i + 1})" for i in range(max_dims)],
        )
        cos2_df = pd.DataFrame(
            cos_matrix[:, :max_dims],
            columns=[f"cos² (Dim {i + 1})" for i in range(max_dims)],
        )

        # Combine DataFrames
        contrib_score_df = pd.concat([contrib_df.reset_index(drop=True), score_df, cos2_df], axis=1)

        # Insert Mass and Point Inertia right after Modality

        contrib_score_df.insert(2, "Mass", masses)
        contrib_score_df.insert(3, "Point Inertia", point_inertia)

        return (
            knext.Table.from_pandas(result_df),
            knext.Table.from_pandas(contrib_score_df),
            buf.getvalue(),
        )

    ## Shared computational methods for CA and MCA
    def _normalize_to_correspondence_matrix(self, table, pd, np):
        """
        Normalizes count data to correspondence matrix by converting frequencies to proportions.

        Creates the fundamental correspondence matrix P = N/n where N is the original
        contingency/indicator table and n is the grand total. This normalization ensures
        the analysis focuses on relative associations rather than absolute frequencies.
        """
        return table.to_numpy().astype(float) / table.to_numpy().sum()

    def _compute_masses(self, matrix, np):
        """
        Computes marginal masses (row and column totals) representing relative importance.

        Row masses r[i] and column masses c[j] indicate how much each category contributes
        to the overall dataset. These masses weight the chi-square distance calculations
        and determine category positioning in the factorial space.
        """
        row_masses = matrix.sum(axis=1)
        col_masses = matrix.sum(axis=0)
        return row_masses, col_masses

    def _standardized_residual_matrix(self, matrix, r, c, np):
        """
        This matrix shows how different the observed values are from what independence would suggest.

        We compute:
        S = (P - r * c^T), scaled by dividing rows and columns by sqrt of their masses.

        This highlights where observed values differ from expected ones.
        """
        return np.diag(1.0 / np.sqrt(r)) @ (matrix - np.outer(r, c)) @ np.diag(1.0 / np.sqrt(c))

    def _compute_svd_filtered(self, S, np, threshold: float = 1e-12):
        """
        Apply SVD to the standardized residual matrix and filter out near-zero eigenvalues.
        S = U * Σ * V^T
        Then:
        - Eigenvalues = squared singular values
        - These tell us how much variance each dimension explains
        We ignore very small values (close to 0) to avoid noise.
        Args:
            S: The standardized residual matrix.
            threshold: Minimum eigenvalue threshold for filtering.

        Returns:
            U: Left singular vectors (filtered)
            singular_vals: Filtered singular values
            VT: Right singular vectors (filtered)
            eigenvals: Filtered eigenvalues (squared singular values)
            all_eigenvals: All eigenvalues before filtering (used for total inertia)
        """
        U, singular_vals, VT = np.linalg.svd(S, full_matrices=False)
        all_eigenvals = singular_vals**2

        valid = all_eigenvals > threshold
        if np.sum(valid) == 0:
            raise ValueError("SVD failed: no eigenvalues passed the threshold.")

        return (
            U[:, valid],
            singular_vals[valid],
            VT[valid, :],
            all_eigenvals[valid],
            all_eigenvals,
        )

    def _compute_coordinates_and_contributions(self, U, VT, singular_vals, r, c, eigenvals, np):
        """
        Compute coordinates and contributions for rows and columns.
        - Coordinates:
        These show the position of each category along each principal axis.
        For rows:
            coord[i, k] = (1 / sqrt(r[i])) * U[i, k] * o[k]
        For columns:
            coord[j, k] = (1 / sqrt(c[j])) * V[j, k] * o[k]
        Where:
            - r[i] and c[j] are the row/column masses
            - U and V are the left/right singular vectors
            - o[k] is the k-th singular value

        - Contributions:
        These tell us how much each point affects each dimension (axis).
        For rows:
            contrib[i, k] = (coord[i, k]²) / λ[k]
        For columns:
            contrib[j, k] = (coord[j, k]²) / λ[k]
        Where λ[k] = o[k]² is the eigenvalue for dimension k

        This is used for both plotting and interpretation.
        Returns:
            row_coords, col_coords, row_contrib, col_contrib
        """
        row_coords = np.diag(1.0 / np.sqrt(r)) @ U * singular_vals
        col_coords = np.diag(1.0 / np.sqrt(c)) @ VT.T * singular_vals

        row_contrib = (row_coords**2) / eigenvals
        col_contrib = (col_coords**2) / eigenvals

        return row_coords, col_coords, row_contrib, col_contrib

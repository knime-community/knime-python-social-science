import knime.extension as knext

# NOTE: Categories currently don't work in modern UI.
# This is a bug that needs to be fixed in the KNIME Analytics Platform.
multivariate_analysis_category = knext.category(
    path="/community/socialscience",
    name="Multivariate Analysis",
    level_id="multivariate_analysis",
    description="Nodes for multivariate analysis",
    icon="icons/FactorAnalyzer.png",
)

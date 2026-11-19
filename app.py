import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="ProteinStruct-ML",
    page_icon="🧬",
    layout="wide"
)
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/casp_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load(
        "outputs/models/optimized_random_forest_compressed.joblib"
    )

st.sidebar.title("🧬 ProteinStruct-ML")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Exploratory Data Analysis",
        "Model Performance",
        "RMSD Prediction"
    ]
)
if page == "Project Overview":
    st.title("🧬 ProteinStruct-ML")
    st.subheader("Predicting Protein Structural Deviation Using Machine Learning")

    st.markdown("""
    ### Project Background

    Protein structure prediction is an important field in
    structural biology and bioinformatics.

    Root Mean Square Deviation (RMSD) measures structural
    differences between corresponding atomic coordinates
    after alignment.

    This project uses machine learning to predict RMSD
    from nine numerical features in the CASP dataset.
    """)

    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Cleaned Observations", "44,019")
    col2.metric("Input Features", "9")
    col3.metric("Target Variable", "RMSD (Å)")

    st.markdown("### Machine Learning Workflow")

    st.markdown("""
    1. Data cleaning and validation
    2. Exploratory data analysis
    3. Regression model development
    4. Random Forest optimization
    5. Evaluation on unseen test data
    6. Interactive RMSD prediction
    """)

    st.markdown("### Final Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Test MAE", "2.532 Å")
    col2.metric("Test RMSE", "3.660 Å")
    col3.metric("Test R²", "0.644")

    st.info(
        "Scientific limitation: The dataset does not contain protein "
        "identifiers, so generalization to entirely unseen proteins "
        "or protein families cannot be established."
    )

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    df = load_data()

    st.write("Explore the cleaned CASP protein structure dataset.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Dataset Dimensions")
    st.write(f"Observations: {df.shape[0]:,}")
    st.write(f"Variables: {df.shape[1]}")
    st.subheader("RMSD Distribution")

    bins = st.slider(
        "Select number of histogram bins",
        min_value=10,
        max_value=100,
        value=40,
        step=5
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["RMSD"],
        bins=bins,
        edgecolor="black",
        alpha=0.75
    )

    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Number of Observations")
    ax.set_title("Distribution of Protein Structural Deviation")

    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Feature Relationships with RMSD")

    selected_feature = st.selectbox(
        "Select a protein structure descriptor",
        [f"F{i}" for i in range(1, 10)]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df[selected_feature],
        df["RMSD"],
        alpha=0.15,
        s=8
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title(f"{selected_feature} vs RMSD")

    st.pyplot(fig)
    plt.close(fig)

    correlation = df[selected_feature].corr(df["RMSD"])

    st.write(
        f"Pearson correlation between {selected_feature} "
        f"and RMSD: **{correlation:.3f}**"
    )
elif page == "Model Performance":
    st.title("Machine Learning Model Performance")

    st.write(
        "This section compares the regression algorithms "
        "used to predict protein structural deviation (RMSD)."
    )

    st.subheader("Cross-Validation Results")

    model_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "HistGradientBoosting",
            "Optimized Random Forest"
        ],
        "CV MAE (Å)": [
            4.355,
            2.682,
            3.426,
            2.570
        ],
        "CV RMSE (Å)": [
            5.204,
            3.775,
            4.399,
            None
        ],
        "CV R²": [
            0.281,
            0.622,
            0.486,
            None
        ]
    })

    st.dataframe(
        model_results,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Model Comparison by MAE")

    st.bar_chart(
        model_results.set_index("Model")["CV MAE (Å)"]
    )

    st.caption(
        "Lower Mean Absolute Error (MAE) indicates "
        "better predictive performance."
    )
    st.subheader("Final Random Forest Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Test MAE", "2.532 Å")
    col2.metric("Test RMSE", "3.660 Å")
    col3.metric("Test R²", "0.644")

    st.markdown("""
    ### Scientific Interpretation

    The optimized Random Forest achieved an R² of 0.644
    on the held-out test dataset, explaining approximately
    64.4% of the observed variation in RMSD.

    The mean absolute prediction error was 2.532 Å.

    **Error analysis revealed systematic differences:**

    - Low RMSD (<5 Å): MAE = 1.795 Å
    - Intermediate RMSD (5–10 Å): MAE = 2.404 Å
    - High RMSD (≥10 Å): MAE = 3.527 Å

    The model tends to overestimate low RMSD values
    and underestimate high RMSD values.

    This suggests that predictions are biased toward
    intermediate RMSD values.
    """)

    st.warning(
        "The CASP dataset does not include protein identifiers. "
        "Although group-aware validation prevents identical feature "
        "combinations from crossing data splits, performance on "
        "entirely unseen proteins or protein families remains unknown."
    )
elif page == "RMSD Prediction":
    st.title("Interactive RMSD Prediction")

    st.write(
        "Enter the nine CASP protein structure descriptors "
        "to estimate RMSD using the optimized Random Forest model."
    )

    model = load_model()
    df = load_data()

    st.success("Optimized Random Forest model loaded successfully!")
    st.subheader("Enter Protein Structure Descriptors")

    st.write(
        "Enter values for the nine numerical descriptors "
        "used to train the model."
    )

    feature_columns = [f"F{i}" for i in range(1, 10)]

    input_values = {}

    st.subheader("Understanding the Input Features")

    feature_descriptions = {
        "F1": "Total surface area of the protein structure.",
        "F2": "Exposed non-polar surface area.",
        "F3": "Fractional area of exposed non-polar residues.",
        "F4": "Exposure measure of non-polar residue portions.",
        "F5": "Molecular mass-weighted exposed surface area.",
        "F6": "Average deviation from standard residue exposure.",
        "F7": "Euclidean distance-based structural descriptor.",
        "F8": "Secondary structure penalty score.",
        "F9": "Spatial distribution constraints (N, K value)."
    }

    with st.expander("What do F1–F9 mean?"):
        for feature, description in feature_descriptions.items():
            st.markdown(f"**{feature}:** {description}")

        st.caption(
            "Feature definitions are based on the UCI "
            "Physicochemical Properties of Protein Tertiary "
            "Structure dataset (CASP 5–9). Exact calculation "
            "formulas and units are not fully documented."
        )


    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        for i, feature in enumerate(feature_columns):
            with [col1, col2, col3][i % 3]:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(df[feature].median()),
                    format="%.4f",
                    help=feature_descriptions[feature]
                )

        predict_button = st.form_submit_button("Predict RMSD")

    if predict_button:
        input_df = pd.DataFrame([input_values], columns=feature_columns)

        prediction = model.predict(input_df)[0]

        st.metric("Predicted RMSD", f"{prediction:.3f} Å")

        st.subheader("Prediction Interpretation")

        if prediction < 5:
            st.success(
                "The predicted RMSD is in the lower range "
                "of structural deviation."
            )
        elif prediction < 10:
            st.info(
                "The predicted RMSD is in the intermediate "
                "range of structural deviation."
            )
        else:
            st.warning(
                "The predicted RMSD is in the higher range "
                "of structural deviation."
            )

        st.subheader("Input Validation")

        outside_range = []

        for feature in feature_columns:
            value = input_values[feature]

            minimum = df[feature].min()
            maximum = df[feature].max()

            if value < minimum or value > maximum:
                outside_range.append(feature)

        if outside_range:
            st.warning(
                "The following features are outside the "
                "observed dataset ranges: "
                + ", ".join(outside_range)
                + ". Predictions may be less reliable."
            )
        else:
            st.success(
                "All input values are within the observed "
                "individual feature ranges."
            )

        st.subheader("Model Reliability")

        col1, col2, col3 = st.columns(3)

        col1.metric("Test MAE", "2.532 Å")
        col2.metric("Test RMSE", "3.660 Å")
        col3.metric("Test R²", "0.644")

        st.caption(
            "These metrics describe performance across the "
            "held-out test dataset, not the uncertainty of "
            "this individual prediction."
        )

        st.info(
            "Scientific limitation: The predicted RMSD is a "
            "statistical estimate based on numerical descriptors. "
            "It does not replace structural alignment or "
            "experimental validation."
        )

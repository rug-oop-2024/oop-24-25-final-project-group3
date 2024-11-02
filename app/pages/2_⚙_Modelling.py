import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import (get_model, REGRESSION_MODELS,
                                  CLASSIFICATION_MODELS)
from autoop.core.ml.metric import (REGRESSION_METRICS, CLASSIFICATION_METRICS,
                                   get_metric)

# Streamlit page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")


# Helper function for styling text
def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning "
                  "pipeline to train a model on a dataset.")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Retrieve available datasets
datasets = automl.registry.list(type="dataset")

if datasets:
    # Dataset Selection
    st.header("1. Select a Dataset")
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Choose a dataset for training",
                                         dataset_names)

    # Retrieve and display selected dataset details
    selected_dataset = next(dataset for dataset in datasets
                            if dataset.name == selected_dataset_name)
    st.write(f"**Dataset Path:** {selected_dataset.asset_path}")
    st.write(f"**Version:** {selected_dataset.version}")
    st.write(f"**Metadata:** {selected_dataset.metadata}")
    st.write(f"**Tags:** {', '.join(selected_dataset.tags)}")

    # Convert bytes to DataFrame once and use for preview and feature detection
    try:
        dataset_data_bytes = selected_dataset.read()  # Returns bytes
        dataset_df = pd.read_csv(io.StringIO(dataset_data_bytes.decode()))

        # Display dataset preview
        st.write("### Dataset Preview")
        st.write(dataset_df.head())  # Display head of dataset DataFrame

    except Exception as e:
        st.error(f"Error reading dataset: {e}")

    st.header("2. Select Features")
    task_type = None  # Initialize to store task type

    try:
        features = detect_feature_types(dataset_df)
        feature_names = [feature.name for feature in features]

        # Feature selection for target w/ filtering in inputs to avoid overlap
        selected_target_feature = st.selectbox("Select target feature",
                                               feature_names)
        available_input_features = [name for name in feature_names if
                                    name != selected_target_feature]

        selected_input_features = st.multiselect(
            "Select input features", available_input_features
        )

        # Determine Task Type
        target_feature = next((f for f in features if
                               f.name == selected_target_feature), None)
        if target_feature:
            task_type = ("Classification" if
                         target_feature.type == "categorical"
                         else "Regression")
            st.write(f"**Detected Task Type:** {task_type}")

    except Exception as e:
        st.error(f"Error detecting features: {e}")

    # Model Selection
    st.header("3. Select a Model")
    model_choices = (REGRESSION_MODELS if task_type == "Regression"
                     else CLASSIFICATION_MODELS)
    selected_model_name = st.selectbox("Choose a model", model_choices)

    # Metric Selection
    st.header("4. Select Metrics")
    write_helper_text("Choose one or more metrics to evaluate the model's "
                      "performance.")
    available_metrics = (REGRESSION_METRICS if task_type == "Regression"
                         else CLASSIFICATION_METRICS)
    selected_metrics = st.multiselect("Choose metrics", available_metrics)

    # Display selected configuration
    st.write("### Pipeline Configuration Summary")
    st.write(f"**Selected Dataset:** {selected_dataset_name}")
    st.write(f"**Selected Model:** {selected_model_name}")
    st.write(f"**Selected Metrics:** {', '.join(selected_metrics)
                                      if selected_metrics else 'None'}")
    st.write(f"**Selected Input Features:** {', '.join(selected_input_features)
                                             if selected_input_features
                                             else 'None selected'}")
    st.write(f"**Selected Target Feature:** {selected_target_feature
                                             if selected_target_feature
                                             else 'None selected'}")

    # Initialize Pipeline
    if st.button("Initialize Pipeline"):
        # Instantiate the model and metrics
        model = get_model(selected_model_name)
        metric_objects = [get_metric(metric_name) for metric_name in
                          selected_metrics]

        st.success("Pipeline initialized successfully!")
        st.write("Model and metrics are ready for training.")

        # Display further training options or next steps
        write_helper_text("Next steps: configure additional parameters, "
                          "split dataset, and run training.")
else:
    st.write("No datasets available.")

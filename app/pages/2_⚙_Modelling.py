import streamlit as st
import pandas as pd
import pickle
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import (get_model, REGRESSION_MODELS,
                                  CLASSIFICATION_MODELS)
from autoop.core.ml.metric import (REGRESSION_METRICS, CLASSIFICATION_METRICS,
                                   get_metric)
from autoop.core.ml.artifact import Artifact

# Streamlit page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Retrieve available datasets
datasets = automl.registry.list(type="dataset")

# Helper function for styling text
def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("Design a machine learning pipeline to train a model on a dataset.")

if datasets:
    # Dataset Selection
    st.header("1. Select a Dataset")
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Choose a dataset for training", dataset_names)

    # Retrieve selected dataset
    selected_dataset = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)
    dataset_data_bytes = selected_dataset.read()
    dataset_df = pd.read_csv(io.StringIO(dataset_data_bytes.decode()))
    st.write("### Dataset Preview")
    st.write(dataset_df.head())

    # Feature selection and task type detection
    st.header("2. Select Features")
    features = detect_feature_types(dataset_df)
    feature_names = [feature.name for feature in features]
    selected_target_feature = st.selectbox("Select target feature", feature_names)
    available_input_features = [name for name in feature_names if name != selected_target_feature]
    selected_input_features = st.multiselect("Select input features", available_input_features)

    target_feature = next((f for f in features if f.name == selected_target_feature), None)
    task_type = "Classification" if target_feature and target_feature.type == "categorical" else "Regression"
    st.write(f"**Detected Task Type:** {task_type}")

    # Model and metric selection
    st.header("3. Select a Model")
    model_choices = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
    selected_model_name = st.selectbox("Choose a model", model_choices)
    selected_model = get_model(selected_model_name)

    st.header("4. Select Dataset Split")
    train_split = st.slider("Training Set Split (%)", min_value=50, max_value=90, value=80, step=5)
    test_split = 100 - train_split

    st.header("5. Select Metrics")
    available_metrics = REGRESSION_METRICS if task_type == "Regression" else CLASSIFICATION_METRICS
    selected_metrics = st.multiselect("Choose metrics", available_metrics)
    metric_objects = [get_metric(metric_name) for metric_name in selected_metrics]

    # Display Pipeline Summary
    st.header("6. Pipeline Summary")
    st.write("### Configuration Summary")
    st.write(f"**Selected Dataset:** {selected_dataset_name}")
    st.write(f"**Selected Target Feature:** {selected_target_feature}")
    st.write(f"**Selected Input Features:** {', '.join(selected_input_features)}")
    st.write(f"**Selected Model:** {selected_model_name}")
    st.write(f"**Training Split:** {train_split}% | **Testing Split:** {test_split}%")
    st.write(f"**Selected Metrics:** {', '.join(selected_metrics)}")

    # Training Section
    if st.button("Train Model"):
        # Split the dataset
        train_size = train_split / 100
        train_data = dataset_df.sample(frac=train_size, random_state=42)
        test_data = dataset_df.drop(train_data.index)

        # Prepare features and target for training and testing
        X_train = train_data[selected_input_features]
        y_train = train_data[selected_target_feature]
        X_test = test_data[selected_input_features]
        y_test = test_data[selected_target_feature]

        # Train the model
        selected_model.fit(X_train.values, y_train.values)
        st.success("Model training completed!")

        # Run predictions and evaluate metrics
        predictions = selected_model.predict(X_test.values)
        st.write("### Evaluation Results")
        for metric in metric_objects:
            result = metric(y_test.values, predictions)
            st.write(f"{metric}: {result}")

    # Pipeline saving section
    st.header("7. Save Pipeline")
    pipeline_name = st.text_input("Enter a name for the pipeline")
    pipeline_version = st.text_input("Enter pipeline version", value="1.0.0")

    # Ensure required fields are selected
    is_ready = all([selected_input_features, selected_target_feature, selected_metrics, pipeline_name])
    if not is_ready:
        st.warning("Please complete all selections before saving the pipeline.")

    # Save Pipeline button
    if st.button("Save Pipeline", disabled=not is_ready):
        # Prepare pipeline data for serialization
        pipeline_data = {
            "model": pickle.dumps(selected_model),  # Serialize model
            "metrics": selected_metrics,
            "input_features": selected_input_features,
            "target_feature": selected_target_feature,
            "train_split": train_split / 100,
        }

        # Create artifact for pipeline
        pipeline_artifact = Artifact(
            name=pipeline_name,
            asset_path=f"pipelines/{pipeline_name}.pkl",
            data=pickle.dumps(pipeline_data),
            version=pipeline_version,
            type="pipeline",
            metadata={"task_type": task_type},
            tags=["pipeline", task_type],
        )

        # Register the pipeline in the artifact registry
        automl.registry.register(pipeline_artifact)
        st.success(f"Pipeline '{pipeline_name}' (v{pipeline_version}) saved successfully!")
else:
    st.write("No datasets available.")

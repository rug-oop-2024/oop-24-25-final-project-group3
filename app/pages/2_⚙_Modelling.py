import streamlit as st
import pandas as pd
from io import BytesIO

from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.classification import LogisticRegression
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.database import Database

st.set_page_config(page_title="Modelling", page_icon="📈")

# Initialize the AutoML system and database
automl = AutoMLSystem.get_instance()
database = Database(storage=automl.storage)  # Ensure the AutoML system has a storage instance

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Fetch available datasets from the database
datasets = database.list("datasets")

# ---- STEP 1: Select a Dataset ----
st.header("Step 1: Select a Dataset")
if datasets:
    dataset_names = [name for name, _ in datasets]
    selected_dataset_name = st.selectbox("Choose a dataset to train on", options=dataset_names)
    
    # Load the selected dataset from the database
    if selected_dataset_name:
        selected_dataset = database.get("datasets", selected_dataset_name)
        df = pd.read_csv(BytesIO(selected_dataset['data']))  # Load from the 'data' field in binary format
        
        st.write("Preview of the selected dataset:")
        st.write(df.head())

        # Detect features
        features = detect_feature_types(df)  # Update to pass the DataFrame directly

        # ---- STEP 2: Select Features ----
        st.header("Step 2: Select Features")
        input_features = st.multiselect("Select Input Features", options=[f.name for f in features])
        target_feature_name = st.selectbox("Select Target Feature", options=[f.name for f in features if f.name not in input_features])

        if input_features and target_feature_name:
            input_features_objs = [f for f in features if f.name in input_features]
            target_feature = next(f for f in features if f.name == target_feature_name)

            # ---- STEP 3: Configure Model and Metrics ----
            st.header("Step 3: Configure Model and Metrics")
            model_type = st.selectbox("Choose a Model", ["MultipleLinearRegression", "LogisticRegression"])
            metric_name = st.selectbox("Choose an Evaluation Metric", ["mean_squared_error", "accuracy", "precision", "recall", "f1_score"])

            # Initialize the model based on the selection
            if model_type == "MultipleLinearRegression":
                model = MultipleLinearRegression()
            elif model_type == "LogisticRegression":
                model = LogisticRegression()

            metric = get_metric(metric_name)

            # ---- STEP 4: Train and Save the Model ----
            if st.button("Train Model"):
                pipeline = Pipeline(
                    dataset=df,
                    model=model,
                    input_features=input_features_objs,
                    target_feature=target_feature,
                    metrics=[metric],
                    split=0.8
                )
                results = pipeline.execute()
                
                st.header("Training Results")
                st.write("Metrics:")
                for metric_obj, score in results["metrics"]:
                    st.write(f"{metric_obj.__class__.__name__}: {score}")

                # Save trained pipeline as an artifact
                save_name = st.text_input("Enter a name to save this pipeline:")
                if st.button("Save Pipeline") and save_name:
                    for artifact in pipeline.artifacts:
                        artifact.name = save_name
                        database.set("pipelines", save_name, artifact.dict())  # Save pipeline in the database
                    st.success(f"Pipeline '{save_name}' saved successfully!")

    # ---- SECTION: Load and Predict with Saved Pipelines ----
    st.header("Load and Predict with Saved Pipelines")
    saved_pipelines = database.list("pipelines")
    if saved_pipelines:
        pipeline_names = [name for name, _ in saved_pipelines]
        selected_pipeline_name = st.selectbox("Select a pipeline for prediction", options=pipeline_names)
        selected_pipeline = database.get("pipelines", selected_pipeline_name)

        if selected_pipeline:
            # Display pipeline summary
            st.write(f"**Pipeline Summary for '{selected_pipeline_name}':**")
            st.write(selected_pipeline['metadata'])

            # Upload data for prediction
            st.write("Upload data for prediction:")
            pred_file = st.file_uploader("Choose a CSV file for prediction", type="csv")
            if pred_file and st.button("Run Prediction"):
                pred_df = pd.read_csv(pred_file)
                predictions = pipeline.model.predict(pred_df[input_features])
                st.write("Predictions:")
                st.write(predictions)
else:
    st.write("No datasets available. Please add a dataset first.")

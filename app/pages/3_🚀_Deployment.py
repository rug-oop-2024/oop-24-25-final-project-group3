import streamlit as st
from app.core.system import AutoMLSystem
from autoop.functional.pipeline_graphing import (
    create_pipeline_model, generate_training_prediction_plot)
from autoop.functional.report_generator import generate_pdf_report
import pandas as pd
import pickle
import io
import os
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment")
st.write("View and manage saved pipelines here.")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Initialize pipeline list refresh trigger in session state
if 'refresh_pipelines' not in st.session_state:
    st.session_state.refresh_pipelines = False

# Retrieve saved pipelines
pipelines = automl.registry.list(type="pipeline")

if pipelines:
    # Select pipeline to load
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_pipeline_name = st.selectbox("Select a pipeline", pipeline_names)
    selected_pipeline = next(p for p in pipelines if
                             p.name == selected_pipeline_name)
    training_plot_path = (f"./assets/plots/{selected_pipeline.name}_training_"
                          "loss_plot.png")

    # Load pipeline data
    pipeline_data = pickle.loads(selected_pipeline.data)
    model = pickle.loads(pipeline_data["model"])

    # Retrieve the dataset used during training
    dataset_name = pipeline_data["dataset_name"]
    target_feature = pipeline_data["target_feature"]
    input_features = pipeline_data["input_features"]

    # Load the original dataset
    selected_dataset = next(d for d in automl.registry.list(type="dataset")
                            if d.name == dataset_name)
    dataset_data_bytes = selected_dataset.read()
    original_data = pd.read_csv(io.StringIO(dataset_data_bytes.decode()))

    # Get training values from the original dataset
    training_values = original_data[target_feature].values

    # Display pipeline info
    st.write(f"**Pipeline Name:** {selected_pipeline.name}")
    st.write(f"**Version:** {selected_pipeline.version}")
    st.write(f"**Tags:** {', '.join(selected_pipeline.tags)}")

    # Delete Dataset Button
    if st.button("Delete Pipeline"):
        try:
            # Ensure dataset ID is valid before deletion
            if selected_pipeline.id:
                automl.registry.delete(selected_pipeline.id)
                os.remove(training_plot_path)
                st.success(f"Pipeline '{selected_pipeline.name}' deleted "
                           "successfully!")
                # Trigger pipeline list refresh
                st.session_state.refresh_pipeline = True
            else:
                st.error("Pipeline ID is missing. Unable to delete.")
        except Exception as e:
            st.error(f"Failed to delete pipeline: {e}")

    # Load pipeline data and display model details
    pipeline_data = pickle.loads(selected_pipeline.data)
    st.write("### Pipeline Details")

    # Deserialize model
    model = pickle.loads(pipeline_data["model"])
    is_trained = getattr(model, "trained", False)
    st.write("**Model Type:**", type(model).__name__)
    st.write("**Training Status:**", "Trained" if is_trained else
             "Not Trained")
    st.write("**Original Dataset:**", dataset_name)
    st.write("**Metrics:**")
    for metric_name, metric_value in pipeline_data["metrics"].items():
        st.write(f"- {metric_name}: {metric_value}")
    st.write("**Input Features:**", ", ".join(pipeline_data["input_features"]))
    st.write("**Target Feature:**", pipeline_data["target_feature"])
    st.write("**Training Split:**", f"{pipeline_data['train_split'] * 100}%")

    # Prediction section
    st.header("Run Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file for predictions",
                                     type="csv")

    if uploaded_file:
        # Prepare data for predictions
        input_data = pd.read_csv(uploaded_file)
        input_features = pipeline_data["input_features"]
        required_features = [f for f in input_features if f in
                             input_data.columns]

        if len(required_features) < len(input_features):
            missing_features = set(input_features) - set(input_data.columns)
            st.error(f"Uploaded data is missing required features: {', '.join(
                missing_features)}")
        else:
            # Select the input columns needed for prediction
            input_features_data = input_data[required_features]

            # Generate predictions
            predictions = model.predict(input_features_data.values)

            # Assuming `predictions` is a 2D array, get the column with the
            # highest value (predicted class)
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                predictions = np.argmax(predictions, axis=1)
            input_data["Predictions"] = predictions

            # Load the label mapping from the pipeline data
            label_mapping = pipeline_data.get("label_mapping")

            if label_mapping:
                # Reverse the mapping for prediction replacement
                reverse_mapping = {v: k for k, v in label_mapping.items()}

                # Replace numeric predictions with original labels
                input_data["Predictions"] = input_data["Predictions"].map(
                    reverse_mapping)

            results_df = input_features_data.copy()
            results_df[pipeline_data["target_feature"] + "_predictions"
                       ] = input_data["Predictions"]
            st.write("### Prediction Results with Features")
            st.write(results_df)

            # Create and plot the Keras model to visualize the pipeline flow
            pipeline_model_plot_path = create_pipeline_model(input_features)

            prediction_plot_path = generate_training_prediction_plot(
                training_values, input_data["Predictions"])

            # Provide download link for predictions as CSV
            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv,
                               file_name="predictions.csv", mime="text/csv")

            with st.spinner("Generating report, please wait..."):
                # Call the function to generate the PDF report
                pdf_output = generate_pdf_report(
                    selected_pipeline=selected_pipeline,
                    model=model,
                    pipeline_data=pipeline_data,
                    training_plot_path=training_plot_path,
                    pipeline_model_plot_path=pipeline_model_plot_path,
                    prediction_plot_path=prediction_plot_path,
                    dataset_name=dataset_name
                )

                # Download PDF
                st.download_button("Download PDF Report", data=pdf_output,
                                   file_name="model_report.pdf",
                                   mime="application/pdf")

else:
    st.write("No saved pipelines available.")

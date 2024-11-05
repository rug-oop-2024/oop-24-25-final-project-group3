import streamlit as st
from app.core.system import AutoMLSystem
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os

# Streamlit page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment")
st.write("View and manage saved pipelines here.")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Initialize pipeline list refresh trigger in session state
if 'refresh_pipelines' not in st.session_state:
    st.session_state.refresh_pipelines = False

# Retrieve available pipelines
pipelines = automl.registry.list(type="pipeline")

if pipelines:
    # Display each pipeline
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_pipeline_name = st.selectbox("Select a pipeline to view",
                                          pipeline_names)

    # Find and display selected pipeline details
    selected_pipeline = next(pipeline for pipeline in pipelines if
                             pipeline.name == selected_pipeline_name)

    st.write(f"**Pipeline Name:** {selected_pipeline.name}")
    st.write(f"**Version:** {selected_pipeline.version}")
    st.write(f"**Tags:** {', '.join(selected_pipeline.tags)}")

    # Delete Dataset Button
    if st.button("Delete Pipeline"):
        try:
            # Ensure dataset ID is valid before deletion
            if selected_pipeline.id:
                automl.registry.delete(selected_pipeline.id)
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

    # Display Model Attributes
    if hasattr(model, "parameters") and model.parameters:
        st.write("**Model Parameters:**", model.parameters)
    else:
        st.write("**Model Attributes:**")
        model_attributes = {k: v for k, v in model.__dict__.items() if not
                            k.startswith('_')}
        for key, value in model_attributes.items():
            st.write(f"{key}: {value}")

    # Display other pipeline configurations
    st.write("**Metrics:**", ", ".join(pipeline_data["metrics"]))
    st.write("**Input Features:**", ", ".join(pipeline_data["input_features"]))
    st.write("**Target Feature:**", pipeline_data["target_feature"])
    st.write("**Training Split:**", f"{pipeline_data['train_split'] * 100}%")

    # Prediction Section
    st.header("Run Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file for predictions",
                                     type="csv")

    if uploaded_file:
        # Read CSV data
        input_data = pd.read_csv(uploaded_file)

        # Ensure input data has the correct features
        required_features = pipeline_data["input_features"]
        missing_features = [f for f in required_features if f not in
                            input_data.columns]
        if missing_features:
            st.error(f"Uploaded data is missing required features: {', '.join(
                missing_features)}")
        else:
            # Select the input columns needed for prediction
            input_features_data = input_data[required_features]

            # Run predictions
            predictions = model.predict(input_features_data.values)
            input_data["Predictions"] = predictions
            st.write("### Prediction Results")

            # Combine predictions with input data for download
            prediction_df = input_data.copy()
            prediction_df["Prediction"] = predictions
            st.write(prediction_df)

            # Plotting the predictions
            st.write("### Prediction Plot")
            fig, ax = plt.subplots()
            ax.plot(prediction_df.index, prediction_df["Prediction"],
                    label="Predictions")
            ax.set_xlabel("Index")
            ax.set_ylabel("Prediction Value")
            ax.legend()
            st.pyplot(fig)

            # Provide download link for predictions as CSV
            csv = prediction_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Save the plot as a temporary image file
            temp_image_path = "/tmp/predictions_plot.png"
            fig.savefig(temp_image_path)

            # Generate PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, txt="Model Prediction Report", ln=True,
                     align="C")

            # Add pipeline details
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Pipeline: {selected_pipeline.name}",
                     ln=True)
            pdf.cell(200, 10, txt=f"Version: {selected_pipeline.version}",
                     ln=True)

            # Add metrics if available
            if "metrics" in pipeline_data:
                pdf.cell(200, 10, txt="Metrics:", ln=True)
                for metric in pipeline_data["metrics"]:
                    pdf.cell(200, 10, txt=f"{metric}", ln=True)

            # Embed plot
            pdf.cell(200, 10, txt="Prediction plot:", ln=True)
            pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=180)

            # Output PDF to BytesIO
            pdf_output = io.BytesIO()
            # Get the PDF content as bytes
            pdf_content = pdf.output(dest='S').encode('latin1')
            pdf_output.write(pdf_content)
            pdf_output.seek(0)

            # Remove the temporary image file
            os.remove(temp_image_path)

            # Download PDF
            st.download_button(
                label="Download PDF Report",
                data=pdf_output,
                file_name="model_report.pdf",
                mime="application/pdf"
            )
else:
    st.write("No saved pipelines available.")

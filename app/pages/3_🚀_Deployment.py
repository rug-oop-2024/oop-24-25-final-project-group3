import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from app.core.system import AutoMLSystem

# Streamlit page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment")
st.write("View and manage saved pipelines here.")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Retrieve available pipelines
pipelines = automl.registry.list(type="pipeline")

if pipelines:
    # Display each pipeline
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_pipeline_name = st.selectbox("Select a pipeline to view", pipeline_names)

    # Find and display selected pipeline details
    selected_pipeline = next(pipeline for pipeline in pipelines if pipeline.name == selected_pipeline_name)

    st.write(f"**Pipeline Name:** {selected_pipeline.name}")
    st.write(f"**Version:** {selected_pipeline.version}")
    st.write(f"**Metadata:** {selected_pipeline.metadata}")
    st.write(f"**Tags:** {', '.join(selected_pipeline.tags)}")

    # Delete Pipeline Button
    if st.button("Delete Pipeline"):
        try:
            if selected_pipeline.id:
                automl.registry.delete(selected_pipeline.id)
                st.success(f"Pipeline '{selected_pipeline.name}' deleted successfully!")
                st.session_state.refresh_pipelines = True
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
    st.write("**Training Status:**", "Trained" if is_trained else "Not Trained")

    # Display other pipeline configurations
    st.write("**Metrics:**", ", ".join(pipeline_data["metrics"]))
    st.write("**Input Features:**", ", ".join(pipeline_data["input_features"]))
    st.write("**Target Feature:**", pipeline_data["target_feature"])
    st.write("**Training Split:**", f"{pipeline_data['train_split'] * 100}%")

    # Function to safely extract metrics with placeholder values if missing
    def extract_metrics_with_names_only(metrics_data):
        """Create a dictionary with metrics as names and 'N/A' as placeholder values if values are missing."""
        if isinstance(metrics_data, list):
            # If only metric names are provided
            if all(isinstance(m, str) for m in metrics_data):
                return {m: 'N/A' for m in metrics_data}
        return {}

    # Extract train and test metrics with names
    train_metrics = extract_metrics_with_names_only(pipeline_data.get("train_metrics", []))
    test_metrics = extract_metrics_with_names_only(pipeline_data.get("metrics", []))

    # Debug output to confirm data extraction
    print("Raw train metrics data:", pipeline_data.get("train_metrics", []))
    print("Raw test metrics data:", pipeline_data.get("metrics", []))
    print("Extracted train metrics:", train_metrics)
    print("Extracted test metrics:", test_metrics)

    # Display the metrics in Streamlit
    st.write("### Training Metrics")
    st.write(train_metrics if train_metrics else "No training metrics available.")
    st.write("### Test Metrics")
    st.write(test_metrics if test_metrics else "No test metrics available.")

    # Plotting section for metrics
    if train_metrics and test_metrics:
        st.write("### Metrics Plot")
        fig, ax = plt.subplots()
        ax.plot(train_metrics.keys(), train_metrics.values(), label="Training Metrics", marker='o')
        ax.plot(test_metrics.keys(), test_metrics.values(), label="Test Metrics", marker='o')
        ax.set_title("Training and Test Metrics")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Values")
        ax.legend()
        st.pyplot(fig)  # Display the plot in Streamlit

    # Experiment Report Section
    st.header("Experiment Report")
    
    # Function to generate experiment report PDF with plot
    def generate_pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add pipeline details to the PDF
        pdf.cell(200, 10, txt="Experiment Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Pipeline Name: {selected_pipeline.name}", ln=True)
        pdf.cell(200, 10, txt=f"Version: {selected_pipeline.version}", ln=True)
        pdf.cell(200, 10, txt=f"Model Type: {type(model).__name__}", ln=True)
        pdf.cell(200, 10, txt=f"Training Status: {'Trained' if is_trained else 'Not Trained'}", ln=True)
        pdf.cell(200, 10, txt=f"Metrics: {', '.join(pipeline_data['metrics'])}", ln=True)
        pdf.cell(200, 10, txt=f"Input Features: {', '.join(pipeline_data['input_features'])}", ln=True)
        pdf.cell(200, 10, txt=f"Target Feature: {pipeline_data['target_feature']}", ln=True)
        pdf.cell(200, 10, txt=f"Training Split: {pipeline_data['train_split'] * 100}%", ln=True)

        # Add the metrics plot to the PDF
        if train_metrics and test_metrics:
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            pdf.image(img_buffer, x=10, y=pdf.get_y() + 10, w=180)

        # Save PDF to buffer for download
        pdf_buffer = BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin1')
        pdf_buffer.write(pdf_output)
        pdf_buffer.seek(0)
        return pdf_buffer

    # Generate PDF report
    pdf_buffer = generate_pdf_report()

    # Add download button for the report
    st.download_button(
        label="Download Experiment Report as PDF",
        data=pdf_buffer,
        file_name="experiment_report.pdf",
        mime="application/pdf"
    )

    # Prediction Section
    st.header("Run Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file for predictions", type="csv")

    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        required_features = pipeline_data["input_features"]
        missing_features = [f for f in required_features if f not in input_data.columns]
        if missing_features:
            st.error(f"Uploaded data is missing required features: {', '.join(missing_features)}")
        else:
            input_features_data = input_data[required_features]
            predictions = model.predict(input_features_data.values)
            results_df = input_features_data.copy()
            results_df[pipeline_data["target_feature"] + "_predictions"] = predictions
            st.write("### Prediction Results with Features")
            st.write(results_df)

            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv_data,
                file_name="predictions_with_features.csv",
                mime="text/csv"
            )
else:
    st.write("No saved pipelines available.")

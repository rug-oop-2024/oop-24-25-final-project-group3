from fpdf import FPDF
from typing import Any, Dict
import os
import io
import pydoc  # noqa: F401


def generate_pdf_report(selected_pipeline: Any, model: Any,
                        pipeline_data: Dict[str, Any], training_plot_path: str,
                        pipeline_model_plot_path: str,
                        prediction_plot_path: str, dataset_name: str
                        ) -> io.BytesIO:
    """
    Generates a PDF report containing model details, plots, and metrics.

    Args:
        selected_pipeline (object): The pipeline object containing metadata.
        model (object): The trained model object.
        pipeline_data (dict): Pipeline data containing metrics and features.
        training_plot_path (str): Path to the training plot image.
        pipeline_model_plot_path (str): Path to the pipeline model plot image.
        prediction_plot_path (str): Path to the prediction plot image.
        dataset_name (str): Name of the dataset used during training.

    Returns:
        io.BytesIO: A buffer containing the PDF report.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.add_font("ComicSans", "", "./assets/fonts/ComicSans.ttf", uni=True)
    pdf.set_font("ComicSans", "", 16)
    pdf.cell(200, 10, txt="Model Prediction Report", ln=True, align="C")

    # Add pipeline details
    pdf.set_font("ComicSans", size=11)
    pdf.cell(200, 10, txt=f"Pipeline: {selected_pipeline.name}", ln=True)
    pdf.cell(200, 10, txt=f"Version: {selected_pipeline.version}", ln=True)
    pdf.cell(200, 10, txt=f"Tags: {', '.join(selected_pipeline.tags)}",
             ln=True)
    pdf.cell(200, 10, txt=f"Model type: {type(model).__name__}", ln=True)
    pdf.cell(200, 10, txt=f"Original dataset: {dataset_name}", ln=True)

    # Add metrics if available
    pdf.cell(200, 10, txt="Metrics:", ln=True)
    for metric_name, metric_value in pipeline_data["metrics"].items():
        pdf.cell(200, 10, txt=f"- {metric_name}: {metric_value}", ln=True)

    pdf.cell(200, 10, txt="Input Features: "
             f"{', '.join(pipeline_data['input_features'])}", ln=True)
    pdf.cell(200, 10, txt=f"Target Feature: {pipeline_data['target_feature']}",
             ln=True)
    pdf.cell(200, 10, txt=f"Training Split: {pipeline_data['train_split'] *
                                             100}%", ln=True)

    # Embed training plot in PDF
    pdf.cell(200, 10, txt="Training Plot:", ln=True)
    pdf.image(training_plot_path, x=10, y=pdf.get_y(), w=180)

    pdf.add_page()

    # Embed pipeline flow plot in PDF
    pdf.cell(200, 10, txt="Pipeline Flow:", ln=True)
    pdf.image(pipeline_model_plot_path, x=10, y=pdf.get_y(), w=180)

    # Adjust the y-coordinate for the next plot to prevent overlap
    pdf.set_y(pdf.get_y() + 100)  # Increase 100 as needed to create space

    # Embed prediction plot in PDF
    pdf.cell(200, 10, txt="Prediction Plot:", ln=True)
    pdf.image(prediction_plot_path, x=10, y=pdf.get_y(), w=180)

    # Output PDF to BytesIO
    pdf_output = io.BytesIO()
    pdf_content = pdf.output(dest="S").encode("latin1")
    pdf_output.write(pdf_content)
    pdf_output.seek(0)

    # Clean up temp image files
    os.remove(pipeline_model_plot_path)
    os.remove(prediction_plot_path)

    return pdf_output

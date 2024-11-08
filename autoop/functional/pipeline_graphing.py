import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
from typing import Dict, List, Tuple
import pydoc  # noqa: F401


def visualise_pipeline(input_features: List[str], X_train: np.ndarray,
                       y_train: np.ndarray, epochs: int = 50,
                       batch_size: int = 32) -> Tuple[Model, Dict[
                           str, List[float]]]:
    """
    This module provides functions to visualize and create Keras-based
    pipelines and training plots.

    Functions:
    - visualise_pipeline: Trains a Keras model and plots training loss over
      time.
    - create_pipeline_model: Creates and visualizes a Keras model to show data
      flow through the pipeline.
    - generate_training_prediction_plot: Generates a plot comparing training
    values and predictions, saving the plot to a file.
    """

    inputs = []
    input_dict = {}

    # Create input layers for each feature
    for i, feature in enumerate(input_features):
        inp = Input(shape=(1,), name=f"input_{feature}")
        inputs.append(inp)
        # Populate input_dict for training
        input_dict[f"input_{feature}"] = X_train[:, i].reshape(-1, 1)

    # If multiple features, concatenate them
    if len(inputs) > 1:
        concatenated = concatenate(inputs)
    else:
        concatenated = inputs[0]  # Single input case

    # Example processing layer
    processed = Dense(4, activation='relu', name="processing_layer")(
        concatenated)
    output = Dense(1, activation='linear', name="output_layer")(processed)

    model = Model(inputs=inputs, outputs=output, name="pipeline_model")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')

    # Train the model and capture the training history
    history = model.fit(input_dict, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=1)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)

    return model, history


def create_pipeline_model(input_features: List[str], hidden_units: int = 4
                          ) -> str:
    """
    Creates a Keras model to visualize the data flow through the pipeline.

    Args:
        input_features (list): A list of input feature names.
        hidden_units (int): Number of units in the hidden layer (default: 4).

    Returns:
        BytesIO: A buffer containing the plot of the model as an image.
    """
    inputs = []

    # Create input layers for each feature
    for feature in input_features:
        inp = Input(shape=(1,), name=f"input_{feature}")
        inputs.append(inp)

    # Concatenate inputs if more than one feature is present
    if len(inputs) > 1:
        concatenated = concatenate(inputs)
    else:
        concatenated = inputs[0]  # Single input case

    # Add a dense processing layer and output layer
    processed = Dense(hidden_units, activation='relu', name="processing_layer"
                      )(concatenated)
    output = Dense(1, activation='linear', name="output_layer")(processed)

    # Create the model
    model = Model(inputs=inputs, outputs=output, name="pipeline_model")

    # Save the model plot to a file
    plot_path = "/tmp/pipeline_model_plot.png"
    plot_model(model, to_file=plot_path, show_shapes=True,
               show_layer_names=True, dpi=150)

    # Return the path of the generated plot image
    return plot_path


def generate_training_prediction_plot(training_values: np.ndarray,
                                      predictions: np.ndarray) -> str:
    """
    Generate a plot comparing training values and predicted values.

    Args:
        training_values (array-like): The true training values.
        predictions (array-like): The predicted values from the model.
        input_data (pd.DataFrame): The input data used for predictions.

    Returns:
        plt.Figure: The generated plot figure.
        bytes: The CSV content for download.
    """
    # Plot training vs predicted values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(training_values)), training_values, label='Training',
            marker='o', color='blue')
    ax.plot(range(len(predictions)), predictions, label='Predictions',
            marker='x', color='red')
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Training vs. Predicted Values")
    ax.legend()

    # Save the plot to a file
    plot_path = "/tmp/plot_prediction.png"
    plt.savefig(plot_path)

    # Return the path of the generated plot image
    return plot_path

# pydoc.writedoc('pipeline_graphing')

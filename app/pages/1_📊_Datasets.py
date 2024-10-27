import streamlit as st
import pandas as pd
import os
from io import BytesIO
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.database import Database
from autoop.core.storage import LocalStorage

# Initialize the AutoMLSystem and Database instance
automl = AutoMLSystem.get_instance()
storage = LocalStorage(base_path="./assets")  # Define the storage path as needed
database = Database(storage=storage)  # Assuming storage is passed as part of system

st.title("Dataset Management")

# Upload a new dataset
st.header("Upload a New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    dataset_name = st.text_input("Enter a name for this dataset:", value=uploaded_file.name.split(".")[0])
    
    if st.button("Save Dataset"):
        dataset = Dataset.from_dataframe(df, name=dataset_name, asset_path=f"{dataset_name}.csv")
        database.set("datasets", dataset_name, dataset.dict())
        st.success(f"Dataset '{dataset_name}' saved successfully.")

# Display list of saved datasets
st.header("Available Datasets")
datasets = database.list("datasets")

if datasets:
    dataset_names = [name for name, _ in datasets]
    selected_dataset_name = st.selectbox("Select a dataset to view", options=dataset_names)

    if selected_dataset_name:
        selected_dataset = database.get("datasets", selected_dataset_name)
        try:
            df = pd.read_csv(BytesIO(selected_dataset['data']))  # Assuming data is stored in binary format
            st.subheader(f"Dataset: {selected_dataset_name}")
            st.write(df.head())
            
            if st.button("Delete Dataset"):
                database.delete("datasets", selected_dataset_name)
                st.warning(f"Dataset '{selected_dataset_name}' has been deleted.")
        except FileNotFoundError:
            st.error(f"File for dataset '{selected_dataset_name}' not found.")
else:
    st.write("No datasets available.")

# TODO:
# - litania ifów, explained below
# - zmienić model attributes w deployment

# - go though instruction and checklist
# - documentation

# - zadecydować do idzie do raportu
# - ustalić co jest w grafie

# - visual representation
# - extra features
# - 11. OOP-011: As a datascientist, I would like to create a model run where
#   I can fine-tune an existing model on a new dataset.

# pipeline details: delete type, parameters if not needed, trained, 
# display metrics in pipeline details (only eval ones)
# wymyślić jak przechowywać model metrics i je dodać do raportu

"""
What these classification models accept:
- logistic:
    - any values in parameter columns
    - only 0 and 1 in the results colums
    - if result column has 2 values, it can be accepted
    - if those values are not 0 and 1, add a button to determinw what is
      treated as 0 and what as 1
    - add a posibility to change function parameters
- knn:
    - I see no issues
    - just allow the user to choose the value of k
- treeclassifier:
    - just allow the user to choose max_depth

Regression models:
- multiple linreg:
    - we just assume params are independent
- treereg:
    - LGTM, just allow the user to choose max_depth
- ridge:
    - when a data set contains a higher number of predictor variables than the
      number of observations. The second-best scenario is when
      multicollinearity is experienced in a set
"""

"""
    if task_type == "Classification":
        if len(set(selected_target_feature)) != 2:
            CLASSIFICATION_MODELS.delete(logistic_regression)
    else:
        tu trzeba sprawdzić multicollinearity
        jak jest to rigde, not multiple
"""


"""
import pandas as pd



"""
"""
    st.header("3.5 Select parameters for the ML model:")

# Get the selected model from the user
selected_model = st.selectbox("Choose a model:", ["knn", "ridge", "logistic", "treereg", "treeclass"])

# Define parameter selection based on the selected model
if selected_model == "knn":
    # Allow the user to choose the value of k for K-Nearest Neighbors
    k = st.number_input("Select value of k:", min_value=1, step=1)
    st.write("You selected k =", k)
    # Pass k to your function as needed here

elif selected_model == "ridge":
    # Allow the user to choose alpha for Ridge Regression
    alpha = st.number_input("Select alpha value:", min_value=0.01, step=0.01)
    st.write("You selected alpha =", alpha)
    # Pass alpha to your function as needed here

elif selected_model == "logistic":
    # Check if params_from_last_col is a valid integer and between 0 and 1
    params_from_last_col = st.text_input("Enter a parameter value (0 or 1):", value="0")
    if not (params_from_last_col.isdigit() and int(params_from_last_col) in [0, 1]):
        st.warning("Please enter 0 or 1.")
        choose_label_btn = st.button("Choose labels (0 and 1)")
        if choose_label_btn:
            st.write("Label selection button clicked.")
            # Implement label selection logic as needed

    # Allow the user to choose learning rate and number of iterations
    learning_rate = st.number_input("Select learning rate:", min_value=0.0001, max_value=1.0, step=0.0001, format="%.4f")
    n_iterations = st.number_input("Select number of iterations:", min_value=1, step=1)
    st.write(f"You selected learning rate = {learning_rate} and iterations = {n_iterations}")
    # Pass learning_rate and n_iterations to your function as needed here

elif selected_model in ["treereg", "treeclass"]:
    # Allow the user to choose max_depth for Tree-based models
    max_depth = st.number_input("Select max depth:", min_value=1, step=1)
    st.write("You selected max_depth =", max_depth)
    # Pass max_depth to your function as needed here
    """

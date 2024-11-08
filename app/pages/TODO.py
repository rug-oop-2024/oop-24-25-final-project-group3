# TODO:
# - graph
# - replace 1 & 0 for log in predict
# - fix log predict

# - go though instruction and checklist
# - documentation
# - report is being generated, please wait...
# - plot ostatni na drugą część strony


"""
FROM GIT
The code should be easy to modify and extend. You should be able to add new
features without having to rewrite the entire codebase or heavy refactoring.
Important: We prefer that you do not use libraries or frameworks for this
reason. We want to see how you structure your code and how you solve problems.
Furthermore, many of the tools available may not be customisable enough for
our needs. Hence, if you want to include a library, please make sure to
justify it.

The code should be secure. Isolate your secrets and keys from the codebase. Do
not commit your secrets and keys to the repository. Avoid XSS and CSRF attacks.
Enforce the principle of least privilege. Validate user input, etc.

Feature column types should be automatically inferred from the data.
Task type should be automatically inferred from the data.
Different models should be trained in parallel to evaluate the best model.
The code should infere wether a target column is valid for prediction given
the capabilities of your algorithm.

The code should be documented. You should be able to understand what the code
does without having to spend a lot of time on it. Significant design decisions
should be documented as well. Each decision should be given an ID. HTML
documentation as in lecture 10 gives +0.5
"""

"""
What these classification models accept:
- logistic:
    - if those values are not 0 and 1, add a button to determinw what is
      treated as 0 and what as 1
    - add a posibility to change function parameters
- knn:
    - I see no issues
    - just allow the user to choose the value of k
- treeclassifier:
    - the user to choose max_depth

Regression models:
- treereg:
    - allow the user to choose max_depth
- ridge:
    - allow choosing alpha
"""

"""
Jakbyśmy mieli za dużo czasu wolnego

if selected_model != "MultipleLinearRegression":

        st.header("3.5 Select parameters for the ML model:")

        # Define parameter selection based on the selected model
        if selected_model == "KNearestNeighbors":
            # Allow the user to choose the value of k for K-Nearest Neighbors
            k = st.number_input("Select value of k:", min_value=1, step=1)
            st.write("You selected k =", k)
            # Pass k to your function as needed here

        elif selected_model == "RidgeRegression":
            # Allow the user to choose alpha for Ridge Regression
            alpha = st.number_input("Select alpha value:", min_value=0.01,
                                    step=0.01)
            st.write("You selected alpha =", alpha)
            # Pass alpha to your function as needed here

        elif selected_model == "LogisticRegression":
            # Check if params_from_last_col is an integer and between 0 and 1
            params_from_last_col = st.text_input("Enter a parameter value (0"
                                                 " or 1):", value="0")
            if not (params_from_last_col.isdigit() and
                    int(params_from_last_col) in [0, 1]):
                st.warning("Please enter 0 or 1.")
                choose_label_btn = st.button("Choose labels (0 and 1)")
                if choose_label_btn:
                    st.write("Label selection button clicked.")
                    # we have to sub for all values here and pass the changed
                    # values into the later parts of the code

            # Allow the user to choose learning rate and number of iterations
            learning_rate = st.number_input("Select learning rate:",
                                            min_value=0.0001, max_value=1.0,
                                            step=0.0001, format="%.4f")
            n_iterations = st.number_input("Select number of iterations:",
                                           min_value=1, step=1)
            st.write(f"You selected learning rate = {learning_rate} and "
                     f"iterations = {n_iterations}")
            # Pass learning_rate and n_iterations to your function here

        elif selected_model in ["DecisionTreeRegression",
                                "DecisionTreeClassification"]:
            # Allow the user to choose max_depth for Tree-based models
            max_depth = st.number_input("Select max depth:", min_value=1,
                                        step=1)
            st.write("You selected max_depth =", max_depth)
            # Pass max_depth to your function as needed here
"""

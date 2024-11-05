# TODO:
# - litania ifów
#       knn -> n_neighbors
#       logistic -> learning_rate, n_iterations, check if numbers in last row,
#                   else declare options for classifying
#       ridge_reg -> alpha, must be > 0
# - zmienić model attributes w deployment
# - parameters adjust in modelling!
#       - knn: k
# - fix logistic regression jeśli categorical data to string (co ma uważać za
#                                                             0 a co za 1)

# - go though instruction and checklist
# - documentation

# - visual representation
# - extra features
# - 11. OOP-011: As a datascientist, I would like to create a model run where
#   I can fine-tune an existing model on a new dataset.

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
    - LGTM
- ridge:
    - when a data set contains a higher number of predictor variables than the
      number of observations. The second-best scenario is when
      multicollinearity is experienced in a set
"""

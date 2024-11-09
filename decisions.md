# DSC-001: Model - Implementation Approach
# Date: 2024-10-23
# Decision: Define a custom implementation for model structures
# Status: Accepted
# Motivation: Provide flexibility in training and customization
# Reason: Allows adaptation to specific needs and model types
# Limitations: Potential complexity and maintenance requirements

# DSC-002: Pipeline - Implementation Approach
# Date: 2024-10-23
# Decision: Custom pipeline implementation with integrated components
# Status: Accepted
# Motivation: Create a unified structure for model development
# Reason: Easier management of data preprocessing, training, and evaluation
# Limitations: Higher initial development time
# Alternatives: Prefabricated pipeline tools like scikit-learn's `Pipeline`

# DSC-003: Artifact - Implementation Approach
# Date: 2024-10-23
# Decision: Implement a custom artifact system for data and model storage
# Status: Accepted
# Motivation: Centralized management of project artifacts
# Reason: Facilitates reproducibility and structured asset management
# Limitations: Custom development and storage space

# DSC-004: Feature - Implementation Approach
# Date: 2024-10-23
# Decision: Implement a dedicated `Feature` class for classifying if data is categorical or numerical
# Status: Accepted
# Motivation: Consistent feature handling within the pipeline
# Reason: Standardizes data input/output processing
# Limitations: Initial development effort for custom classes
# Alternatives: Using Pandas DataFrame directly

# DSC-005: Metric Selection - Approach
# Date: 2024-11-01
# Decision: Include specific metrics and log metrics for evaluation
# Status: Accepted
# Motivation: Provide diverse metrics for better model evaluation, with logistic regression using different metrics due to its structure
# Reason: add deeper insights for classification and regression tasks
# Limitations: Logistic regression metrics require binary classification
# Alternatives: Standard metrics only

# DSC-006: Model Extensions - Choice of Models and scikit-learn
# Date: 2024-11-01
# Decision: Utilize models with extensions and integrate scikit-learn
# Status: Accepted
# Motivation: Use proven, efficient model libraries, using different models so that data with multicollinearity or without it could be analysed in regression models and binary and non-binary categorical data could be analysed for classification models
# Reason: scikit-learn provides reliable and well-documented models
# Limitations: Limited flexibility compared to custom models
# Alternatives: Fully custom implementations or using TensorFlow

# DSC-007: Modelling - Selection of Features and Inputs
# Date: 2024-11-03
# Decision: Implement feature and input selection with defined limits
# Status: Accepted
# Motivation: Ensure model reliability and maintain relevance of inputs
# Reason: Certain models require speficic data values, e.g. logistic regression need binary data in its target feature
# Limitations: User needs domain knowledge for proper input selection
# Alternatives: Automated feature selection tools

# DSC-0008: Graphs for Visualization - Decision
# Date: 2024-11-03
# Decision: plotting predictions vs original data
# Status: Accepted
# Motivation: Enhance understanding of model performance and data trends
# Reason: Visuals aid in identifying patterns and anomalies
# Limitations: Graphs may require interpretation skills
# Alternatives: Text-based data summaries only

# DSC-009: Data for Report - Selection and User Generation
# Date: 2024-11-03
# Decision: Allow users to select data for generating reports
# Status: Accepted
# Motivation: Provide customizability for user-specific reports
# Reason: Improves user engagement and relevance of outputs
# Limitations: Users may require guidance for optimal report generation
# Alternatives: Pre-defined data selection

# DSC-0010: Keras - Use for Graph Implementation
# Date: 2024-11-08
# Decision: Use Keras for creating the loss plot and to visualise data flow through the pipeline
# Status: Accepted
# Motivation: Keras offers an easy-to-use API for graphic display of machine learning processes
# Reason: Suitable for quick and effective graph implementation
# Limitations: Additional dependencies
# Alternatives: Directly using TensorFlow or PyTorch

# DSC-0011: Sphinx - Documentation Tool Selection
# Date: 2024-11-09
# Decision: Use Sphinx for generating project documentation
# Status: Accepted
# Motivation: Create structured and easy-to-navigate documentation
# Reason: Widely used in the Python community with excellent support
# Limitations: Setup and configuration take time
# Alternatives: Using Markdown-only or other doc generators such as pydoc
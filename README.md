# AI/ML-Based Techniques for Early Detection of Liver Ailments

## Overview
This project aims to develop a system for the early detection of liver ailments using a combination of image processing techniques and machine learning algorithms. Initially, the disease detection was attempted through image processing methods, but due to suboptimal results, the approach shifted towards utilizing machine learning algorithms such as decision trees, naive Bayes, logistic regression, etc. The project evaluates the accuracy of these methods on both test and train datasets and investigates different types of errors encountered.

## Image Processing Techniques
In the image processing phase, various techniques were employed to preprocess and analyze liver images for disease detection. These techniques include:

- **Edge Detection**: Utilization of edge detection algorithms such as Prewitt, Robert, etc., to identify edges and boundaries within liver images.
- **Segmentation**: Implementation of segmentation techniques including thresholding, adaptive thresholding, watershed, boundary tracking, etc., to isolate and extract the liver region from the images.
Others steps include data pre processing,feature extraction etc.

## Machine Learning Algorithms
Due to limitations in the effectiveness of image processing alone, the project transitions to machine learning algorithms to enhance accuracy. The following algorithms were employed:

- **Decision Trees**: Decision tree-based classifiers are used to predict the presence of liver ailments based on features extracted from liver images.
- **Naive Bayes**: Naive Bayes classifiers are utilized for probabilistic predictions of liver ailments, assuming independence among features.
- **Logistic Regression**: Logistic regression models are applied to estimate the probability of liver ailment occurrence based on input features.
- **Support Vector Machines (SVM)**: SVMs are utilized as binary classifiers to distinguish between healthy and diseased liver conditions. SVMs aim to find the hyperplane that best separates the two classes while maximizing the margin between them.
These additional machine learning algorithms complement decision trees, naive Bayes, and logistic regression, offering a diverse set of techniques to address the challenges in detecting liver ailments accurately. Each algorithm brings unique strengths and capabilities, contributing to the overall effectiveness of the detection system.

## Evaluation Metrics
To assess the performance of the developed models, various evaluation metrics are utilized, including:

- **Accuracy**: Overall correctness of predictions made by the models.
- **Precision**: Proportion of true positive predictions among all positive predictions made.
- **Recall**: Proportion of true positive predictions among all actual positive instances.
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced measure of model performance.
These are only few to be mentioned here.

## Getting Started
To begin with the project, follow these steps:

- Download the Main folder and four CSV files to your local device.
- Ensure that you have MATLAB installed on your system.
- Place the Main folder in your MATLAB directory.
- Open MATLAB and navigate to the Main folder.
- Run the normal.m script for image processing tasks.
- Execute the AIMLGUI2.m script for machine learning tasks.
- These scripts will guide you through the setup and usage of the respective modules for image processing and machine learning.

## Contributors
- Mallika Muskan.
- Mahi Saxena.
- Khushi Rawat.

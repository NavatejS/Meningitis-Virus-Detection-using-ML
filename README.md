# Meningitis-Virus-Detection-using-ML

This project proposes a machine learning model SVM and Random Forest to detect meningitis virus. The models extracted significant features from the dataset then classified 3 classes: Healthy, Bacterial and Virus. Among different supervised learning techniques, we here explored the meningitis virus and many models of machine learning to compare all of them.

## Approach:

Data Analysis and Preprocessing: Utilize pandas for dataset analysis and preprocessing tasks. Employ matplotlib and seaborn for data visualization to gain insights into the dataset's characteristics and distributions.

Feature Extraction and Selection: Extract significant features from the dataset that are indicative of meningitis virus presence. Explore various feature selection techniques to enhance model performance.

Model Building and Evaluation: Implement SVM and Random Forest classifiers using the scikit-learn library. Train these models on the preprocessed dataset and evaluate their performance using appropriate metrics. Compare the performance of both models to identify the most effective approach for meningitis virus detection.

Label Encoding: Utilize label encoder from the scikit-learn library to convert categorical values into numerical format for model training. Encode categorical values into integers (0, 1, 2, etc.) to prepare the data for model input.

The dataset used is Meningitis Classification. It has 1000 rows and  12 columns. 


## Evaluation

Evaluation is very important in data science. It helps you to understand the performance of your model . There are many different evaluation metrics out there but only some of them are suitable to be used for regression.

In this project we will MSE, RMSE to evaluate our models:

Mean Square Error: The MSE is calculated as the mean or average of the squared differences between predicted and expected target values in a dataset.

Root Mean Square Error: RMSE is the square root of the mean square error (MSE).

We will evaluate the model based on accuracy, mean square error (MSE), root mean square error (RMSE), precision, recall, f1-score and confusion matrix.

<img width="899" alt="Screenshot 2024-05-10 at 3 14 31 PM" src="https://github.com/NavatejS/Meningitis-Virus-Detection-using-ML/assets/164386165/334813e9-a8a4-40b6-be44-b2bbab1fe90f">

<img width="871" alt="Screenshot 2024-05-10 at 3 12 17 PM" src="https://github.com/NavatejS/Meningitis-Virus-Detection-using-ML/assets/164386165/bcb5843f-0a8f-4f4c-b641-24675a402326">


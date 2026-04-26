# probabilistic-ml
# Probabilistic ML Foundations & Robust Regression 🚀

A deep dive into Probabilistic Machine Learning concepts, featuring custom implementations of Naive Bayes, probability density estimations, and an analysis of how statistical distributions shape neural network cost functions (MSE vs. MAE) using PyTorch.

## 📌 Overview

This repository bridges the gap between statistical theory and applied machine learning. It demonstrates how underlying probabilistic assumptions (like Gaussian vs. Heavy-tailed distributions) directly impact model performance, especially in the presence of outliers. 

The project is divided into two main parts:
1. **Bayesian Classification:** Implementing Naive Bayes from scratch using both Gaussian and Kernel Density Estimation (KDE).
2. **Probabilistic Cost Functions:** Exploring how Mean Squared Error (MSE) assumes a Gaussian distribution, while Mean Absolute Error (MAE) assumes a Laplace distribution, and how to implement robust regression in PyTorch to handle outliers.

## ✨ Key Features & Concepts Covered

### Part 1: Classification & Density Estimation
* **Gaussian Naive Bayes (From Scratch):** Implementation of Bayes' theorem computing class probabilities from feature likelihoods without relying on high-level library estimators.
* **Kernel Density Estimation (KDE):** Non-parametric density estimation to model complex data distributions where Gaussian assumptions fail.
* **Sklearn Benchmarking:** Comparing the custom from-scratch implementation against `sklearn.naive_bayes.GaussianNB`.

### Part 2: Statistical Distributions & Cost Functions
* **The impact of "Heavy Tails":** Visualizing Gaussian vs. Student's t-distribution vs. Laplace distributions.
* **Heteroscedastic Variance Estimation:** Building a PyTorch model that predicts both the mean and the variance (uncertainty) of data points simultaneously using Negative Log-Likelihood (NLL).
* **Robust Regression:** Demonstrating why MSE (Gaussian) fails with outliers and how switching to L1 Loss (Laplace assumption) provides a highly robust fit.

## 🛠️ Technologies Used

* **Python 3.x**
* **PyTorch** (for custom loss functions and neural network optimization)
* **Scikit-Learn** (for KDE and baseline models)
* **SciPy & NumPy** (for statistical distributions and matrix operations)
* **Matplotlib** (for statistical visualizations and decision boundaries)
* **Pandas** (for data manipulation)

## 📊 Visualizations

The code generates several rich visualizations to help understand the math behind the models:
* Gaussian vs. KDE Density Estimation distributions.
* The visual difference between thin tails (Gaussian) and heavy tails (t-distribution/Laplace).
* Linear Regression bounds showing $\pm 1\sigma$ and $\pm 2\sigma$ confidence intervals for heteroscedastic data.
* A direct visual comparison of MSE vs. MAE regression lines reacting to synthetic outliers.

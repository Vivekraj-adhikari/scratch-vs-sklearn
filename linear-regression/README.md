# Used Car Price Prediction using Linear Regression (From Scratch)

This project implements Linear Regression from scratch using gradient descent
to predict used car prices based on vehicle age and kilometers driven.

## Project Overview

The objective of this project is to understand the internal working of
Linear Regression by implementing it from scratch and comparing its
performance with sklearn’s LinearRegression model.

## Dataset

- Used Cars dateset from kaggle
- Features used:
  - `km_driven`
  - `years_old`
- Target variable:
  - `price`

## Model Implementation

A Linear Regression model was implemented from scratch using
batch gradient descent to minimize Mean Squared Error (MSE).

Key components:
- Weight and bias initialization
- Batch Gradient Descent
- Mean Squared Error (MSE)
- R² Score calculation

## Data Preprocessing

- Performed outlier analysis using box plots
- Extreme outliers were removed using percentile-based thresholds
- Split the dataset into training and testing sets

## Model Evaluation

The model was evaluated using:
- Mean Squared Error (MSE)
- R² Score

The custom implementation was compared with sklearn's LinearRegression
model to validate correctness and performance.

## Results

The custom Linear Regression model achieved performance comparable
to sklearn’s implementation, demonstrating the correctness of
the gradient descent-based approach.

## How to Run

1. Clone the repository
2. Install required libraries:
   - numpy
   - pandas
   - matplotlib
3. Open `used_cars.ipynb` and run all cells

## Key Learnings

- Understood the mathematics behind Linear Regression
- Implemented gradient descent from scratch
- Learned how to evaluate regression models using MSE and R²
- Gained experience comparing custom ML models with sklearn

# Housing Prices Prediction

This project implements a Random Forest Regression model to predict house prices based on structural and demographic property features. The dataset is taken from the [Home Data for ML Course](https://www.kaggle.com/competitions/home-data-for-ml-course) competition on Kaggle.

## Objective

The goal of this project is to build, train, and evaluate a machine learning model capable of predicting the sale price of a house using a set of input features such as lot size, year built, number of rooms, and bathrooms.

## Technologies Used

- Python 3
- Pandas for data manipulation
- Scikit-learn for machine learning
- RandomForestRegressor as the main model

## Project Structure

housing-prices-ml/
│
├── data/ # Contains the training and test CSV files (not uploaded to GitHub)
│
├── src/
│ └── run_model.py # Main script for training and generating predictions
│
└── submission.csv # Model predictions (generated automatically)

##Example Output

When executed, the script prints the validation MAE (Mean Absolute Error) and creates the submission file:

Validation MAE: 17,500
File 'submission.csv' created successfully!

_This project was developed for educational purposes as part of a machine learning practice exercise using Python and Scikit-learn._

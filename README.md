# Bike share demand 🚴

This repository is used for creating a machine leaning model that can accurately forecast bike rental demand at by hour in the day based on seasonal and environmental factors.

## Dataset

The dataset is used is from a [Kaggle competition here]([https://link-url-here.org](https://www.kaggle.com/competitions/bike-sharing-demand/)).
An EDA is included for this dataset to identify any feature extraction or extension possibilities to improve the models performance.

## Result

The result of this project is a model that is in the within the top 7% of the models submitted.
The best submitted score of this model was 0.39809.
This submission was done after the competition took place so the score is now shown on the leaderboard.

## Key points

There are various file with different ways of creating a model. Other files are used to contain shared functions and one is made for making a Kaggle submission file.

## Development sequence

The development of the project happened in the following sequence

1. Setup pipeline to Kaggle with basic features
2. Start more advanced feature engineering
    - Extract information from the DateTime index
3. Start extending features
    - Use polynomial features
    - My r2 score was going up but my Kaggle score was going down
4. Start own implemented grid search with RandomTree Regressor
5. Start using built in Grid search with Cross validation 
6. Start Multi model Voting with scoring threshold
    - Set scoring threshold (informed Democracy)
7. Start grid search on different feature selection configurations
# HarvardX_Capstone_MovieLens_R
Data Science Capstone - HarvardX Professional Certificate in Data Science - MovieLens Project

## Dataset
For this project, I created a movie recommendation system using the MovieLens dataset. You can find the entire latest MovieLens dataset at https://grouplens.org/datasets/movielens/latest/. 

I used the 10M version of the MovieLens dataset to make the computation a little easier.

## Evaluation
RMSE was used for model evaluation. 

## File structure
There are three files:
- Report in the form of an .Rmd file
- Report in the form of a PDF document knit from the .Rmd file
- R script that generates predicted movie ratings and calculates RMSE

## RMSE grading
- 0 points: No RMSE reported AND/OR code used to generate the RMSE appears to violate the edX Honor Code
- 5 points: RMSE >= 0.90000 AND/OR the reported RMSE is the result of overtraining (validation set - the final hold-out test set - ratings used for anything except reporting the final RMSE value) AND/OR the reported RMSE is the result of simply copying and running code provided in previous courses in the series
- 10 points: 0.86550 <= RMSE <= 0.89999
- 15 points: 0.86500 <= RMSE <= 0.86549
- 20 points: 0.86490 <= RMSE <= 0.86499
- 25 points: RMSE < 0.86490

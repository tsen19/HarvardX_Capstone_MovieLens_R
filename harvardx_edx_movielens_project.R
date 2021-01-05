###################################################################################################
## Script name: harvardx_edx_movielens_project.R
## Project: HarvardX PH125.9x | Data Science: Capstone
## Script purpose: Train a recommender model and evaluate with RMSE
## Date: September 1, 2020
## Author: Neli Tsereteli
## Contact information: 
## Notes: 
###################################################################################################

# Set up the environment
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)  # a set of packages that work in harmony
library(caret)      # misc functions for training and plotting classification and regression models
library(data.table) # extension of 'data.frame'
library(stringr)    # simple, consistent wrappers for common string operations
library(lubridate)  # functions to work with date-times and time-spans
library(knitr)      # general-purpose tool for dynamic report generation in R


# Download the datasets 1) ratings and 2) movies
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Read in ratings
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Read in movies
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

# Clean up the datasets and merge them
# Assign column names
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                            title = as.character(title),
#                                            genres = as.character(genres))

# Merge the tables
movielens <- left_join(ratings, movies, by = "movieId")

# Create train-test partitions
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set. 
# That is, ensure that we do not include users and movies in the test
# set that do not appear in the training set. 
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Create train-test partitions
set.seed(42)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-test_index,]
edx_test <- edx[test_index,]

# Make sure that we do not include users and movies in the test
# set that do not appear in the training set
edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Clean up the environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Helper functions
# get_genres
get_genres <- function(dataset) {
  # Create a logical variable for each genre
  dataset <- dataset %>% mutate(
    is_comedy = ifelse(str_detect(genres, "Comedy"), 1, 0),
    is_romance = ifelse(str_detect(genres, "Romance"), 1, 0),
    is_action = ifelse(str_detect(genres, "Action"), 1, 0),
    is_crime = ifelse(str_detect(genres, "Crime"), 1, 0),
    is_thriller = ifelse(str_detect(genres, "Thriller"), 1, 0),
    is_drama = ifelse(str_detect(genres, "Drama"), 1, 0),
    is_scifi = ifelse(str_detect(genres, "Sci-Fi"), 1, 0),
    is_adventure = ifelse(str_detect(genres, "Adventure"), 1, 0),
    is_children = ifelse(str_detect(genres, "Children"), 1, 0),
    is_fantasy = ifelse(str_detect(genres, "Fantasy"), 1, 0),
    is_war = ifelse(str_detect(genres, "War"), 1, 0),
    is_animation = ifelse(str_detect(genres, "Animation"), 1, 0),
    is_musical = ifelse(str_detect(genres, "Musical"), 1, 0),
    is_western = ifelse(str_detect(genres, "Western"), 1, 0),
    is_mystery = ifelse(str_detect(genres, "Mystery"), 1, 0),
    is_filmnoir = ifelse(str_detect(genres, "Film-Noir"), 1, 0),
    is_horror = ifelse(str_detect(genres, "Horror"), 1, 0),
    is_documentary = ifelse(str_detect(genres, "Documentary"), 1, 0),
    is_imax = ifelse(str_detect(genres, "IMAX"), 1, 0),
    is_no_genre = ifelse(str_detect(genres, "(no genres listed)"), 1, 0))
}

# genre_effect
# Create a function that calculates genre effect.
# as.symbol() functionality inspired by the post at:
# https://stackoverflow.com/questions/49371260/using-variables-as-arguments-in-summarize
genre_effect <- function(train_set, genre) {
  # Get effect name
  effect <- str_sub(genre, start = 4, end = -1)
  effect <- as.symbol((paste0("effect_", effect)))
  # Calculate average
  average <- mean(train_set$rating)
  # Calculate effect size
  genre_compared_to_average <- train_set %>%
    group_by(!!genre) %>%
    summarize(!!effect := mean(rating - average - effect_movie - effect_user))
}

# get_genre_score()
# Calculate genre score by summing up all the genre effects.  
# This was inspired by the idea behind genetic risk scores.
get_genre_score <- function(dataset) {
  dataset <- dataset %>% 
    # sum up all the effects
    mutate(genre_score = 
             effect_comedy + 
             effect_romance + 
             effect_action +
             effect_crime +
             effect_thriller +
             effect_drama +
             effect_scifi + 
             effect_adventure +
             effect_children +
             effect_fantasy +
             effect_war +
             effect_animation +
             effect_musical +
             effect_western +
             effect_mystery +
             effect_filmnoir +
             effect_horror +
             effect_documentary +
             effect_imax +
             effect_no_genre)
}

# get_release_year()
# Create a function to extract release year for each movie.  
# For each movie we expect a name with a year in parentheses, like in "Flinstones, The (1994)".
get_release_year <- function(dataset) {
  dataset$release_year <- str_extract(dataset$title, pattern = "\\([0-9]{4}\\)")
  dataset$release_year <- str_replace_all(dataset$release_year, 
                                          pattern = "\\(|\\)", 
                                          replacement = "")
  dataset$release_year <- as.numeric(dataset$release_year)
  return(dataset)
}

# Loss function - rmse()
rmse <- function(true_rating, predicted_rating) {
  sqrt(mean((true_rating - predicted_rating)^2))
}

# Data exploration    
# Let's look at the edx set
str(edx)
head(edx)

# Number of unique users and unique movies
edx %>% summarize(unique_users = n_distinct(userId), unique_movies = n_distinct(movieId))

# Distribution of movie ratings. 
# Some movies get rated more than others.
edx %>% group_by(movieId) %>%
  summarise(num_ratings = n()) %>%
  ggplot(aes(x = num_ratings)) + geom_histogram()

# Similarly, some users rate more movies than other users.
edx %>% group_by(userId) %>%
  summarise(num_ratings = n()) %>%
  ggplot(aes(x = num_ratings)) + geom_histogram(bins = 40)

# Feature engineering
# Get all the genres - 20 unique
genre_list <- unlist(str_split(edx$genres, pattern = "\\|"))
(genre_unique <- unique(genre_list))

# Apply helper function to get genres
edx_train <- get_genres(edx_train)
edx_test <- get_genres(edx_test)

# Calculate number of unique genres for each movie
edx_train <- edx_train %>% mutate(num_genre = str_count(genres, pattern = "\\|") + 1)
edx_test <- edx_test %>% mutate(num_genre = str_count(genres, pattern = "\\|") + 1)

# Get release year for each movie 
edx_train <- get_release_year(edx_train)
edx_test <- get_release_year(edx_test)

# Release year distribution
hist(edx_train$release_year)
range(edx_train$release_year)

# Divide movie age into 5-year intervals and look at them
edx_train <- edx_train %>% 
  mutate(age_group = (year(now())-edx_train$release_year) - ((year(now())-edx_train$release_year)%%5))
edx_test <- edx_test %>% 
  mutate(age_group = (year(now())-edx_test$release_year) - ((year(now())-edx_test$release_year)%%5))

# Tabulate
table(edx_train$age_group)

# Sequel or not? 
# Naive sequel logical variable.
edx_train <- edx_train %>% mutate(sequel = str_detect(title, pattern = ":"))
edx_test <- edx_test %>% mutate(sequel = str_detect(title, pattern = ":"))

# Let's see if it makes sense
edx_train %>% filter(sequel) %>% head(20) %>% select(title)
table(edx_train$sequel)

# Extract timestamp years
edx_train <- edx_train %>% mutate(year_stamp = year(as_datetime(timestamp)))
edx_test <- edx_test %>% mutate(year_stamp = year(as_datetime(timestamp)))

# Building and comparing models
# Model 1: rating = average
average <- mean(edx_train$rating)
(rmse_average_guess <- rmse(edx_test$rating, average))

# Model 2: rating = average + movie effect
# First calculate how good or bad a movie is compared to the average. 
movie_compared_to_average <- edx_train %>%
  group_by(movieId) %>%
  summarize(effect_movie = mean(rating - average))

# Merge movie effect with test set
edx_test <- left_join(edx_test, movie_compared_to_average, by = "movieId")
edx_train <- left_join(edx_train, movie_compared_to_average, by = "movieId")

# Predictions and RMSE
predicted_rating <- average + edx_test$effect_movie
(rmse_movie_effect <- rmse(edx_test$rating, predicted_rating))

# Model 3: rating = average + movie effect + user effect
# Calculate how "impressed" or "unimpressed" a user is compared 
# to the average user after considering the overall mean and the movie effect.
user_compared_to_average <- edx_train %>%
  group_by(userId) %>%
  summarize(effect_user = mean(rating - average - effect_movie))

# Merge user effect with test set
edx_test <- left_join(edx_test, user_compared_to_average, by = "userId")
edx_train <- left_join(edx_train, user_compared_to_average, by = "userId")

# Predictions and RMSE
predicted_rating <- average + edx_test$effect_movie + edx_test$effect_user
(rmse_user_effect <- rmse(edx_test$rating, predicted_rating))

# Model 4: rating = average + movie + user + genre score

# Create genre effects for all the genres
# Create genre effects
# 1. C O M E D Y 
comedy_compared_to_average <- genre_effect(edx_train, sym("is_comedy"))
edx_train <- left_join(edx_train, comedy_compared_to_average, by = "is_comedy")
edx_test <- left_join(edx_test, comedy_compared_to_average, by = "is_comedy")

# 2. R O M A N C E
romance_compared_to_average <- genre_effect(edx_train, sym("is_romance"))
edx_train <- left_join(edx_train, romance_compared_to_average, by = "is_romance")
edx_test <- left_join(edx_test, romance_compared_to_average, by = "is_romance")

# 3. A C T I O N
action_compared_to_average <- genre_effect(edx_train, sym("is_action"))
edx_train <- left_join(edx_train, action_compared_to_average, by = "is_action")
edx_test <- left_join(edx_test, action_compared_to_average, by = "is_action")

# 4. C R I M E
crime_compared_to_average <- genre_effect(edx_train, sym("is_crime"))
edx_train <- left_join(edx_train, crime_compared_to_average, by = "is_crime")
edx_test <- left_join(edx_test, crime_compared_to_average, by = "is_crime")

# 5. T H R I L L E R 
thriller_compared_to_average <- genre_effect(edx_train, sym("is_thriller"))
edx_train <- left_join(edx_train, thriller_compared_to_average, by = "is_thriller")
edx_test <- left_join(edx_test, thriller_compared_to_average, by = "is_thriller")

# 6. D R A M A 
drama_compared_to_average <- genre_effect(edx_train, sym("is_drama"))
edx_train <- left_join(edx_train, drama_compared_to_average, by = "is_drama")
edx_test <- left_join(edx_test, drama_compared_to_average, by = "is_drama")

# 7. S C I - F I  
scifi_compared_to_average <- genre_effect(edx_train, sym("is_scifi"))
edx_train <- left_join(edx_train, scifi_compared_to_average, by = "is_scifi")
edx_test <- left_join(edx_test, scifi_compared_to_average, by = "is_scifi")

# 8. A D V E N T U R E 
adventure_compared_to_average <- genre_effect(edx_train, sym("is_adventure"))
edx_train <- left_join(edx_train, adventure_compared_to_average, by = "is_adventure")
edx_test <- left_join(edx_test, adventure_compared_to_average, by = "is_adventure")

# 9. C H I L D R E N
children_compared_to_average <- genre_effect(edx_train, sym("is_children"))
edx_train <- left_join(edx_train, children_compared_to_average, by = "is_children")
edx_test <- left_join(edx_test, children_compared_to_average, by = "is_children")

# 10. F A N T A S Y 
fantasy_compared_to_average <- genre_effect(edx_train, sym("is_fantasy"))
edx_train <- left_join(edx_train, fantasy_compared_to_average, by = "is_fantasy")
edx_test <- left_join(edx_test, fantasy_compared_to_average, by = "is_fantasy")

# 11. W A R 
war_compared_to_average <- genre_effect(edx_train, sym("is_war"))
edx_train <- left_join(edx_train, war_compared_to_average, by = "is_war")
edx_test <- left_join(edx_test, war_compared_to_average, by = "is_war")

# 12. A N I M A T I O N
animation_compared_to_average <- genre_effect(edx_train, sym("is_animation"))
edx_train <- left_join(edx_train, animation_compared_to_average, by = "is_animation")
edx_test <- left_join(edx_test, animation_compared_to_average, by = "is_animation")

# 13. M U S I C A L  
musical_compared_to_average <- genre_effect(edx_train, sym("is_musical"))
edx_train <- left_join(edx_train, musical_compared_to_average, by = "is_musical")
edx_test <- left_join(edx_test, musical_compared_to_average, by = "is_musical")

# 14. W E S T E R N
western_compared_to_average <- genre_effect(edx_train, sym("is_western"))
edx_train <- left_join(edx_train, western_compared_to_average, by = "is_western")
edx_test <- left_join(edx_test, western_compared_to_average, by = "is_western")

# 15. M Y S T E R Y
mystery_compared_to_average <- genre_effect(edx_train, sym("is_mystery"))
edx_train <- left_join(edx_train, mystery_compared_to_average, by = "is_mystery")
edx_test <- left_join(edx_test, mystery_compared_to_average, by = "is_mystery")

# 16. F I L M - N O I R
filmnoir_compared_to_average <- genre_effect(edx_train, sym("is_filmnoir"))
edx_train <- left_join(edx_train, filmnoir_compared_to_average, by = "is_filmnoir")
edx_test <- left_join(edx_test, filmnoir_compared_to_average, by = "is_filmnoir")

# 17. H O R R O R
horror_compared_to_average <- genre_effect(edx_train, sym("is_horror"))
edx_train <- left_join(edx_train, horror_compared_to_average, by = "is_horror")
edx_test <- left_join(edx_test, horror_compared_to_average, by = "is_horror")

# 18. D O C U M E N T A R Y 
documentary_compared_to_average <- genre_effect(edx_train, sym("is_documentary"))
edx_train <- left_join(edx_train, documentary_compared_to_average, by = "is_documentary")
edx_test <- left_join(edx_test, documentary_compared_to_average, by = "is_documentary")

# 19. I M A X 
imax_compared_to_average <- genre_effect(edx_train, sym("is_imax"))
edx_train <- left_join(edx_train, imax_compared_to_average, by = "is_imax")
edx_test <- left_join(edx_test, imax_compared_to_average, by = "is_imax")

# 20. N O   G E N R E 
nogenre_compared_to_average <- genre_effect(edx_train, sym("is_no_genre"))
edx_train <- left_join(edx_train, nogenre_compared_to_average, by = "is_no_genre")
edx_test <- left_join(edx_test, nogenre_compared_to_average, by = "is_no_genre")

# Clean up the environment
rm(action_compared_to_average, adventure_compared_to_average,
   animation_compared_to_average, children_compared_to_average, 
   comedy_compared_to_average, crime_compared_to_average, 
   documentary_compared_to_average, drama_compared_to_average,
   fantasy_compared_to_average, filmnoir_compared_to_average, 
   horror_compared_to_average, imax_compared_to_average,
   musical_compared_to_average, mystery_compared_to_average,
   nogenre_compared_to_average, romance_compared_to_average,
   scifi_compared_to_average, thriller_compared_to_average,
   user_compared_to_average, war_compared_to_average,
   western_compared_to_average, 
   genre_list, genre_unique, random3_movies, random3_users,
   movie_compared_to_average
)

# Calculate genre scores and look at their distribution
edx_train <- get_genre_score(edx_train)
edx_test <- get_genre_score(edx_test)

# Histogram
hist(edx_train$genre_score)

# Divide genre_score into 10 equally-distanced groups
edx_train$group <- as.numeric(cut_number(edx_train$genre_score, 10))
edx_test$group <- as.numeric(cut_number(edx_test$genre_score, 10))

# Tabulate
table(edx_train$group)

# Find genre group effect
group_compared_to_average <- edx_train %>%
  group_by(group) %>%
  summarize(effect_group = mean(rating - average - effect_movie - effect_user))

# Merge group effect with the train and test sets
edx_train <- left_join(edx_train, group_compared_to_average, by = "group")
edx_test <- left_join(edx_test, group_compared_to_average, by = "group")

# Predictions and RMSE
predicted_rating <- (average + 
                       edx_test$effect_movie + 
                       edx_test$effect_user + 
                       edx_test$effect_group)
(rmse_genre_effect <- rmse(edx_test$rating, predicted_rating))

# Model 5: rating = average + movie + user + genre score + number of genres
# Get genre number effect
number_compared_to_average <- edx_train %>%
  group_by(num_genre) %>%
  summarize(effect_num = mean(rating - average - effect_movie - effect_user - effect_group))

# Merge the group effect with the train and test sets
edx_train <- left_join(edx_train, number_compared_to_average, by = "num_genre")
edx_test <- left_join(edx_test, number_compared_to_average, by = "num_genre")

# Make predictions
predicted_rating <- (average + 
                       edx_test$effect_movie + 
                       edx_test$effect_user + 
                       edx_test$effect_group +
                       edx_test$effect_num)

# RMSE
(rmse_genre_num_effect <- rmse(edx_test$rating, predicted_rating))


# Model 6a: rating = average + movie + user + 
# genre score + number of genres + movie age group
# Get movie age group effect by calculating group means
age_compared_to_average <- edx_train %>%
  group_by(age_group) %>%
  summarize(effect_age = mean(rating - average - effect_movie - effect_user - effect_group - effect_num))

# Merge movie age effect with the train and test sets
edx_train <- left_join(edx_train, age_compared_to_average, by = "age_group")
edx_test <- left_join(edx_test, age_compared_to_average, by = "age_group")

# Make predictions
predicted_rating <- (average + 
                       edx_test$effect_movie + 
                       edx_test$effect_user + 
                       edx_test$effect_group +
                       edx_test$effect_num +
                       edx_test$effect_age)

# RMSE
(rmse_age_effect <- rmse(edx_test$rating, predicted_rating))

# Model 6b: rating = average + movie + user + genre score + number of genres + release year
# Get grouped means
release_compared_to_average <- edx_train %>%
  group_by(release_year) %>%
  summarize(effect_release = mean(rating - average - effect_movie - effect_user - effect_group - effect_num))

# Merge movie age effect with the train and test sets
edx_train <- left_join(edx_train, release_compared_to_average, by = "release_year")
edx_test <- left_join(edx_test, release_compared_to_average, by = "release_year")

# Make predictions
predicted_rating <- (average + 
                       edx_test$effect_movie + 
                       edx_test$effect_user + 
                       edx_test$effect_group +
                       edx_test$effect_num +
                       edx_test$effect_release)

# RMSE
(rmse_release_effect <- rmse(edx_test$rating, predicted_rating))

# Model 7: rating = average + movie + user + genre score + 
# number of genres + release year + sequel
# Get grouped means
sequel_compared_to_average <- edx_train %>%
  group_by(sequel) %>%
  summarize(effect_sequel = mean(rating - average - effect_movie - effect_user - effect_group - effect_num - effect_release))

# Merge movie sequel effect with the train and test sets
edx_train <- left_join(edx_train, sequel_compared_to_average, by = "sequel")
edx_test <- left_join(edx_test, sequel_compared_to_average, by = "sequel")

# Make predictions
predicted_rating <- (average + 
                       edx_test$effect_movie + 
                       edx_test$effect_user + 
                       edx_test$effect_group +
                       edx_test$effect_num +
                       edx_test$effect_release +
                       edx_test$effect_sequel)

# RMSE
(rmse_sequel_effect <- rmse(edx_test$rating, predicted_rating))

# Model8: rating = average + movie + user + genre score + 
# number of genres + release year + sequel + rating year
stamp_year_compared_to_average <- edx_train %>%
  group_by(year_stamp) %>%
  summarize(effect_year_stamp = mean(rating - average - effect_movie - effect_user 
                                     - effect_group - effect_num - 
                                       effect_release - effect_sequel))

# Merge stamp year effect with the train and test sets
edx_train <- left_join(edx_train, stamp_year_compared_to_average, by = "year_stamp")
edx_test <- left_join(edx_test, stamp_year_compared_to_average, by = "year_stamp")

# Make predictions
predicted_rating <- (average + 
                       edx_test$effect_movie + 
                       edx_test$effect_user + 
                       edx_test$effect_group +
                       edx_test$effect_num +
                       edx_test$effect_release +
                       edx_test$effect_sequel +
                       edx_test$effect_year_stamp)

# RMSE
(rmse_year_stamp_effect <- rmse(edx_test$rating, predicted_rating))

## RMSE comparison table
rmse_table <- tibble(
  model = c("Average star rating", 
            "Average + movie", 
            "Average + movie + user",
            "Average + movie + user + genre",
            "Average + movie + user + genre + num genre",
            "Average + movie + user + genre + num genre + year",
            "Average + movie + user + genre + num genre + year + sequel",
            "Average + movie + user + genre + num genre + year + sequel + year stamp"),
  
  RMSE = c(rmse_average_guess, 
           rmse_movie_effect, 
           rmse_user_effect,
           rmse_genre_effect,
           rmse_genre_num_effect,
           rmse_release_effect,
           rmse_sequel_effect,
           rmse_year_stamp_effect
           
  ))
rmse_table %>% mutate_if(is.numeric, format, digits=6) %>% kable()

# Clean up the environment
rm(age_compared_to_average, group_compared_to_average,
   number_compared_to_average, release_compared_to_average,
   sequel_compared_to_average, stamp_year_compared_to_average,
   edx_train, edx_test, subset)

# Re-training the final model
This entails repeating all steps from above but using edx instead of edx_train. 

# Average
average <- mean(edx$rating)

# average + movie effect
movie_compared_to_average <- edx %>%
  group_by(movieId) %>%
  summarize(effect_movie = mean(rating - average))

edx <- left_join(edx, movie_compared_to_average, by = "movieId")
validation <- left_join(validation, movie_compared_to_average, by = "movieId")

# average + movie effect + user effect
user_compared_to_average <- edx %>%
  group_by(userId) %>%
  summarize(effect_user = mean(rating - average - effect_movie))

# Merge user effect with the exd set
edx <- left_join(edx, user_compared_to_average, by = "userId")
validation <- left_join(validation, user_compared_to_average, by = "userId")

# Apply helper function to get genres
edx <- get_genres(edx)
validation <- get_genres(validation)

# Create genre effects
# 1. C O M E D Y 
comedy_compared_to_average <- genre_effect(edx, sym("is_comedy"))
edx <- left_join(edx, comedy_compared_to_average, by = "is_comedy")
validation <- left_join(validation, comedy_compared_to_average, by = "is_comedy")

# 2. R O M A N C E
romance_compared_to_average <- genre_effect(edx, sym("is_romance"))
edx <- left_join(edx, romance_compared_to_average, by = "is_romance")
validation <- left_join(validation, romance_compared_to_average, by = "is_romance")

# 3. A C T I O N
action_compared_to_average <- genre_effect(edx, sym("is_action"))
edx <- left_join(edx, action_compared_to_average, by = "is_action")
validation <- left_join(validation, action_compared_to_average, by = "is_action")

# 4. C R I M E
crime_compared_to_average <- genre_effect(edx, sym("is_crime"))
edx <- left_join(edx, crime_compared_to_average, by = "is_crime")
validation <- left_join(validation, crime_compared_to_average, by = "is_crime")

# 5. T H R I L L E R 
thriller_compared_to_average <- genre_effect(edx, sym("is_thriller"))
edx <- left_join(edx, thriller_compared_to_average, by = "is_thriller")
validation <- left_join(validation, thriller_compared_to_average, by = "is_thriller")

# 6. D R A M A 
drama_compared_to_average <- genre_effect(edx, sym("is_drama"))
edx <- left_join(edx, drama_compared_to_average, by = "is_drama")
validation <- left_join(validation, drama_compared_to_average, by = "is_drama")

# 7. S C I - F I  
scifi_compared_to_average <- genre_effect(edx, sym("is_scifi"))
edx <- left_join(edx, scifi_compared_to_average, by = "is_scifi")
validation <- left_join(validation, scifi_compared_to_average, by = "is_scifi")

# 8. A D V E N T U R E 
adventure_compared_to_average <- genre_effect(edx, sym("is_adventure"))
edx <- left_join(edx, adventure_compared_to_average, by = "is_adventure")
validation <- left_join(validation, adventure_compared_to_average, by = "is_adventure")

# 9. C H I L D R E N
children_compared_to_average <- genre_effect(edx, sym("is_children"))
edx <- left_join(edx, children_compared_to_average, by = "is_children")
validation <- left_join(validation, children_compared_to_average, by = "is_children")

# 10. F A N T A S Y 
fantasy_compared_to_average <- genre_effect(edx, sym("is_fantasy"))
edx <- left_join(edx, fantasy_compared_to_average, by = "is_fantasy")
validation <- left_join(validation, fantasy_compared_to_average, by = "is_fantasy")

# 11. W A R 
war_compared_to_average <- genre_effect(edx, sym("is_war"))
edx <- left_join(edx, war_compared_to_average, by = "is_war")
validation <- left_join(validation, war_compared_to_average, by = "is_war")

# 12. A N I M A T I O N
animation_compared_to_average <- genre_effect(edx, sym("is_animation"))
edx <- left_join(edx, animation_compared_to_average, by = "is_animation")
validation <- left_join(validation, animation_compared_to_average, by = "is_animation")

# 13. M U S I C A L  
musical_compared_to_average <- genre_effect(edx, sym("is_musical"))
edx <- left_join(edx, musical_compared_to_average, by = "is_musical")
validation <- left_join(validation, musical_compared_to_average, by = "is_musical")

# 14. W E S T E R N
western_compared_to_average <- genre_effect(edx, sym("is_western"))
edx <- left_join(edx, western_compared_to_average, by = "is_western")
validation <- left_join(validation, western_compared_to_average, by = "is_western")

# 15. M Y S T E R Y
mystery_compared_to_average <- genre_effect(edx, sym("is_mystery"))
edx <- left_join(edx, mystery_compared_to_average, by = "is_mystery")
validation <- left_join(validation, mystery_compared_to_average, by = "is_mystery")

# 16. F I L M - N O I R
filmnoir_compared_to_average <- genre_effect(edx, sym("is_filmnoir"))
edx <- left_join(edx, filmnoir_compared_to_average, by = "is_filmnoir")
validation <- left_join(validation, filmnoir_compared_to_average, by = "is_filmnoir")

# 17. H O R R O R
horror_compared_to_average <- genre_effect(edx, sym("is_horror"))
edx <- left_join(edx, horror_compared_to_average, by = "is_horror")
validation <- left_join(validation, horror_compared_to_average, by = "is_horror")

# 18. D O C U M E N T A R Y 
documentary_compared_to_average <- genre_effect(edx, sym("is_documentary"))
edx <- left_join(edx, documentary_compared_to_average, by = "is_documentary")
validation <- left_join(validation, documentary_compared_to_average, by = "is_documentary")

# 19. I M A X 
imax_compared_to_average <- genre_effect(edx, sym("is_imax"))
edx <- left_join(edx, imax_compared_to_average, by = "is_imax")
validation <- left_join(validation, imax_compared_to_average, by = "is_imax")

# 20. N O   G E N R E 
nogenre_compared_to_average <- genre_effect(edx, sym("is_no_genre"))
edx <- left_join(edx, nogenre_compared_to_average, by = "is_no_genre")
validation <- left_join(validation, nogenre_compared_to_average, by = "is_no_genre")

# Clean up the environment a bit
rm(action_compared_to_average, adventure_compared_to_average,
   animation_compared_to_average, children_compared_to_average, 
   comedy_compared_to_average, crime_compared_to_average, 
   documentary_compared_to_average, drama_compared_to_average,
   fantasy_compared_to_average, filmnoir_compared_to_average, 
   horror_compared_to_average, imax_compared_to_average,
   musical_compared_to_average, mystery_compared_to_average,
   nogenre_compared_to_average, romance_compared_to_average,
   scifi_compared_to_average, thriller_compared_to_average,
   user_compared_to_average, war_compared_to_average,
   western_compared_to_average, 
   genre_list, genre_unique, movie_compared_to_average
)

# Get genre scores
edx <- get_genre_score(edx)
validation <- get_genre_score(validation)

# Divide genre_score into 10 equally-distanced groups
edx$group <- as.numeric(cut_number(edx$genre_score, 10))
validation$group <- as.numeric(cut_number(validation$genre_score, 10))

# Now find group effect
group_compared_to_average <- edx %>%
  group_by(group) %>%
  summarize(effect_group = mean(rating - average - effect_movie - effect_user))

# Merge the group effect with the train and test sets
edx <- left_join(edx, group_compared_to_average, by = "group")
validation <- left_join(validation, group_compared_to_average, by = "group")

# rating = average + movie effect + user effect + genre effect + number of genres
edx <- edx %>% mutate(num_genre = str_count(genres, pattern = "\\|") + 1)
validation <- validation %>% mutate(num_genre = str_count(genres, pattern = "\\|") + 1)

# Get genre number effect
number_compared_to_average <- edx %>%
  group_by(num_genre) %>%
  summarize(effect_num = mean(rating - average - effect_movie - effect_user - effect_group))

# Merge group effect with edx and validation
edx <- left_join(edx, number_compared_to_average, by = "num_genre")
validation <- left_join(validation, number_compared_to_average, by = "num_genre")

# Model 6: rating = average + movie effect + user effect + genre effects + release year
# Get release year
edx <- get_release_year(edx)
validation <- get_release_year(validation)

# Get grouped means
release_compared_to_average <- edx %>%
  group_by(release_year) %>%
  summarize(effect_release = mean(rating - average - effect_movie - effect_user - effect_group - effect_num))

edx <- left_join(edx, release_compared_to_average, by = "release_year")
validation <- left_join(validation, release_compared_to_average, by = "release_year")

# Model 7: Model 6 + sequel effect
edx <- edx %>% mutate(sequel = str_detect(title, pattern = ":"))
validation <- validation %>% mutate(sequel = str_detect(title, pattern = ":"))

# Get grouped means
sequel_compared_to_average <- edx %>%
  group_by(sequel) %>%
  summarize(effect_sequel = mean(rating - average - effect_movie - effect_user - effect_group - effect_num - effect_release))

# Merge the movie age effect with the train and test sets
edx <- left_join(edx, sequel_compared_to_average, by = "sequel")
validation <- left_join(validation, sequel_compared_to_average, by = "sequel")

# Model: Model 7 + rating year
# Extract timestamp years
edx <- edx %>% mutate(year_stamp = year(as_datetime(timestamp)))
validation <- validation %>% mutate(year_stamp = year(as_datetime(timestamp)))

# Get grouped means
stamp_year_compared_to_average <- edx %>%
  group_by(year_stamp) %>%
  summarize(effect_year_stamp = mean(rating - average - effect_movie - effect_user - effect_group - effect_num - effect_release - effect_sequel))

# Merge the movie age effect with the train and test sets
edx <- left_join(edx, stamp_year_compared_to_average, by = "year_stamp")
validation <- left_join(validation, stamp_year_compared_to_average, by = "year_stamp")

# Final RMSE of the re-trained model
# The model here achieves an RMSE of 0.8647926. 
predicted_rating <- (average + 
                       validation$effect_movie + 
                       validation$effect_user + 
                       validation$effect_group +
                       validation$effect_num +
                       validation$effect_release +
                       validation$effect_sequel +
                       validation$effect_year_stamp)
# RMSE
(rmse_final <- rmse(validation$rating, predicted_rating))

# Clean the environment
rm(edx, validation)
#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

### Spliting edx data set into a train and test set

set.seed(200)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
my_train <- edx[-test_index,]
test <- edx[test_index,]

# Make sure userId and movieId in validation set are also in edx set

my_test <- test %>% 
  semi_join(my_train, by = "movieId") %>%
  semi_join(my_train, by = "userId")

# Because you cannot work from the Global Environment I create a csv file for edx and validation
# to read into the RMD.
write.csv(edx,
          "edx.csv", na = "", row.names=FALSE)
write.csv(validation,
          "validation.csv", na = "", row.names=FALSE)
write.csv(my_train,
          "my_train.csv", na = "", row.names=FALSE)
write.csv(my_test,
          "my_test.csv", na = "", row.names=FALSE)

############### Script for my RMD #####################

# This code chunk simply makes sure that all the libraries used here are installed. I left this code to be seen just for purpose of submission. Otherwise, I would have echo = FALSE
packages <- c("knitr","dplyr",  "tidyr", "caret", "ggplot2")
if ( length(missing_pkgs <- setdiff(packages, rownames(installed.packages()))) > 0) {
  message("Installing missing package(s): ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs)
}


# These are the libraries needed for my report
library(knitr)
library(tidyr)
library(dplyr)
library(ggplot2)


#This chunch reads in the original data set and my training data set
file_location <- "edx.csv"
edx <- read.csv(file_location, header = TRUE)
file_location <- "my_train.csv"
my_train <- read.csv(file_location, header = TRUE)


# This chunk reads in my test set
file_location <- "my_test.csv"
my_test <- read.csv(file_location, header = TRUE)

#The dimentions of my data set (# of rows and columns)
dim(my_train)

# This chunk gives the column names
colnames(my_train)

# This code provides the number of distinct users and movies
my_train %>%
summarize(Users= n_distinct(userId),
Movies = n_distinct(movieId) )

# This code builds a histogram on a log scale the number of times movies have been rated
my_train %>% 
count(movieId) %>% 
ggplot(aes(n)) + 
geom_histogram(bins = 30, color = "skyblue") + 
scale_x_log10() + 
ggtitle("Movies")

# This code builds a histogram on a log scale the number of times a users have rated movies
my_train %>%
count(userId) %>% 
ggplot(aes(n)) + 
geom_histogram(bins = 30, color = "orange") + 
scale_x_log10() + 
ggtitle("Users")


#The following code generates the top 10 genres for the movies in our data set.
genreCount <-my_train %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10)
genreCount %>% knitr::kable()

#The following code generates the top 10 most rated the movies.
Top10 <- my_train %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% head(n=10)
Top10 %>% knitr::kable()

#The following code generates the most common rating given by the users.
ratings <- my_train %>% group_by(rating) %>% summarize(count = n()) %>% 
mutate(percent = count/nrow(my_train)*100) %>%
top_n(5) %>%
arrange(desc(count))  
ratings %>% knitr::kable()

#The following code generates a function that will calculate the RMSE for actual values (true_ratings) from our test set to their corresponding predictors from our models:
RMSE <- function(true_ratings, predicted_ratings){
sqrt(mean((true_ratings - predicted_ratings)^2))
}

#The following code creates our mu and then prints it
mu_hat <- mean(my_train$rating)
mu_hat

#We put our baseline model into our RMSE function and then print our results
baseline_rmse <- RMSE(my_test$rating, mu_hat)
baseline_rmse

#The next code shows that any other number entered will yields a worse result
predictions <- rep(4, nrow(my_test))
RMSE(my_test$rating, predictions)


#The following code produces a table in our report that will keep track of our results
rmse_results <- data_frame(method = "Baseline", RMSE = baseline_rmse)
rmse_results %>% knitr::kable()

# The folowing code calculates the movie bias
mu <- mean(my_train$rating) 
movie_avgs <- my_train %>% 
group_by(movieId) %>% 
summarize(b_i = mean(rating - mu))

#The following code creates a histogram to show the movie bias
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("skyblue"))

#This code creates our predictions
predicted_ratings <- mu_hat + my_test %>% 
left_join(movie_avgs, by='movieId') %>%
.$b_i

#This code measures how well our predictions preformed on the my_test set and then adds the RMSE to our results table
model_1_rmse <- RMSE(predicted_ratings, my_test$rating)
rmse_results <- bind_rows(rmse_results,
data_frame(method="Movie Effect Model",  
RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()


#Movie and User Effects Model
# This code calculates our user bias and then produces a histogram to show the user bias
my_train %>% 
group_by(userId) %>% 
summarize(b_u = mean(rating)) %>% 
filter(n()>=100) %>%
ggplot(aes(b_u)) + 
geom_histogram(bins = 30, color = "orange")

# The folowing code calculates the user bias
user_avgs <- my_train %>% 
left_join(movie_avgs, by='movieId') %>%
group_by(userId) %>%
summarize(b_u = mean(rating - mu_hat - b_i))

# This code will calculate our predictions when considering the user and movie bias
predicted_ratings <- my_test %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
mutate(pred = mu_hat + b_i + b_u) %>%
.$pred

#This code measures how well our predictions preformed on the my_test set and then adds the RMSE to our results table
model_2_rmse <- RMSE(predicted_ratings, my_test$rating)
rmse_results <- bind_rows(rmse_results,
data_frame(method="Movie + User Effects Model",  
RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()


# Regularization
#This will produce our largest errors our predictions made on our my_test set
my_test %>% 
left_join(movie_avgs, by='movieId') %>%
mutate(residual = rating - (mu_hat + b_i)) %>%
arrange(desc(abs(residual))) %>% 
select(title,  residual) %>% slice(1:10) %>% knitr::kable()

#This code will remove all the movie duplications 
movie_titles <- my_train %>% 
select(movieId, title) %>%
distinct()

#The following code shows the 10 best movies according to our model
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
arrange(desc(b_i)) %>% 
select(title, b_i) %>% 
slice(1:10) %>%  
knitr::kable()

#The following code shows the 10 worst movies according to our model.
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
arrange(b_i) %>% 
select(title, b_i) %>% 
slice(1:10) %>%  
knitr::kable()

#The following counts how many times the best movies have been rated. 
my_train %>% count(movieId) %>% 
left_join(movie_avgs) %>%
left_join(movie_titles, by="movieId") %>%
arrange(desc(b_i)) %>% 
select(title, b_i, n) %>% 
slice(1:10) %>% 
knitr::kable()

#The following counts how many times the worst movies have been rated. 
my_train %>% count(movieId) %>%
left_join(movie_avgs) %>%
left_join(movie_titles, by="movieId") %>%
arrange(b_i) %>% 
select(title, b_i, n) %>% 
slice(1:10) %>% 
knitr::kable()

# The following adds a penalty lambda (in this case 2.5) to movies that have been rated few times
lambda <- 2.5
mu <- mean(my_train$rating)
movie_reg_avgs <- my_train %>% 
group_by(movieId) %>% 
summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#The next code creates a graph that shows how the estimates shrink with the penalty
data_frame(original = movie_avgs$b_i, 
regularlized = movie_reg_avgs$b_i, 
n = movie_reg_avgs$n_i) %>%
ggplot(aes(original, regularlized, size=sqrt(n))) + 
geom_point(shape=1, alpha=0.5)

#This code now shows the top 10 movies after regularization 
my_train %>%
count(movieId) %>% 
left_join(movie_reg_avgs) %>%
left_join(movie_titles, by="movieId") %>%
arrange(desc(b_i)) %>% 
select(title, b_i, n) %>% 
slice(1:10) %>% 
knitr::kable()

#This code shows the 10 worst rated movies after regularization
my_train %>%
count(movieId) %>% 
left_join(movie_reg_avgs) %>%
left_join(movie_titles, by="movieId") %>%
arrange(b_i) %>% 
select(title, b_i, n) %>% 
slice(1:10) %>% 
knitr::kable()

#This code creates our predictions for movie bias with Regularization
predicted_ratings <- my_test%>% 
left_join(movie_reg_avgs, by='movieId') %>%
mutate(pred = mu_hat + b_i) %>%
.$pred

#This code measures how well our predictions preformed on the my_test set and then adds the RMSE to our results table
model_3_rmse <- RMSE(predicted_ratings, my_test$rating)
rmse_results <- bind_rows(rmse_results,
data_frame(method="Regularized Movie Effect Model",  
RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

#The following code shows how use cross validation to choose lambda. 

#We will find the RMSE for different penalties (from 0 to 10 at a sequence of .25) applied to the 
#movie bias. We then can use the lambda that will produce the lowest RMSE.
lambdas <- seq(0, 10, 0.25)
#This uses the sapply to join the lambdas with the following function which applies lambda to 
# the user and movie bias. Then it returns the result of how our predictions compared to my_test


mu <- mean(my_train$rating)
just_the_sum <- my_train %>% 
group_by(movieId) %>% 
summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
predicted_ratings <- my_test %>% 
left_join(just_the_sum, by='movieId') %>% 
mutate(b_i = s/(n_i+l)) %>%
mutate(pred = mu + b_i) %>%
.$pred
return(RMSE(predicted_ratings, my_test$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# We will repeat the step above and apply it to the user and movie bias to find the lambda that 
#gives us the smallest RMSE.

lambdas <- seq(0, 10, 0.25)
#This uses the sapply to join the lambdas with the following function which applies lambda to 
# the user and movie bias. Then it returns the result of how our predictions compared to our test set
rmses <- sapply(lambdas, function(l){

mu <- mean(my_train$rating)

b_i <- my_train %>% 
group_by(movieId) %>%
summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- my_train %>% 
left_join(b_i, by="movieId") %>%
group_by(userId) %>%
summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- 
  my_test %>% 
left_join(b_i, by = "movieId") %>%
left_join(b_u, by = "userId") %>%
mutate(pred = mu + b_i + b_u) %>%
.$pred

return(RMSE(predicted_ratings, my_test$rating))
})

qplot(lambdas, rmses) 

# This code tells us the lambda for the lowest RMSE
lambda <- lambdas[which.min(rmses)]
lambda

#This code measures how well our predictions preformed on the my_test set and then adds the RMSE to our results table
rmse_results <- bind_rows(rmse_results,
data_frame(method="Regularized Movie + User Effect Model",  
RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#The End
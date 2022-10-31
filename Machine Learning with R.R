#SuperVised Learning in R: Classification

# Load the 'class' package
library(class)

# Create a vector of labels
sign_types <- signs$sign_type

# Classify the next sign observed
knn(train = signs[-1], test = next_sign, cl =sign_types)

# Examine the structure of the signs dataset
str(signs)

# Count the number of signs of each type
table(signs$sign_type)

# Check r10's average red level by sign type
aggregate(r10 ~ sign_type, data = signs, mean)

# Use kNN to identify the test road signs
sign_types <- signs$sign_type
signs_pred <- knn(train = signs[-1], test = test_signs[-1], cl =sign_types)

# Create a confusion matrix of the predicted versus actual values
signs_actual <- test_signs$sign_type
table(signs_pred,signs_actual)

# Compute the accuracy
mean(signs_actual ==signs_pred)

## Smaller K has a very small neighbourhood (but can identify subtle patterns) whereas bigger K could also point to wrong info ( but can overcome noise patterns) as in case of tie between two a random one is chosen.
##Square root of number of observations in training data

# Compute the accuracy of the baseline model (default k = 1)
k_1 <- knn(train = signs[-1], test = signs_test[-1], cl =sign_types)
mean(signs_actual ==k_1)

# Modify the above to set k = 7
k_7 <- knn(train = signs[-1], test = signs_test[-1], cl =sign_types, k=7)
mean(signs_actual ==k_7)

# Set k = 15 and compare to the above
k_15 <- knn(train = signs[-1], test = signs_test[-1], cl =sign_types, k=15)
mean(signs_actual ==k_15)

#Seeing how the neighbors voted
#When multiple nearest neighbors hold a vote, it can sometimes be useful to examine whether the voters were unanimous or widely separated.
#For example, knowing more about the voters' confidence in the classification could allow an autonomous vehicle to use caution in the case there is any chance at all that a stop sign is ahead.
#In this exercise, you will learn how to obtain the voting results from the knn() function.The class package has already been loaded in your workspace along with the datasets signs, sign_types, and signs_test

# Use the prob parameter to get the proportion of votes for the winning class
sign_pred <- knn(train = signs[-1], test = signs_test[-1], cl =sign_types, k=7,prob=TRUE)

# Get the "prob" attribute from the predicted classes
sign_prob <- attr(sign_pred, "prob")

# Examine the first several predictions
head(sign_pred)

# Examine the proportion of votes for the winning class
head(sign_prob)


#Rescaling reduces the influence of extreme values on kNN's distance function.
x<-mtcars$mpg
normalize_1<-function(x){return ((x-min(x))/(max(x)-min(x))}


# Compute P(A) 
p_A <- p_A<-nrow(subset(where9am, location == "office"))/nrow(where9am)

# Compute P(B)
p_B <-nrow(subset(where9am, daytype == "weekday"))/nrow(where9am)

# Compute the observed P(A and B)
p_AB <-nrow(subset(where9am, location == "office", daytype == "weekday"))/nrow(where9am)

# Compute P(A | B) and print its value
p_A_given_B <-(p_AB)/(p_B)
p_A_given_B


# Load the naivebayes package
library(naivebayes)
# Build the location prediction model
locmodel <- naive_bayes(location~daytype, data =where9am)

# Predict Thursday's 9am location
predict(locmodel,data.frame(daytype=thursday9am))

# Predict Saturdays's 9am location
predict(locmodel,data.frame(daytype=saturday9am))

#Examining "raw" probabilities                                 

##The naivebayes package offers several ways to peek inside a Naive Bayes model.Typing the name of the model object provides the a priori (overall) and conditional probabilities of each of the model's predictors. If one were so inclined, you might use these for calculating posterior (predicted) probabilities by hand.
##Alternatively, R will compute the posterior probabilities for you if the type = "prob" parameter is supplied to the predict() function.
##Using these methods, examine how the model's predicted 9am location probability varies from day-to-day. The model locmodel that you fit in the previous exercise is in your workspace.

# The 'naivebayes' package is loaded into the workspace
# and the Naive Bayes 'locmodel' has been built

# Examine the location prediction model
locmodel

# Obtain the predicted probabilities for Thursday at 9am
predict(locmodel,data.frame(daytype=thursday9am), type = "prob")

# Obtain the predicted probabilities for Saturday at 9am
predict(locmodel,data.frame(daytype=saturday9am), type = "prob")

## Naive base algo uses a shortcut to calculate the conditional probability by using the events to be independent.
##The joint probability of independent events can be computed much more simply by multiplying their individual probabilities.
## Multiplies intersection of individual probabilities to calculate joint probability
##Suppose you have a chain of predictors under the naive model and suppose that one of those events has never been
## observed previously in combination with the outcome and thus the joint probability of these events is 0.
## correction would be to add 1 to each individual joint probabilities~ Laplace correction.

#A more sophisticated location model
##The locations dataset records Brett's location every hour for 13 weeks. Each hour, the tracking information includes the daytype (weekend or weekday) as well as the hourtype (morning, afternoon, evening, or night).
##Using this data, build a more sophisticated model to see how Brett's predicted location not only varies by the day of week but also by the time of day. The dataset locations is already loaded in your workspace.


# The 'naivebayes' package is loaded into the workspace already

# Build a NB model of location
locmodel <-naive_bayes(location~daytype + hourtype, data =locations)

# Predict Brett's location on a weekday afternoon
predict(locmodel,data.frame(weekday_afternoon))

# Predict Brett's location on a weekday evening
predict(locmodel,data.frame(weekday_evening))


#Preparing for unforeseen circumstances
##While Brett was tracking his location over 13 weeks, he never went into the office during the weekend. Consequently, the joint probability of P(office and weekend) = 0. 
##Explore how this impacts the predicted probability that Brett may go to work on the weekend in the future. Additionally, you can see how using the Laplace correction will allow a small chance for these types of unforeseen circumstances.
##The model locmodel is already in your workspace, along with the dataframe weekend_afternoon




# The 'naivebayes' package is loaded into the workspace already
# The Naive Bayes location model (locmodel) has already been built

# Observe the predicted probabilities for a weekend afternoon
predict(locmodel,data.frame(weekend_afternoon), type = "prob")

# Build a new model using the Laplace correction
locmodel2 <- naive_bayes(location~daytype + hourtype, data =locations, laplace =1)

# Observe the new predicted probabilities for a weekend afternoon
predict(locmodel2,data.frame(weekend_afternoon),type = "prob")


#Building simple logistic regression models
##The donors dataset contains 93,462 examples of people mailed in a fundraising solicitation for paralyzed military veterans. The donated column is 1 if the person made a donation in response to the mailing and 0 otherwise. This binary outcome will be the dependent variable for the logistic regression model.
##The remaining columns are features of the prospective donors that may influence their donation behavior. These are the model's independent variables.
##When building a regression model, it is often helpful to form a hypothesis about which independent variables will be predictive of the dependent variable. The bad_address column, which is set to 1 for an invalid mailing address and 0 otherwise, seems like it might reduce the chances of a donation. Similarly, one might suspect that religious interest (interest_religion) and interest in veterans affairs (interest_veterans) would be associated with greater charitable giving.
##In this exercise, you will use these three factors to create a simple model of donation behavior. The dataset donors is available in your workspace.



# Examine the dataset to identify potential independent variables
str(donors)

# Explore the dependent variable
table(donors$donated)

# Build the donation model
donation_model <- glm(donated~bad_address+interest_religion+interest_veterans, 
                      data = donors, family = "binomial")

# Summarize the model results
summary(donation_model)



#Making a binary prediction
##In the previous exercise, you used the glm() function to build a logistic regression model of donor behavior. As with many of R's machine learning methods, you can apply the predict() function to the model object to forecast future behavior. By default, predict() outputs predictions in terms of log odds unless type = "response" is specified. This converts the log odds to probabilities.
##Because a logistic regression model estimates the probability of the outcome, it is up to you to determine the threshold at which the probability implies action. One must balance the extremes of being too cautious versus being too aggressive. For example, if you were to solicit only the people with a 99% or greater donation probability, you may miss out on many people with lower estimated probabilities that still choose to donate. This balance is particularly important to consider for severely imbalanced outcomes, such as in this dataset where donations are relatively rare.
##The dataset donors and the model donation_model are already loaded in your workspace.


# Estimate the donation probability
donors$donation_prob <- predict(donation_model, type = "response")

# Find the donation probability of the average prospect
mean(donors$donated)

# Predict a donation if probability of donation is greater than average (0.0504)
donors$donation_pred <- ifelse(donors$donation_prob > 0.0504,1,0)

# Calculate the model's accuracy
mean(donors$donation_pred == donors$donated)


#Calculating ROC Curves and AUC
##The previous exercises have demonstrated that accuracy is a very misleading measure of model performance on imbalanced datasets. Graphing the model's performance better illustrates the tradeoff between a model that is overly agressive and one that is overly passive.
##In this exercise you will create a ROC curve and compute the area under the curve (AUC) to evaluate the logistic regression model of donations you built earlier. 
##The dataset donors with the column of predicted probabilities, donation_prob ,is already loaded in your workspace.
##When AUC values are very close, it's important to know more about how the model will be used.

# Load the pROC package
library(pROC)

# Create a ROC curve
ROC <- roc(donors$donated,donors$donation_prob)

# Plot the ROC curve
plot(ROC, col = "blue")

# Calculate the area under the curve (AUC)
auc(ROC)



#Coding categorical features
##Sometimes a dataset contains numeric values that represent a categorical feature.
##In the donors dataset, wealth_rating uses numbers to indicate the donor's wealth level:
##0 = Unknown 1 = Low 2 = Medium 3 = High
##This exercise illustrates how to prepare this type of categorical feature and the examines its impact on a logistic regression model. The dataframe donors is loaded in your workspace.







# Convert the wealth rating to a factor
donors$wealth_rating <- factor(donors$wealth_rating, levels= c(0, 1, 2,3), labels=c("Unknown", "Low", "Medium", "High"))
# Use relevel() to change reference category
donors$wealth_rating <-relevel(donors$wealth_rating, ref ="Medium")
donation_model1<-glm(donated~wealth_rating, 
                     data = donors, family = "binomial")
# See how our factor coding impacts the model
summary(donation_model1)







#Handling missing data
##Some of the prospective donors have missing age data. Unfortunately, R will exclude any cases with NA values when building a regression model.
##One workaround is to replace, or impute, the missing values with an estimated value. After doing so, you may also create a missing data indicator to model the possibility that cases with missing data are different in some way from those without.
##The dataframe donors is loaded in your workspace.

# Find the average age among non-missing values
summary(donors$age)

# Impute missing age values with the mean age
donors$imputed_age <- ifelse(is.na(donors$age),round(mean(donors$age, na.rm=TRUE),2),donors$age)

# Create missing value indicator for age
donors$missing_age <- ifelse(is.na(donors$age),1,0)


#Building a more sophisticated model
##One of the best predictors of future giving is a history of recent, frequent, and large gifts. In marketing terms, this is known as R/F/M:
##  Recency Frequency Money
##Donors that haven't given both recently and frequently may be especially likely to give again; in other words, the combined impact of recency and frequency may be greater than the sum of the separate effects.
##Because these predictors together have a greater impact on the dependent variable, their joint effect must be modeled as an interaction. The donors dataset has been loaded for you.

# Build a recency, frequency, and money (RFM) model

rfm_model <-glm(donated~money+recency*frequency, data= donors, family="binomial") 

# Summarize the RFM model to see how the parameters were coded
summary(rfm_model)

# Compute predicted probabilities for the RFM model
rfm_prob <- predict(rfm_model, data= donors, type= "response")

# Plot the ROC curve and find AUC for the new model
library(pROC)
ROC <- roc(donors$donated,rfm_prob)
plot(ROC, col = "red")
auc(ROC)


#Building a stepwise regression model
##In the absence of subject-matter expertise, stepwise regression can assist with the search for the most important predictors of the outcome of interest.
##In this exercise, you will use a forward stepwise approach to add predictors to the model one-by-one until no additional benefit is seen. The donors dataset has been loaded for you.


# Specify a null model with no predictors
null_model <- glm(formula= donated ~ 1, data = donors, family = "binomial")

# Specify the full model using all of the potential predictors
full_model <- glm(donated~., data = donors, family = "binomial")


# Use a forward stepwise algorithm to build a parsimonious model
step_model <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "forward")

# Estimate the stepwise donation probability
step_prob <- predict(step_model,data= donors, type="response")

# Plot the ROC of the stepwise model
library(pROC)
ROC <- roc(donors$donated,step_prob)
plot(ROC, col = "red")
auc(ROC)


#Building a simple decision tree
##The loans dataset contains 11,312 randomly-selected people who applied for and later received loans from Lending Club, a US-based peer-to-peer lending company. 
##You will use a decision tree to try to learn patterns in the outcome of these loans (either repaid or default) based on the requested loan amount and credit score at the time of application.
##Then, see how the tree's predictions differ for an applicant with good credit versus one with bad credit. 
##The dataset loans is already in your workspace.

# Load the rpart package
library(rpart)

# Build a lending model predicting loan outcome versus loan amount and credit score
loan_model <- rpart(outcome~loan_amount+ credit_score, data =loans,method = "class",control = rpart.control(cp = 0))

# Make a prediction for someone with good credit
predict(loan_model, data.frame(good_credit),type = "class")

# Make a prediction for someone with bad credit
predict(loan_model, data.frame(bad_credit),type = "class")



#Visualizing classification trees
##Due to government rules to prevent illegal discrimination, lenders are required to explain why a loan application was rejected.
##The structure of classification trees can be depicted visually, which helps to understand how the tree makes its decisions. The model loan_model that you fit in the last exercise is in your workspace.


# Examine the loan_model object
loan_model

# Load the rpart.plot package
library(rpart.plot)

# Plot the loan_model with default settings
rpart.plot(loan_model)

# Plot the loan_model with customized settings
rpart.plot(loan_model, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)



#Divide-and-conquer always looks to create the split resulting in the greatest improvement to purity
Creating random test datasets
Before building a more sophisticated lending model, it is important to hold out a portion of the loan data to simulate how well it will predict the outcomes of future loan applicants.
As depicted in the following image, you can use 75% of the observations for training and 25% for testing the model. 

The sample() function can be used to generate a random sample of rows to include in the training set. Simply supply it the total number of observations and the number needed for training.
Use the resulting vector of row IDs to subset the loans into training and testing datasets. The dataset loans is loaded in your workspace.




# Determine the number of rows for training
nrow(loans)*0.75


# Create a random sample of row IDs
sample_rows <- sample(nrow(loans), nrow(loans)*0.75)

# Create the training dataset
loans_train <- loans[sample_rows,]

# Create the test dataset
loans_test <- loans[-sample_rows,]

#Building and evaluating a larger tree
##Previously, you created a simple decision tree that used the applicant's credit score and requested loan amount to predict the loan outcome.
##Lending Club has additional information about the applicants, such as home ownership status, length of employment, loan purpose, and past bankruptcies, that may be useful for making more accurate predictions.
##Using all of the available applicant data, build a more sophisticated lending model using the random training dataset created previously. Then, use this model to make predictions on the testing dataset to estimate the performance of the model on future loan applications.
##The rpart package is loaded into the workspace and the loans_train and loans_test datasets have been created.



# Grow a tree using all of the available applicant data
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class", control = rpart.control(cp = 0))

# Make predictions on the test dataset
loans_test$pred <- predict(loan_model, loans_test, type = "class")

# Examine the confusion matrix
table(loans_test$pred, loans_test$outcome)

# Compute the accuracy on the test dataset
mean(loans_test$pred == loans_test$outcome)


#Due to overfitting, training set accuracy often exaggerates the tree's true performance.
#The model's accuracy is affected by the rarity of the outcome.
Preventing overgrown trees
The tree grown on the full set of applicant data grew to be extremely large and extremely complex, with hundreds of splits and leaf nodes containing only a handful of applicants. This tree would be almost impossible for a loan officer to interpret.
Using the pre-pruning methods for early stopping, you can prevent a tree from growing too large and complex. See how the rpart control options for maximum tree depth and minimum split count impact the resulting tree.
rpart is loaded.

# Grow a tree with maxdepth of 6
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class", control = rpart.control(cp = 0, maxdepth = 6))

# Make a class prediction on the test set
loans_test$pred <- predict(loan_model, loans_test, type = "class")

# Swap maxdepth for a minimum split of 500 
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class", control = rpart.control(cp = 0,minsplit= 500))

# Run this. How does the accuracy change?
loans_test$pred <- predict(loan_model, loans_test, type = "class")
mean(loans_test$pred == loans_test$outcome)


#Creating a nicely pruned tree
##Stopping a tree from growing all the way can lead it to ignore some aspects of the data or miss important trends it may have discovered later. 
##By using post-pruning, you can intentionally grow a large and complex tree then prune it to be smaller and more efficient later on. 
##In this exercise, you will have the opportunity to construct a visualization of the tree's performance versus complexity, and use this information to prune the tree to an appropriate level.
##The rpart package is loaded into the workspace, along with loans_test and loans_train

# Grow an overly complex tree
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class", control = rpart.control(cp = 0))


# Examine the complexity plot
plotcp(loan_model)

# Prune the tree
loan_model_pruned <- prune(loan_model, cp = 0.0014)

# Compute the accuracy of the pruned tree
loans_test$pred <- predict(loan_model_pruned, loans_test, type = "class")
mean(loans_test$pred==loans_test$outcome)

# Compute the accuracy of the simpler tree
mean(loans_test$pred == loans_test$outcome)

Building a random forest model
In spite of the fact that a forest can contain hundreds of trees, growing a decision tree forest is perhaps even easier than creating a single highly-tuned tree.
Using the randomForest package, build a random forest and see how it compares to the single trees you built previously. 
Keep in mind that due to the random nature of the forest, the results may vary slightly each time you create the forest.

# Load the randomForest package
library(randomForest)

# Build a random forest model
loan_model <- randomForest(outcome~., data = loans_train, ntree=500)

# Compute the accuracy of the random forest
loans_test$pred <- predict(loan_model, loans_test, type = "class")
mean(loans_test$pred == loans_test$outcome)
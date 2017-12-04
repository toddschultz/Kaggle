# Source packages
library("randomForest")

# Log loss function for scoring
llfun <- function(actual, prediction) {
  epsilon <- .000000000000001
  yhat <- pmin(pmax(prediction, epsilon), 1-epsilon)
  logloss <- -mean(actual*log(yhat)
                   + (1-actual)*log(1 - yhat))
  return(logloss)
}



# Import 500k sample data

# Training data
train <- read.csv('train_500k.csv')
ytrain <- factor(train$click)
d <- strptime(train$hour,format="%y%m%d%H")
train$h <- as.integer(strftime(d, '%H'))
train$wd <- factor(weekdays(d),levels = c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"),ordered=TRUE)

# Validation data
val <- read.csv('val_500k.csv')
yval <- factor(val$click)
d <- strptime(val$hour,format="%y%m%d%H")
val$h <- as.integer(strftime(d, '%H'))
val$wd <- factor(weekdays(d),levels = c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"),ordered=TRUE)

# Test data
#test <- read.csv('test.csv',numerals = "no.loss",as.is="id")
test <- read.csv('test.csv',colClasses = c("character","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL","NULL"))
idtest <- test$id
test <- read.csv('test.csv')
d <- strptime(test$hour,format="%y%m%d%H")
test$h <- as.integer(strftime(d, '%H'))
test$wd <- factor(weekdays(d),levels = c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"),ordered=TRUE)


# Keep only banner position, site domain, site category, app domain, app category, device conn type, C14, C16, hour of the day, and day of the week

# Training data
train$id <- NULL
train$click <- NULL
train$hour <- NULL
train$C1 <- NULL
train$site_id <- NULL
train$app_id <- NULL
train$device_id <- NULL
train$device_ip <- NULL
train$device_model <- NULL
#train$device_type <- NULL
train$C15 <- NULL
train$C17 <- NULL
train$C18 <- NULL
train$C19 <- NULL
train$C20 <- NULL
train$C21 <- NULL

# Validation data
val$id <- NULL
val$click <- NULL
val$hour <- NULL
val$C1 <- NULL
val$site_id <- NULL
val$app_id <- NULL
val$device_id <- NULL
val$device_ip <- NULL
val$device_model <- NULL
#val$device_type <- NULL
val$C15 <- NULL
val$C17 <- NULL
val$C18 <- NULL
val$C19 <- NULL
val$C20 <- NULL
val$C21 <- NULL

# Test data
test$id <- NULL
test$click <- NULL
test$hour <- NULL
test$C1 <- NULL
test$site_id <- NULL
test$app_id <- NULL
test$device_id <- NULL
test$device_ip <- NULL
test$device_model <- NULL
#test$device_type <- NULL
test$C15 <- NULL
test$C17 <- NULL
test$C18 <- NULL
test$C19 <- NULL
test$C20 <- NULL
test$C21 <- NULL


# Make factors
# Make factors for banner_pos, site_domain, site_category, app_domain, app_category,
# device_type, device_conn_type

clevels <- c(unique(train$banner_pos),unique(val$banner_pos),unique(test$banner_pos))
clevels <- unique(clevels)
train$banner_pos <- factor(x = train$banner_pos, levels = clevels)
val$banner_pos <- factor(x = val$banner_pos, levels = clevels)
test$banner_pos <- factor(x = test$banner_pos, levels = clevels)

clevels <- c(levels(train$site_domain),levels(val$site_domain),levels(test$site_domain))
clevels <- unique(clevels)
train$site_domain <- factor(x = train$site_domain, levels = clevels) # labels = clevels
val$site_domain <- factor(x = val$site_domain, levels = clevels)
test$site_domain <- factor(x = test$site_domain, levels = clevels)

clevels <- c(levels(train$site_category),levels(val$site_category),levels(test$site_category))
clevels <- unique(clevels)
train$site_category <- factor(x = train$site_category, levels = clevels) # labels = clevels
val$site_category <- factor(x = val$site_category, levels = clevels)
test$site_category <- factor(x = test$site_category, levels = clevels)

clevels <- c(levels(train$app_domain),levels(val$app_domain),levels(test$app_domain))
clevels <- unique(clevels)
train$app_domain <- factor(x = train$app_domain, levels = clevels) # labels = clevels
val$app_domain <- factor(x = val$app_domain, levels = clevels)
test$app_domain <- factor(x = test$app_domain, levels = clevels)

clevels <- c(levels(train$app_category),levels(val$app_category),levels(test$app_category))
clevels <- unique(clevels)
train$app_category <- factor(x = train$app_category, levels = clevels) # labels = clevels
val$app_category <- factor(x = val$app_category, levels = clevels)
test$app_category <- factor(x = test$app_category, levels = clevels)

clevels <- c(unique(train$device_type),unique(val$device_type),unique(test$device_type))
clevels <- unique(clevels)
train$device_type <- factor(x = train$device_type, levels = clevels) # labels = clevels
val$device_type <- factor(x = val$device_type, levels = clevels)
test$device_type <- factor(x = test$device_type, levels = clevels)

clevels <- c(unique(train$device_conn_type),unique(val$device_conn_type),unique(test$device_conn_type))
clevels <- unique(clevels)
train$device_conn_type <- factor(x = train$device_conn_type, levels = clevels) # labels = clevels
val$device_conn_type <- factor(x = val$device_conn_type, levels = clevels)
test$device_conn_type <- factor(x = test$device_conn_type, levels = clevels)

# Clean up variable space
remove('d','clevels')

# Prepare variables for logistic regression
trainR <- train
trainR$click <- ytrain

valR <- val
valR$click <- yval


# Prepare variables for random forest algorithm
#Remove variable with more than 53 categories
trainA <- train
valA <- val
testA <- test

# site domain 2301 categories
trainA$site_domain <- NULL
valA$site_domain <- NULL
testA$site_domain <- NULL

# app domain 159 categories
trainA$app_domain <- NULL
valA$app_domain <- NULL
testA$app_domain <- NULL



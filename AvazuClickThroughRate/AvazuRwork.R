# Source packages
library("randomForest")

# Import 500k sample data

# Training data
train <- read.csv('train_500k.csv')
ytrain <- factor(train$click)
date <- strptime(train$hour,format="%y%m%d%H")
train$h <- as.integer(strftime(date, '%H'))
train$wd <- factor(weekdays(date))

# Validation data
val <- read.csv('val_500k.csv')
yval <- factor(val$click)
date <- strptime(val$hour,format="%y%m%d%H")
val$h <- as.integer(strftime(date, '%H'))
val$wd <- factor(weekdays(date))

# Test data
test <- read.csv('test.csv')
idtest <- test$id
date <- strptime(test$hour,format="%y%m%d%H")
test$h <- as.integer(strftime(date, '%H'))
test$wd <- factor(weekdays(date))

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
train$device_type <- NULL
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
val$device_type <- NULL
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
test$device_type <- NULL
test$C15 <- NULL
test$C17 <- NULL
test$C18 <- NULL
test$C19 <- NULL
test$C20 <- NULL
test$C21 <- NULL

# Make factors
# Make factors for banner_pos, site_domain, site_category, app_domain, app_category,
# device_type, device_conn_type




# Clean up variable space
remove('df','C14','C16','app_category','app_domain','site_category','site_domain','device_type')
remove('device_conn_type','h','wd','date','banner_pos')

# Build first model with train dataframe
rfmodel <- randomForest(train, y = ytrain, xtest = val, ytest = yval, ntree = 100,
             replace=FALSE, nodesize = 10, maxnodes = 20,
             importance = TRUE, norm.votes = TRUE, do.trace = TRUE,
             keep.forest = TRUE, keep.inbag=FALSE)

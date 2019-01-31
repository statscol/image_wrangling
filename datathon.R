####DATATHON OIL_PALM IMAGE PREDICTION FOR KAGGLE
#### BY JHON PARRA

###############################################MXNET################################################################
####################################################################################################################

#################################################GPU VERSION########################################################

cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
options(repos = cran)
install.packages("mxnet")


############################################CPU VERSION#############################################################

install.packages("colorspace")
cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet",dependencies = T)

######################################################################################################################
######################################################################################################################


library(magick)
library(mxnet)
library(EBImage)
library(imager)
library(dplyr)
library(caret)
library(ggplot2)
library(reshape2)
library(parallel)


##############################TRAINING SET##################################
labels=read.csv("traininglabels.csv",header = T,colClasses = c("character","numeric","numeric"))
directory="F:/widsdatathon2019/train_images/"  ##DIRECTORY WHERE IMAGES ARE LOCATED
ima_df=data.frame()


##################################FUNCTION TO EXTRACT ARRAY FROM IMG#######################################

resize_custom<-function(directory_file,dim_f=c(30,30,3)){
  
  ##change to grayscale and resize images to 100x100 resolution
  img=load.image(directory_file) %>% resize(size_x = dim_f[1],size_y = dim_f[2])
  m=array(img,dim = dim_f)
  
  #image as 100x100 vector 
  img_vector <- as.vector(m) 
  
  
  return(img_vector)
}

files_loc=paste(directory,labels[,1],sep="")

###PARALELL COMPUTING 

fin=mcmapply(files_loc,FUN = resize_custom,MoreArgs = list(dim_f=c(40,40,3)))

pix_array=t(fin)


# Set names. The first columns are the labels, the other columns are the pixels.

set.seed(123)
data_model=data.frame(labels$has_oilpalm,pix_array)
colnames(data_model)=c("has_oilpalm",paste("px",1:ncol(pix_array)))

##test to see if image was not affected by wrangling process

plot(Image(data_model[8,-1],dim = c(35,35,3),colormode = "Color"))

#write model data to avoid reading it again

write.table(data_model,"modeldata.csv",row.names = F)
#########################################################MODEL BUILDING########################################

index=createDataPartition(data_model$has_oilpalm,p=0.7,list = F)
train_data=data.matrix(data_model[index,])
test_data=data.matrix(data_model[-index,])

dim(train_data)

train_x=t(train_data[,-1])
train_y=train_data[,1]
train_array <- train_x
dim(train_array) <- c(40, 40, 3, ncol(train_x))

test_x <- t(test_data[, -1])
test_y <- test_data[, 1]
test_array <- test_x
dim(test_array) <- c(40, 40, 3, ncol(test_x))


###############################################NEURAL NET PARAMETERS#####################################

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 200)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)


##########################################################################################################

mx.set.seed(123)

# Device used.
devices <- mx.cpu()
devices2<-mx.gpu()

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 35,
                                     array.batch.size = 50,
                                     learning.rate = 0.1,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback =mx.callback.log.train.metric(100)
)


# Predict labels
predicted <- predict(model, test_array)

dim(predicted)

test_array[]
# Assign labels
predicted_labels <- as.factor(max.col(t(predicted))-1)

predicted
summary(predicted)

# Get accuracy
confusionMatrix(reference = as.factor(test_data[, 1]),data =  predicted_labels,positive = '1')
summary(as.factor(predicted_labels))


##################################################USING H2O###################################################

#load the package
require(h2o)

#start h2o
localH2o <- h2o.init(nthreads = -1, max_mem_size = "10G")

#load data on H2o
trainh2o <- as.h2o(train_data)
testh2o <- as.h2o(test_data)

#set variables
colnames(trainh2o)

#train the model - without hidden layer
deepmodel <- h2o.deeplearning( y = "has_oilpalm"
                              ,training_frame = trainh2o
                              ,standardize = F
                              ,model_id = "deep_model"
                              ,activation = "Rectifier"
                              ,epochs = 50
                              ,seed = 1
                              ,nfolds = 5
                              )

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,xval = T)
#84.5 % CV accuracy



deepmodel <- h2o.deeplearning(x = x      
                              ,y = y      
                              ,training_frame = trainh2o      
                              ,validation_frame = testh2o         
                              ,standardize = T        
                              ,model_id = "deep_model"        
                              ,activation = "Rectifier"       
                              ,epochs = 100       
                              ,seed = 1       
                              ,hidden = 5         
                              ,variable_importances = T)

h2o.performance(deepmodel,valid = T)
#






##################################################USING XGBOOST############################################
require(xgboost)

dtrain <- xgb.DMatrix(data = train_data[,-1], label= train_data[,1])
dtest <- xgb.DMatrix(data = test_data[,-1], label= test_data[,1])





# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 100, # the maximum depth of each decision tree
                       nround = 50, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic") # control for imbalanced classes

# generate predictions for our held-out testing data
pred <- ifelse(predict(model_tuned, dtest)>=0.5,1,0)

table(test_data[,1])

confusionMatrix(reference= as.factor(test_data[,1]),data = as.factor(pred),positive = '1')


# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

######################XGBOOST USING CARET FOR CROSS-VALIDATION AND GRID SEARCH






################################VALIDATIONS SET########################################################


labels_validation=read.csv("SampleSubmission.csv",header=T,colClasses = c("character","numeric"))



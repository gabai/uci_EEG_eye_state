library(keras)
library(magrittr)
library(mlbench)
library(dplyr)
library(neuralnet)
library(RWeka)

#Get data
temp <- tempfile()
download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff",
              temp)
data <- read.arff(temp)
unlink(temp)

#Modify data

```{python}

import numpy as np
import pandas as pd

def to_sequence(seq_size, obs):
  x = []
  y = []
  
  for i in range(len(obs)-SEQUENCE_SIZE-1):
    #print(i)
    window = obs[i:(i+SEQUENCE_SIZE)]
    after_window = obs[i+SEQUENCE_SIZE]
    window = [[x] for in window]
    x.append(window)
    y.append(after_window)
    
  return np.array(x), np.array(y)

SEQUENCE_SIZE = 10
x_train, y_train = to_sequences(SEQUENCE_SIZE, EEG_train)
x_test, y_test = to_sequences(SEQUENCE_SIZE, EEG_test)

print("Shape of training set: {}", format(x_train.shape))
print("Shape of test set: {}", format(x_test.shape))
  
```


data <- as.matrix(data)
dimnames(data) <- NULL

data[, 1:14] <- normalize(data[, 1:14])
data[, 15] <- as.numeric(data[, 9])

#Set random seed
set.seed(7)

#Split data 70/30 for trainng and testing
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind == 1, 1:14]
test <- data[ind == 2, 1:14]
trainingtarget <- data[ind == 1, 15]
testtarget <- data[ind == 2, 15]

trainlabel <- to_categorical(trainingtarget)
testlabel <- to_categorical(testtarget)

#Model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = )
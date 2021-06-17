################################ Training: Find the best configuration across CNN layers.
################### 1-layer CNN

## Generate 1-layer CNN configuration table for 5min epoch data w/ HR (D-5min-w.HR).
source("generate_config_table.R")

# metric table should have the same shape as hyperparameter table
# col name: val_accuracy, val_recall, val_precision, val_AUC, compute_cost
metric_tb = tibble(configuration=1:nrow(hyper),
                   val_accuracy=0,val_recall=0,
                   val_precision=0,val_AUC=0,
                   compute_cost=0)
# Set values for the hyperparameters of the 1-layer CNN model
for (i in 1:nrow(hyper)){
  # Building the CNN model
  # flags {tfruns}
  FLAGS <- flags(
    # seed for cnn
    flag_integer("seed",hyper$seed[i]),
    # m
    flag_integer("m", hyper$m[i]),
    # l
    flag_integer("l", hyper$l[i]),
    # stepsize
    flag_integer("step_size", hyper$step_size[i]),
    # hr indicator
    flag_boolean("hr", hyper$hr[i]),
    
    # kernel size: h*w
    flag_integer("kernel_size_h", hyper$kernel_size_h[i]),
    flag_integer("kernel_size_w", hyper$kernel_size_w[i]),
    # number of kernels
    flag_integer("num_kernel", hyper$num_kernel[i]),
    # pool size h*w
    flag_integer("pool_size_h", hyper$pool_size_h[i]),
    flag_integer("pool_size_w", hyper$pool_size_w[i]),
    # fully-connected layer size
    flag_integer("fcl_size", hyper$fcl_size[i]),
    # fraction of the units to drop for the linear transformation of the inputs
    flag_numeric("dropout", hyper$dropout[i]),
    # optimizer
    flag_string("optimizer", hyper$optimizer[i]),
    # learning rate
    flag_numeric("lr", hyper$learning_rate[i]),
    # training batch size
    flag_integer("batch_size", hyper$batch_size[i]),
    # num_epoch
    flag_integer("num_epoch", hyper$num_epoch[i]),
    # parameter to the early stopping callback
    flag_integer("patience", 15)
  )
  ##### reshape data
  
  # IF THE SAME AS LAST ONE, NOT CHANGE!!!!!!
  gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m,hr=FLAGS$hr)
  
  dt_train=gen_data$dt_train
  dt_val=gen_data$dt_val
  dt_test=gen_data$dt_test
  
  train_x = dt_train$dt_x
  train_y = data.matrix(dt_train$dt_y)
  
  val_x = dt_val$dt_x
  val_y =  data.matrix(dt_val$dt_y)
  
  # for test subject i in 1:4
  test_x_lst = list()
  test_y_lst = list()
  for (j in 1:4){
    test_x_lst = c(test_x_lst,list(dt_test[[j]]$dt_x))
    test_y_lst = c(test_y_lst,list(data.matrix(dt_test[[j]]$dt_y)))
  }
  str(test_x_lst)
  str(test_y_lst)
  
  #checking the dimentions
  dim(train_x) 
  cat("No of training samples\t",dim(train_x)[[1]],
      "\tNo of validation samples\t",dim(val_x)[[1]])
  
  
  ############# 1-layer CNN Training 
  library(tensorflow)
  library(keras)
  # in order to get reproducible result, each time should run set seet first!
  set.seed(FLAGS$seed)
  tensorflow::tf$random$set_seed(FLAGS$seed)
  #use_session_with_seed(890,disable_gpu = F,disable_parallel_cpu = F)
  
  model1 <-keras_model_sequential()
  #configuring the Model
  model1 %>% layer_conv_2d(filters=FLAGS$num_kernel,
                           kernel_size=c(FLAGS$kernel_size_h,FLAGS$kernel_size_w),
                           padding="same",                
                           input_shape=c(FLAGS$l,FLAGS$m,1) ) %>%   
    layer_activation("relu") %>%   # keep unchanged
    layer_max_pooling_2d(pool_size=c(FLAGS$pool_size_h,
                                     FLAGS$pool_size_w)) %>%  # change for 5min to 3x2
    layer_dropout(FLAGS$dropout) %>% #dropout layer to avoid overfitting
    layer_flatten() %>%  #flatten the input  
    layer_dense(FLAGS$fcl_size) %>% # units: dimensionality of the output space.
    layer_activation("relu") %>%  
    layer_dense(1) %>% 
    layer_activation("sigmoid") 
  opt<-optimizer_adam(lr=FLAGS$lr)# , decay = 1e-6 ) 
  model1 %>% compile(loss="binary_crossentropy",
                     optimizer=FLAGS$optimizer,
                     metrics = list("AUC","accuracy","Recall","Precision"))
  summary(model1)
  
  starttime=Sys.time()
  history1 = model1 %>% fit( train_x,train_y ,batch_size=FLAGS$batch_size,
                             validation_data = list(val_x, val_y),
                             callbacks=callback_early_stopping(monitor = "val_accuracy",
                                                               min_delta = 0,
                                                               patience = FLAGS$patience),
                             epochs=FLAGS$num_epoch)    
  endtime=Sys.time()
  metric_tb$compute_cost[i] = difftime(endtime, starttime, units = "secs")[[1]]
  maxepoch = length(history1$metrics$val_accuracy)
  metric_tb$val_accuracy[i] = history1$metrics$val_accuracy[maxepoch]
  metric_tb$val_AUC[i] = history1$metrics$val_AUC[maxepoch]
  metric_tb$val_recall[i] = history1$metrics$val_Recall[maxepoch]
  metric_tb$val_precision[i] = history1$metrics$val_Precision[maxepoch]
}

metric_tb %>% write_csv(paste0("metric_",epoch_type[epo],".csv"))  

######### delete later
# hyper
# metric_tb = read_csv(paste0("metric_",epoch_type[epo],".csv"))
# metric_tb
######### delete later

######################################################################
######################################################################
######################################################################

################### 2-layer CNN
## Generate 2-layer CNN configuration table for 5min epoch data w/ HR (D-5min-w.HR).
source("generate_config_table_layer2.R")
hyper2

# metric table should have the same shape as hyperparameter table
# col name: val_accuracy, val_recall, val_precision, val_AUC, compute_cost
metric_tb2 = tibble(configuration=1:nrow(hyper2),
                   val_accuracy=0,val_recall=0,
                   val_precision=0,val_AUC=0,
                   compute_cost=0)


for (i in 1:nrow(hyper2)){
  # flags {tfruns}
  FLAGS <- flags(
    # seed for cnn
    flag_integer("seed",hyper2$seed[i]),
    # m
    flag_integer("m", hyper2$m[i]),
    # l
    flag_integer("l", hyper2$l[i]),
    # stepsize
    flag_integer("step_size", hyper2$step_size[i]),
    # hr indicator
    flag_boolean("hr", hyper2$hr[i]),
    
    ##### layer1
    # kernel size: h*w
    flag_integer("kernel_size_h1", hyper2$kernel_size_h1[i]),
    flag_integer("kernel_size_w1", hyper2$kernel_size_w1[i]),
    # number of kernels
    flag_integer("num_kernel1", hyper2$num_kernel1[i]),
    # pool size h*w
    flag_integer("pool_size_h1", hyper2$pool_size_h1[i]),
    flag_integer("pool_size_w1", hyper2$pool_size_w1[i]),
    # dropout: fraction of the units to drop for the linear transformation of the inputs
    flag_numeric("dropout1", hyper2$dropout1[i]),
    
    ##### layer2
    # kernel size: h*w
    flag_integer("kernel_size_h2", hyper2$kernel_size_h2[i]),
    flag_integer("kernel_size_w2", hyper2$kernel_size_w2[i]),
    # number of kernels
    flag_integer("num_kernel2", hyper2$num_kernel2[i]),
    # pool size h*w
    flag_integer("pool_size_h2", hyper2$pool_size_h2[i]),
    flag_integer("pool_size_w2", hyper2$pool_size_w2[i]),
    # dropout
    flag_numeric("dropout2", hyper2$dropout2[i]),
    
    #### output layer
    # fully-connected layer size
    flag_integer("fcl_size", hyper2$fcl_size[i]),
    # optimizer
    flag_string("optimizer", hyper2$optimizer[i]),
    # learning rate
    flag_numeric("lr", hyper2$learning_rate[i]),
    # training batch size
    flag_integer("batch_size", hyper2$batch_size[i]),
    # num_epoch
    flag_integer("num_epoch", hyper2$num_epoch[i]),
    # parameter to the early stopping callback
    flag_integer("patience", 15)
  )
  ##### reshape data
  gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m,hr=FLAGS$hr)
  
  dt_train=gen_data$dt_train
  dt_val=gen_data$dt_val
  dt_test=gen_data$dt_test
  
  train_x = dt_train$dt_x
  train_y = data.matrix(dt_train$dt_y)
  
  val_x = dt_val$dt_x
  val_y =  data.matrix(dt_val$dt_y)
  
  # for test subject i in 1:4
  test_x_lst = list()
  test_y_lst = list()
  for (j in 1:4){
    test_x_lst = c(test_x_lst,list(dt_test[[j]]$dt_x))
    test_y_lst = c(test_y_lst,list(data.matrix(dt_test[[j]]$dt_y)))
  }
  str(test_x_lst)
  str(test_y_lst)
  
  #checking the dimentions
  dim(train_x) 
  cat("No of training samples\t",dim(train_x)[[1]],
      "\tNo of validation samples\t",dim(val_x)[[1]])
  
  
  ############# CNN Training 
  library(tensorflow)
  library(keras)
  # in order to get reproducible result, each time should run set seet first!
  set.seed(FLAGS$seed)
  tensorflow::tf$random$set_seed(FLAGS$seed)
  #use_session_with_seed(890,disable_gpu = F,disable_parallel_cpu = F)
  
  model1 <-keras_model_sequential()
  #configuring the Model
  model1 %>% 
    #### layer1
    layer_conv_2d(filters=FLAGS$num_kernel1,
                  kernel_size=c(FLAGS$kernel_size_h1,FLAGS$kernel_size_w1),
                  padding="same",                
                  input_shape=c(FLAGS$l,FLAGS$m,1) ) %>%   
    layer_activation("relu") %>%   # keep unchanged
    layer_max_pooling_2d(pool_size=c(FLAGS$pool_size_h1,FLAGS$pool_size_w1)) %>%  
    layer_dropout(FLAGS$dropout1) %>% #dropout layer to avoid overfitting
    
    #### layer2
    layer_conv_2d(filter=FLAGS$num_kernel2,
                  kernel_size=c(FLAGS$kernel_size_h2,FLAGS$kernel_size_w2),
                  padding="same") %>%  
    layer_activation("relu") %>%  
    layer_max_pooling_2d(pool_size=c(FLAGS$pool_size_h2,FLAGS$pool_size_w2)) %>%  
    layer_dropout(FLAGS$dropout2) %>%
    
    #### output
    layer_flatten() %>%  #flatten the input  
    layer_dense(FLAGS$fcl_size) %>% # units: dimensionality of the output space.
    layer_activation("relu") %>%  
    layer_dense(1) %>% 
    layer_activation("sigmoid") 
  opt<-optimizer_adam(lr=FLAGS$lr)    
  model1 %>% compile(loss="binary_crossentropy",
                     optimizer=FLAGS$optimizer,
                     metrics = list("AUC","accuracy","Recall","Precision"))
  summary(model1)
  
  starttime=Sys.time()
  history1 = model1 %>% fit( train_x,train_y ,batch_size=FLAGS$batch_size,
                             validation_data = list(val_x, val_y),
                             callbacks=callback_early_stopping(monitor = "val_accuracy",
                                                               min_delta = 0,
                                                               patience = FLAGS$patience),
                             epochs=FLAGS$num_epoch)    
  endtime=Sys.time()
  
  metric_tb2$compute_cost[i] = difftime(endtime, starttime, units = "secs")[[1]]
  maxepoch = length(history1$metrics$val_accuracy)
  metric_tb2$val_accuracy[i] = history1$metrics$val_accuracy[maxepoch]
  metric_tb2$val_AUC[i] = history1$metrics$val_AUC[maxepoch]
  metric_tb2$val_recall[i] = history1$metrics$val_Recall[maxepoch]
  metric_tb2$val_precision[i] = history1$metrics$val_Precision[maxepoch]
}

metric_tb2 %>% write_csv(paste0("metric_",epoch_type[epo],"_layer2_",".csv")) 




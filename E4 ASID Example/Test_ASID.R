####################### Test process of the ASID Workflow
################################################### build the best cnn 
i = 1  # best config_idx, no need to specify, since hyper_star only has one row

hyper_star[i,]
# flags {tfruns}

if (hyper_star$conv_layer=="1 layer"){
  FLAGS <- flags(
    # m
    flag_integer("m", hyper_star$m[i]),
    # l
    flag_integer("l", hyper_star$l[i]),
    # stepsize
    flag_integer("step_size", hyper_star$step_size[i]),
    # hr indicator
    flag_boolean("hr", hyper_star$hr[i]),
    
    # kernel size: h*w
    flag_integer("kernel_size_h", hyper_star$kernel_size_h1[i]),
    flag_integer("kernel_size_w", hyper_star$kernel_size_w1[i]),
    # number of kernels
    flag_integer("num_kernel", hyper_star$num_kernel1[i]),
    # pool size h*w
    flag_integer("pool_size_h", hyper_star$pool_size_h1[i]),
    flag_integer("pool_size_w", hyper_star$pool_size_w1[i]),
    # fully-connected layer size
    flag_integer("fcl_size", hyper_star$fcl_size[i]),
    # fraction of the units to drop for the linear transformation of the inputs
    flag_numeric("dropout", hyper_star$dropout1[i]),
    # optimizer
    flag_string("optimizer", hyper_star$optimizer[i]),
    # learning rate
    flag_numeric("lr", hyper_star$learning_rate[i]),
    # training batch size
    flag_integer("batch_size", hyper_star$batch_size[i]),
    # num_epoch
    flag_integer("num_epoch", hyper_star$num_epoch[i]),
    # parameter to the early stopping callback
    flag_integer("patience", 15)
  )
}else{
  FLAGS <- flags(
    # m
    flag_integer("m", hyper_star$m[i]),
    # l
    flag_integer("l", hyper_star$l[i]),
    # stepsize
    flag_integer("step_size", hyper_star$step_size[i]),
    # hr indicator
    flag_boolean("hr", hyper_star$hr[i]),
    
    ##### layer1
    # kernel size: h*w
    flag_integer("kernel_size_h1", hyper_star$kernel_size_h1[i]),
    flag_integer("kernel_size_w1", hyper_star$kernel_size_w1[i]),
    # number of kernels
    flag_integer("num_kernel1", hyper_star$num_kernel1[i]),
    # pool size h*w
    flag_integer("pool_size_h1", hyper_star$pool_size_h1[i]),
    flag_integer("pool_size_w1", hyper_star$pool_size_w1[i]),
    # dropout: fraction of the units to drop for the linear transformation of the inputs
    flag_numeric("dropout1", hyper_star$dropout1[i]),
    
    ##### layer2
    # kernel size: h*w
    flag_integer("kernel_size_h2", hyper_star$kernel_size_h2[i]),
    flag_integer("kernel_size_w2", hyper_star$kernel_size_w2[i]),
    # number of kernels
    flag_integer("num_kernel2", hyper_star$num_kernel2[i]),
    # pool size h*w
    flag_integer("pool_size_h2", hyper_star$pool_size_h2[i]),
    flag_integer("pool_size_w2", hyper_star$pool_size_w2[i]),
    # dropout
    flag_numeric("dropout2", hyper_star$dropout2[i]),
    
    #### output layer
    # fully-connected layer size
    flag_integer("fcl_size", hyper_star$fcl_size[i]),
    # optimizer
    flag_string("optimizer", hyper_star$optimizer[i]),
    # learning rate
    flag_numeric("lr", hyper_star$learning_rate[i]),
    # training batch size
    flag_integer("batch_size", hyper_star$batch_size[i]),
    # num_epoch
    flag_integer("num_epoch", hyper_star$num_epoch[i]),
    # parameter to the early stopping callback
    flag_integer("patience", 15)
  )
}

# reshape data under the final best configuration j*
gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m,hr=FLAGS$hr)

dt_train=gen_data$dt_train
dt_val=gen_data$dt_val
dt_test=gen_data$dt_test
dt_train_val=gen_data$dt_train_val

train_x = dt_train$dt_x
train_y = data.matrix(dt_train$dt_y)

val_x = dt_val$dt_x
val_y =  data.matrix(dt_val$dt_y)

# for test subject i in 1:4
test_x_lst = list()
test_y_lst = list()
test_time_lst = list()  # get test time for image plot later.
len = rep(0,4)  # get total size of each subject's data
for (j in 1:4){
  test_x_lst = c(test_x_lst,list(dt_test[[j]]$dt_x))
  test_y_lst = c(test_y_lst,list(data.matrix(dt_test[[j]]$dt_y)))
  # get time
  test_time_lst = c(test_time_lst,list(dt_test[[j]]$dt_time))
  
  len[j] = nrow(test_y_lst[[j]])
}
len
str(test_x_lst)
str(test_y_lst)
str(test_time_lst)

test_time=unlist(test_time_lst)
str(test_time)

train_val_x_lst = list()
train_val_y_lst = list()
len_train_val=rep(0,20)
for (j in 1:20){
  train_val_x_lst = c(train_val_x_lst,list(dt_train_val[[j]]$dt_x))
  train_val_y_lst = c(train_val_y_lst,list(data.matrix(dt_train_val[[j]]$dt_y)))
  len_train_val[j] = nrow(train_val_y_lst[[j]])
}


#checking the dimentions
dim(train_x) 
cat("No of training samples\t",dim(train_x)[[1]],
    "\tNo of validation samples\t",dim(val_x)[[1]])


########################## Test

# Different seeds should be used to get an unbiased evaluation of the model's performance
## 10 different seeds were used in original experiment.
seed = 178  

if (hyper_star$conv_layer=="1 layer"){
  library(tensorflow)
  library(keras)
  # in order to get reproducible result, each time should run set seet first!
  set.seed(seed)
  tensorflow::tf$random$set_seed(seed)
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
}else{
  library(tensorflow)
  library(keras)
  # in order to get reproducible result, each time should run set seet first!
  set.seed(seed)
  tensorflow::tf$random$set_seed(seed)
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
}


starttime=Sys.time()
history1 = model1 %>% fit( train_x,train_y ,batch_size=FLAGS$batch_size,
                           validation_data = list(val_x, val_y),
                           callbacks=callback_early_stopping(monitor = "val_accuracy",
                                                             min_delta = 0,
                                                             patience = FLAGS$patience),
                           epochs=FLAGS$num_epoch)


########### tune the window size of Majority voting on the training set
all_x=abind::abind(train_x,val_x,along = 1)
all_y=abind::abind(train_y,val_y,along = 1)
pred_train = model1 %>% predict(all_x,all_y, batch_size = 5)

cum_len=c(0,cumsum(len_train_val))
diff=NULL
for(i in 1:50){
  before=NULL
  after=NULL
  for (j in 1:20){
    before=c(before,weighted_acc(as.integer(pred_train[(cum_len[j]+1):cum_len[j+1]]>0.5),all_y[(cum_len[j]+1):cum_len[j+1]]))
    mv_pred=majority_vote(pred_train[(cum_len[j]+1):cum_len[j+1]],w_size=i)
    after=c(after,weighted_acc(mv_pred,all_y[(cum_len[j]+1):cum_len[j+1]]))
  }
  diff = c(diff, mean(after-before))
}
win_size=which.max(diff)
#win_size

# pre majority vote
test_acc1 = rep(0,4)  
test_AUC = rep(0,4)
test_recall = rep(0,4)
test_precision = rep(0,4)
test_specificity = rep(0,4)
test_acc_weighted = rep(0,4)

# post majority vote
mv_test_acc1=rep(0,4)
test_recall_mv = rep(0,4)
test_precision_mv = rep(0,4)
test_specificity_mv = rep(0,4)
weighted_test_acc1_mv = rep(0,4) 


for (i in 1:4){
  subject_index = idx_test[i]
  test_x=test_x_lst[[i]]
  test_y=test_y_lst[[i]]
  result = evaluate(model1,test_x_lst[[i]],test_y_lst[[i]],batch_size = 5)
  result = as.list(result)  
  print(result)
  # metrics pre majority voting
  test_acc1[i] = result$accuracy
  test_AUC[i] = result$AUC
  test_recall[i] = result$Recall
  test_precision[i] = result$Precision
 
  # get real prediction for later plot use
  pred = model1 %>% predict(test_x_lst[[i]],test_y_lst[[i]], batch_size = 5)
  pred_binary = rep(0,length(pred)) # covnert pred to binary 
  pred_binary[pred>.5] =1
  conf_matrix <- table(pred_binary, test_y) # create confusion matrix pre majority vote
  mv_pred=majority_vote(pred,w_size=win_size) # perform majority voting
  conf_matrix_mv<-table(mv_pred,test_y) # create confusion matrix post majority vote
  
  # specificity pre majoirty voting
  test_specificity[i] = specificity(conf_matrix, negative = "0")

  # weighted accuracy pre majority voting
  test_acc_weighted[i] = weighted_acc(pred_binary, test_y) # weighted accuracy
  
  # metrics post majority voting
  mv_test_acc1[i]=mean(mv_pred==test_y) # accuracy 
  test_specificity_mv[i] = specificity(conf_matrix_mv, negative = "0") # specificity
  test_recall_mv[i] = sensitivity(conf_matrix_mv, positive = "1") # sensitivity
  test_precision_mv[i] = precision(conf_matrix_mv, relevant = "1") # precision
  weighted_test_acc1_mv[i] = weighted_acc(mv_pred,test_y) # weighted accuracy
}

endtime=Sys.time()
comp_cost1 = as.numeric(difftime(endtime,starttime,units="secs"))



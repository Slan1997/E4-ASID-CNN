################################ Training: Find the best configuration across ASID Workflow w/ different number of CNN layers
################### ASID Workflow w/ 1-layer CNN

## Generate ASID Workflow w/ 1-layer CNN configuration table 
source("generate_config_table.R")

# get hyperparameter configuration 
hyper = read_csv(paste0(hyperpara_path,"hyperparameter_table_L1_align_full_ver.csv"))  %>% 
  rename(row_idx=`...1`) %>% #first column is configuration index 
  # simplify the tunning process for this D-5min-HR example
  filter(seed_split==890,hr==T,epoch_type=='5min',m==18)
hyper
dim(hyper)

# Create metric table to store results.
# column name: val_accuracy, val_recall, val_precision, val_AUC, compute_cost
metric_tb = tibble(configuration=1:nrow(hyper), # metric table should have the same number of rows as hyperparameter table
                   val_accuracy=0,val_recall=0,
                   val_precision=0,val_AUC=0,
                   compute_cost=0)

# Set values for the hyperparameters of the 1-layer CNN model
for (i in 1:nrow(hyper)){
  # Building the LSTM model
  # flags {tfruns}
  FLAGS <- flags(
    flag_string("modality",hyper$modal[i]),
    flag_string("var_lst",hyper$var_lst[i]),
    
    flag_string("epoch",hyper$epoch_type[i]),
    # seed for train/test split and cnn
    flag_integer("seed",hyper$seed_split[i]),
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
  #### read in data
  epoch_ty = which(epoch_type==hyper$epoch_type[i])
  # time = read_csv(paste0(epoch_type[epo],"_permute.csv"))  %>%  ##col_types = cols(calender_time=col_datetime())) %>%
  #   dplyr::select(unix_sec) # already in the global environment
  
  var_lst = c("e4_id", unlist(str_split(FLAGS$var_lst,',')),"sleep")
  modal = unlist(str_split(FLAGS$modality,','))

  dt_ready = read_csv(paste0(epoch_type[epo],"_permute_norm.csv")) %>%
    dplyr::select(all_of(var_lst)) %>% 
    dplyr::select(e4_id,starts_with(modal),sleep) %>% 
    mutate_at(c("hourofday","age"),funs(scale)) %>%
    bind_cols(time)  
  
  #### get train/test split [In this example, this part is fixed and thus be omitted.]
  ###create train, validation and test set
  # student index for train, validation and test
  # idx_files=1:25
  # set.seed(FLAGS$seed)
  # idx_all = sample(c(1,3:25),24)
  # idx_train = idx_all[1:15]   ## 15 files for training
  # idx_val = idx_all[16:20]    ## 5 for validation
  # idx_train_val=idx_all[1:20] ## 20 for training and validation
  # idx_test = sort(idx_all[21:24])   ## 4 for testing
  #print(idx_train)
  #print(idx_val)
  
  ##### reshape data
  gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m) 
  
  dt_train=gen_data$dt_train
  dt_val=gen_data$dt_val
  dt_train_val=gen_data$dt_train_val
  
  train_x = dt_train$dt_x
  train_y = data.matrix(dt_train$dt_y)
  
  val_x = dt_val$dt_x
  val_y =  data.matrix(dt_val$dt_y)
  
  #checking the dimensions
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
  
  metric_tb$compute_cost[i] = as.numeric(difftime(endtime,starttime,units="secs"))
  
  maxepoch = length(history1$metrics$val_accuracy)
  metric_tb$val_accuracy[i] = history1$metrics$val_accuracy[maxepoch]
  metric_tb$val_AUC[i] = history1$metrics$val_auc[maxepoch]
  metric_tb$val_recall[i] = history1$metrics$val_recall[maxepoch]
  metric_tb$val_precision[i] = history1$metrics$val_precision[maxepoch]
}

metric_tb %>% write_csv(paste0(folder_path,"/sample_result/align_metric.csv"))
  #write_csv(paste0("Models/metrics/align_metric","_CNN1L",r1,"_",r2,"full_ver.csv")) 

# if run all the hyperparamter combinations on cluster, use following code to combine and summarize all the results
# model_metric_path = "~/E4_CNN_2022/30s_HR_align/Models/metrics/"
# list.files <- list.files(path = model_metric_path, "^align_metric_CNN1L[0-9]+_([0-9]+)full_ver.csv")
# idx = as.numeric(gsub('align_metric_CNN1L[0-9]+_([0-9]+)full_ver.*','\\1',list.files))
# idx
# 
# list.files=list.files[order(idx)]
# list.files
# # Initiate a blank data frame 
# combined_edacat <- tibble()
# # combine datasets
# for (i in 1:length(list.files)){
#   temp_data <- read_csv(paste0(model_metric_path,list.files[i]))
#   combined_edacat <- bind_rows(combined_edacat, temp_data)
# }
# dim(combined_edacat)
# 
# combined_edacat1=combined_edacat %>% rename(row_idx=configuration) %>% left_join(hyper_config,by="row_idx")
# combined_edacat2 = combined_edacat1 %>% group_by(epoch_type,hr,config_align) %>% summarise_at(vars(starts_with(c("val_",'compute_'))),funs(mean)) %>% ungroup
# 
# hyper1 = left_join(hyper,hyper_config%>%select(row_idx,config_align),by='row_idx') %>% group_by(epoch_type,hr,config_align) %>% 
#   filter(row_number() == 1) %>% ungroup %>% select(-seed_split,-var_lst)
# 
# train_config_metric = full_join(hyper1,combined_edacat2,by=c('epoch_type','hr','config_align'))
# train_config_metric %>% write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L1train_config_metric_full_ver.csv")) 
# 
# #best_config = tibble()
# #for (i in c(TRUE,FALSE)){
# config = train_config_metric %>% group_by(epoch_type,hr,align)  # %>% filter(hr==i) %>% group_by(epoch_type) 
# best_config_auc = config %>% arrange(epoch_type,hr,desc(val_AUC)) %>% filter(row_number()==1) %>% ungroup
# best_config_accuracy = config  %>% arrange(epoch_type,hr,desc(val_accuracy)) %>% filter(row_number()==1)%>% ungroup
# best_config_recall = config  %>% arrange(epoch_type,hr,desc(val_recall)) %>% filter(row_number()==1)%>% ungroup
# best_config_precision = config  %>% arrange(epoch_type,hr,desc(val_precision)) %>% filter(row_number()==1)%>% ungroup
# #best_config = bind_rows(best_config,best_config_auc,best_config_accuracy,best_config_recall,best_config_precision) 
# best_config = bind_rows(best_config_auc,best_config_accuracy,best_config_recall,best_config_precision) 
# #}
# best_config
# 
# train_config_metric %>% group_by(epoch_type,hr)  %>% arrange(epoch_type,hr,desc(val_AUC)) %>% filter(row_number()==1) %>%
#   write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L1train_config_bestAUC_full_ver.csv")) 
# 
# train_config_metric %>% group_by(epoch_type,hr,align)  %>% arrange(epoch_type,hr,desc(val_AUC)) %>% filter(row_number()==1) %>%
#   write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L1train_config_bestAUC_allalign_full_ver.csv")) 
# 
# best_config %>% mutate(criteria = rep(c('val_AUC','val_accuracy','val_recall','val_precision'),each=30))%>% 
#   write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L1train_config_best_4metrics_full_ver.csv")) 
# 

## above original results are saved in the "original_result/hyperparameter" folder


######################################################################
######################################################################
######################################################################

################### ASID Workflow w/ 2-layer CNN
## Generate ASID Workflow w/ 2-layer CNN configuration table 
source("generate_config_table_layer2.R")
hyper2= read_csv(paste0(hyperpara_path,"hyperparameter_table_L2_align_full_ver.csv"))  %>% 
  rename(row_idx=`...1`) %>% #first column is configuration index 
  # simplify the tunning process for this D-5min-HR example
  filter(seed_split==890,hr==T,epoch_type=='5min')
hyper2
dim(hyper2)

# col name: val_accuracy, val_recall, val_precision, val_AUC, compute_cost
metric_tb2 = tibble(configuration=1:nrow(hyper2),
                   val_accuracy=0,val_recall=0,
                   val_precision=0,val_AUC=0,
                   compute_cost=0)


for (i in 1:nrow(hyper2)){
  # flags {tfruns}
  FLAGS <- flags(
    flag_string("modality",hyper2$modal[i]),
    flag_string("var_lst",hyper2$var_lst[i]),
    flag_string("epoch",hyper2$epoch_type[i]),
    # seed for cnn
    flag_integer("seed",hyper2$seed_split[i]),
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
  #### read in data
  epoch_ty = which(epoch_type==hyper$epoch_type[i])
  # time = read_csv(paste0(epoch_type[epo],"_permute.csv"))  %>%  ##col_types = cols(calender_time=col_datetime())) %>%
  #   dplyr::select(unix_sec) # already in the global environment
  
  var_lst = c("e4_id", unlist(str_split(FLAGS$var_lst,',')),"sleep")
  modal = unlist(str_split(FLAGS$modality,','))
  
  dt_ready = read_csv(paste0(epoch_type[epo],"_permute_norm.csv")) %>%
    dplyr::select(all_of(var_lst)) %>% 
    dplyr::select(e4_id,starts_with(modal),sleep) %>% 
    mutate_at(c("hourofday","age"),funs(scale)) %>%
    bind_cols(time)  
  
  #### get train/test split [In this example, this part is fixed and thus be omitted.]
  ###create train, validation and test set
  # student index for train, validation and test
  # idx_files=1:25
  # set.seed(FLAGS$seed)
  # idx_all = sample(c(1,3:25),24)
  # idx_train = idx_all[1:15]   ## 15 files for training
  # idx_val = idx_all[16:20]    ## 5 for validation
  # idx_train_val=idx_all[1:20] ## 20 for training and validation
  # idx_test = sort(idx_all[21:24])   ## 4 for testing
  #print(idx_train)
  #print(idx_val)
  #print(idx_test)
  
  ##### reshape data
  gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m) # use hr=T here, actually var_lst already consider hr scenario
  
  dt_train=gen_data$dt_train
  dt_val=gen_data$dt_val
  dt_train_val=gen_data$dt_train_val
  
  train_x = dt_train$dt_x
  train_y = data.matrix(dt_train$dt_y)
  
  val_x = dt_val$dt_x
  val_y =  data.matrix(dt_val$dt_y)
  
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
  opt<-optimizer_adam(learning_rate=FLAGS$lr)    
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
  
  metric_tb2$compute_cost[i] = as.numeric(difftime(endtime,starttime,units="secs"))
  
  maxepoch = length(history1$metrics$val_accuracy)
  metric_tb2$val_accuracy[i] = history1$metrics$val_accuracy[maxepoch]
  metric_tb2$val_AUC[i] = history1$metrics$val_auc[maxepoch]
  metric_tb2$val_recall[i] = history1$metrics$val_recall[maxepoch]
  metric_tb$val_precision[i] = history1$metrics$val_precision[maxepoch]
}
metric_tb2 %>% write_csv(paste0(folder_path,"/sample_result/align_metric2.csv"))

# similar as in layer 1, if run all the hyperparamter combinations on cluster, 
# use following code to combine and summarize all the results

# hyper_config = read_csv(paste0(hyperpara_path,"hyperparameter_table_L2_align_full_ver.csv")) %>% rename(row_idx=X1) #first column is row index
# 
# model_metric_path = "~/E4_CNN_2022/30s_HR_align/Models/metrics/"
# list.files <- list.files(path = model_metric_path, "^align_metric_CNN2L[0-9]+_([0-9]+)full_ver.csv")
# idx = as.numeric(gsub('align_metric_CNN2L[0-9]+_([0-9]+)full_ver.csv','\\1',list.files))
# idx
# 
# 
# list.files=list.files[order(idx)]
# list.files
# # Initiate a blank data frame 
# combined_edacat <- tibble()
# # combine datasets
# for (i in 1:length(list.files)){
#   temp_data <- read_csv(paste0(model_metric_path,list.files[i]))
#   combined_edacat <- bind_rows(combined_edacat, temp_data)
# }
# dim(combined_edacat)
# 
# combined_edacat1=combined_edacat %>% rename(row_idx=configuration) %>% left_join(hyper_config,by="row_idx")
# combined_edacat2 = combined_edacat1 %>% group_by(epoch_type,hr,config_align_L2) %>% summarise_at(vars(starts_with(c("val_",'compute_'))),funs(mean)) %>% ungroup
# 
# hyper1 = hyper_config %>% group_by(epoch_type,hr,config_align_L2) %>% 
#   filter(row_number() == 1) %>% ungroup %>% select(-seed_split,-var_lst) # can also delete l here, since l is affected by seed_split
# 
# train_config_metric = full_join(hyper1,combined_edacat2,by=c('epoch_type','hr','config_align_L2'))
# #train_config_metric %>% write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_metric.csv")) 
# 
# train_config_metric %>% write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_metric_full_ver.csv")) 
# 
# config = train_config_metric %>% group_by(epoch_type,hr,align)  # %>% filter(hr==i) %>% group_by(epoch_type) 
# best_config_auc = config %>% arrange(epoch_type,hr,desc(val_AUC)) %>% filter(row_number()==1) %>% ungroup
# best_config_accuracy = config  %>% arrange(epoch_type,hr,desc(val_accuracy)) %>% filter(row_number()==1)%>% ungroup
# best_config_recall = config  %>% arrange(epoch_type,hr,desc(val_recall)) %>% filter(row_number()==1)%>% ungroup
# best_config_precision = config  %>% arrange(epoch_type,hr,desc(val_precision)) %>% filter(row_number()==1)%>% ungroup
# #best_config = bind_rows(best_config,best_config_auc,best_config_accuracy,best_config_recall,best_config_precision) 
# best_config = bind_rows(best_config_auc,best_config_accuracy,best_config_recall,best_config_precision) 
# #}
# best_config
# 
# train_config_metric %>% group_by(epoch_type,hr)  %>% arrange(epoch_type,hr,desc(val_AUC)) %>% filter(row_number()==1) %>%
#   # write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_bestAUC.csv")) 
#   write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_bestAUC_full_ver.csv")) 
# 
# train_config_metric %>% group_by(epoch_type,hr,align)  %>% arrange(epoch_type,hr,desc(val_AUC)) %>% filter(row_number()==1) %>%
#   #write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_bestAUC_allalign.csv")) 
#   write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_bestAUC_allalign_full_ver.csv")) 
# 
# best_config %>% mutate(criteria = rep(c('val_AUC','val_accuracy','val_recall','val_precision'),each=30))%>% 
#   #write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_best_4metrics.csv")) 
#   write_csv(paste0("~/E4_CNN_2022/30s_HR_align/Models/","L2train_config_best_4metrics_full_ver.csv")) 

## above original results are saved in the "original_result/hyperparameter" folder


## combine training metrics across 1 and 2 layers
L1_best = read_csv(paste0(hyperpara_path,"L1train_config_bestAUC_full_ver.csv"))
L2_best = read_csv(paste0(hyperpara_path,"L2train_config_bestAUC_full_ver.csv"))

L1_best_all = L1_best %>% transmute(row_idx,align,l,modal,epoch_type,hr,m,step_size, # don't need l here
                                    kernel_size_h1=kernel_size_h,kernel_size_w1=kernel_size_w, 
                                    kernel_shape1=kernel_shape,num_kernel1=num_kernel,
                                    pool_size_h1=pool_size_h, pool_size_w1=pool_size_w,dropout1=dropout,
                                    kernel_size_h2=NA,kernel_size_w2=NA,kernel_shape2=NA,
                                    num_kernel2=NA,pool_size_h2=NA, pool_size_w2=NA,dropout2=NA,
                                    fcl_size,optimizer,learning_rate,batch_size,num_epoch,config_align,
                                    val_accuracy,val_recall,val_precision,val_AUC,compute_cost)

L2_best_all = L2_best %>% rename(config_align=config_align_L2) %>% 
  dplyr::select(-config,-row_idx_L1)

best_comb =  bind_rows(L1_best_all%>%mutate(conv_layer="1 layer"),L2_best_all%>%mutate(conv_layer="2 layers"))
best_comb %>% as.data.frame

best_comb %>% group_by(epoch_type,hr)  %>% 
  arrange(desc(val_AUC)) %>% filter(row_number()==1)%>% ungroup %>% as.data.frame

best_comb %>% group_by(epoch_type,hr)  %>% 
  arrange(desc(val_AUC)) %>% filter(row_number()==1)%>% ungroup %>% 
  arrange(epoch_type,hr) %>%
  write_csv(paste0(hyperpara_path,"testhyper_full_ver.csv"))



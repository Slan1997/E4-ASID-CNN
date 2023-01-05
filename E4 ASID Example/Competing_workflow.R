source('path_keeper.R')
source('functions.R')
epoch_type = c("30s","1min","5min")
hr_status = c("hr","nohr")
sec = c(30,60,300)
proc_path = c(min30processed_path,min1processed_path,min5processed_path)
library(pacman)
pacman::p_load(tidyr, dplyr, readr, grid, gridExtra, scales, openair, 
               caret, randomForest,e1071,xgboost,rminer,rattle, klaR)

array_table = expand.grid(seed_split= c(890, 9818,7,155,642),epo=1:3,h=1:2)

index=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))  # index = 1 - 30
array_table[index,]

epo = array_table[index,'epo'] 
h = array_table[index,'h']
seed_split = array_table[index,'seed_split']

var_lst = c("e4_id","acc_mean","acc_sd","acc_q1","acc_q2","acc_q3",
            "temp_mean","temp_sd","temp_q1","temp_q2","temp_q3",
            "hr_mean","hr_sd","hr_q1","hr_q2","hr_q3",
            "hourofday","eda_cat","age","sex","sleep")

# remove hr related var
if (h==2){
  var_lst = var_lst[-grep('^hr',var_lst)]
}

dt_time <-read_csv(paste0(proc_path[epo],sec[epo],"_ready.csv")) %>% 
  dplyr::select(c("e4_id","unix_sec"))
time = dt_time %>% dplyr::select(unix_sec)

dt_norm <-read_csv(paste0(proc_path[epo],sec[epo],"_ready_norm.csv")) %>%
  dplyr::select(all_of(var_lst)) %>% mutate_at(c("sex","eda_cat","sleep"),funs(as.factor)) %>%
  mutate_at(c("hourofday","age"),funs(scale)) %>% bind_cols(time)  %>%
  filter(!(e4_id==22&unix_sec>1582262785&unix_sec<1582316887)) %>% dplyr::select(-unix_sec)# handle subject 22 
dt_norm

dt_time = dt_time %>% filter(!(e4_id==22&unix_sec>1582262785&unix_sec<1582316887)) # handle subject 22 
dt_time

tag=read_csv(paste0(tag_path,"/sleep_tag.csv"))

dt_dum = dt_norm %>% dplyr::select(ncol(dt_norm),1:(ncol(dt_norm)-1)) 

#### use different split seed by system array
# student index for train, validation and test
idx_files=1:25
# seed_split = 890
set.seed(seed_split)
idx_all = sample(c(1,3:25),24)
idx_train_val=idx_all[1:20] ## 20 for training and validation
idx_test = sort(idx_all[21:24])   ## 5 for testing
idx_test

# split data for train_val and test
dt_train =  dt_dum %>% filter(e4_id %in% idx_train_val) # check the order of e4_id
dt_test = dt_dum %>% filter(e4_id %in% idx_test)

# create folds which will be used in cross-validation 
group_folds =  groupKFold(dt_train$e4_id, k = 10)  #GroupKFold is a variation of k-fold which ensures that the same group is not represented in both testing and training sets.
TrainingParameters = trainControl(index=group_folds,
                                  method = "cv")
# model formula
f = DF2formula(dt_dum[,-2]) # remove e4_id


############# logistic regression
starttime=Sys.time()
LGModel = train(f, data = dt_train,method = "glm",family = "binomial",
                trControl = TrainingParameters, na.action = na.omit)
endtime=Sys.time()
comp_train_LG = as.numeric(difftime(endtime,starttime,units='secs'))

# best model and  parameter 
LGModel$finalModel
# save the model to disk
saveRDS(LGModel, paste0(test_result_tm,"saved_best_models/", epoch_type[epo],"_",hr_status[h],"_split",seed_split,"_LG_model.rds"))

############# SVM linear
starttime=Sys.time()
SVML = train(f, data = dt_train, method = "svmLinear",
             trControl= TrainingParameters,tuneLength = 10, na.action = na.omit)
endtime=Sys.time()
comp_train_svml = as.numeric(difftime(endtime,starttime,units='secs'))
#Best model and parameter
SVML$finalModel
SVML$bestTune
saveRDS(SVML, paste0(test_result_tm,"saved_best_models/", epoch_type[epo],"_",hr_status[h],"_split",seed_split,"_SVML_model.rds"))

############## SVM Radial
grid = expand.grid(sigma = c(0.01, 0.1, 0.5), C = c(1,5,10,100))
starttime=Sys.time()
SVMR = train(f, data = dt_train,method = "svmRadial",
             trControl= TrainingParameters,tuneGrid = grid, na.action = na.omit)
endtime=Sys.time()
comp_train_svmr = as.numeric(difftime(endtime,starttime,units='secs'))
#Best model and parameter
SVMR$finalModel
SVMR$bestTune
saveRDS(SVMR, paste0(test_result_tm,"saved_best_models/", epoch_type[epo],"_",hr_status[h],"_split",seed_split,"_SVMR_model.rds"))

############## KNN
starttime=Sys.time()
KNN_fit = train(f, data = dt_train,method = "knn",trControl= TrainingParameters,
                tuneLength = 10, na.action = na.omit)
endtime=Sys.time()
comp_train_knn =as.numeric(difftime(endtime,starttime,units='secs'))
#Best model and parameter
KNN_fit$finalModel
KNN_fit$bestTune
saveRDS(KNN_fit, paste0(test_result_tm,"saved_best_models/", epoch_type[epo],"_",hr_status[h],"_split",seed_split,"_KNN_model.rds"))

############## Random Forest
grid = expand.grid(mtry = c(2:6))
set.seed(seed_split)
starttime=Sys.time()
RFModel = train(sleep~., data = dt_train,method = "rf",
                trControl= TrainingParameters,
                tuneGrid = grid, na.action = na.omit)
endtime=Sys.time()
comp_train_rf =as.numeric(difftime(endtime,starttime,units='secs'))
saveRDS(RFModel, paste0(test_result_tm,"saved_best_models/", epoch_type[epo],"_",hr_status[h],"_split",seed_split,"_rf_model.rds"))


############################# generate output table for train/val predicted values
train_out0 = dt_time %>% filter(e4_id %in% idx_train_val) 
train_out0 

train_out = train_out0 %>% mutate(seed_split_val=rep(seed_split,nrow(train_out0))) # add seed for train/test split.
pred_train = matrix(NA,nrow=nrow(train_out),ncol=10) #  columns for 5 methods  + 5 comp_cost
colnames(pred_train) = c("lg","svm_l","svm_r","knn","rf","comp_lg","comp_svml","comp_svmr","comp_knn","comp_rf")
train_output = cbind(train_out,pred_train)
# predict train/val for mv
train_output['lg'] = predict(LGModel, dt_train)
train_output['svm_l'] = predict(SVML, dt_train)
train_output['svm_r'] = predict(SVMR, dt_train)
train_output['knn'] = predict(KNN_fit, dt_train)
train_output['rf'] = predict(RFModel, dt_train)
train_output['comp_lg'] = rep(comp_train_LG,nrow(train_out))
train_output['comp_svml'] = rep(comp_train_svml,nrow(train_out))
train_output['comp_svmr'] =rep(comp_train_svmr,nrow(train_out))
train_output['comp_knn'] = rep(comp_train_knn,nrow(train_out))
train_output['comp_rf'] = rep(comp_train_rf,nrow(train_out))

train_output %>% write_csv(paste0(test_result_tm,"pred_values/",'train_',index,'.csv'))
# test_result_tm: "/home/lanshi/E4_CNN_2022/Models/traditional_models/"

############################# generate final table for test results
test_out0 = dt_time %>% filter(e4_id %in% idx_test) 
test_out = test_out0 %>% mutate(seed_split_val=rep(seed_split,nrow(test_out0))) # add seed for train/test split.
pred_test = matrix(NA,nrow=nrow(test_out),ncol=10) #  columns for 4 methods + 4 comp_cost
colnames(pred_test) = c("lg","svm_l","svm_r","knn","rf","comp_lg","comp_svml","comp_svmr","comp_knn","comp_rf")
test_output = cbind(test_out,pred_test)

for(i in 1:length(idx_test)){
  row_idx = test_output$e4_id==idx_test[i]
  subject_test = dt_test[row_idx,] 
  
  # logistic regression
  starttime=Sys.time()
  test_output[row_idx,'lg'] = predict(LGModel, subject_test)
  endtime=Sys.time()
  test_output[row_idx,'comp_lg'] = as.numeric(difftime(endtime,starttime,units='secs'))
  
  # svm linear
  starttime=Sys.time()
  test_output[row_idx,'svm_l'] = predict(SVML, subject_test)
  endtime=Sys.time()
  test_output[row_idx,'comp_svml'] = as.numeric(difftime(endtime,starttime,units='secs'))
  
  # svm radial
  starttime=Sys.time()
  test_output[row_idx,'svm_r'] = predict(SVMR, subject_test)
  endtime=Sys.time()
  test_output[row_idx,'comp_svmr'] = as.numeric(difftime(endtime,starttime,units='secs'))
  
  # KNN
  starttime=Sys.time()
  test_output[row_idx,'knn'] = predict(KNN_fit, subject_test)
  endtime=Sys.time()
  test_output[row_idx,'comp_knn'] = as.numeric(difftime(endtime,starttime,units='secs'))
  
  # RF
  starttime=Sys.time()
  test_output[row_idx,'rf'] = predict(RFModel, subject_test)
  endtime=Sys.time()
  test_output[row_idx,'comp_rf'] = as.numeric(difftime(endtime,starttime,units='secs'))
}

# save: test e4_id unix_sec predict_value  comp_cost
test_output %>% write_csv(paste0(test_result_tm,"pred_values/",'test_',index,'.csv'))



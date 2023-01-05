source('~/E4_CNN_2022/Rcodes/path_keeper.R')
source('~/E4_CNN_2022/Rcodes/functions.R')
epoch_type = c("30s","1min","5min")
sec = c(30,60,300)

library(pacman)
pacman::p_load(tidyr, dplyr, readr,tfruns,rappdirs,scales,magrittr,zoo,MLmetrics,stringr)
proc_path = c(min30processed_path,min1processed_path,min5processed_path)

array_table = expand.grid(seed_split= c(890, 9818,7,155,642),epo=1:3,h=1:2) 


# transition period different combinations
shorter_periods = c(30,60)
longer_periods = c(60,90)
periods = expand.grid(shorter=shorter_periods,longer=longer_periods)


index=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))  # index = 1 - 30
array_table[index,]

epoch_ty = array_table[index,'epo']  # # 1 for 30s, 2 for 1min, 3 for 5min
epoch1 = epoch_type[epoch_ty]
epoch1 
h = array_table[index,'h'] # 1 for hr, 2 for nohr
h
seed_split_val = array_table[index,'seed_split']
seed_split_val
# remember to save model

hyper = read_csv(paste0(hyperpara_path,"testhyper_full_ver_4L.csv")) %>% filter(epoch_type==epoch1 & hr== (h==1) )%>%
  select(-l)
al = read_csv(paste0(path_fs,"alignment_full_ver.csv")) %>% rename(epoch_type=epoch) #read_csv(paste0(path_fs,"alignment_full_ver.csv")) %>% rename(epoch_type=epoch)
l_var_lst = al %>% filter(seed_split==seed_split_val&epoch_type==epoch1 & hr== (h==1)&align==hyper$align) %>% select(seed_split,l,var_lst)

hyper = hyper%>% bind_cols(l_var_lst)#  HR
hyper %>% as.data.frame
dim(hyper)


##############
#tag=read_csv(paste0(tag_path,"/sleep_tag.csv"))


###################################################  cnn part
i = 1 # best config_idx, no need to specify, since hyper only has one row

hyper[i,]
# Building the LSTM model
# flags {tfruns}
FLAGS = get_flags(hyper) 

#### read in data
epoch_ty = which(epoch_type==hyper$epoch_type[i])
time = read_csv(paste0(proc_path[epoch_ty],sec[epoch_ty],"_ready.csv")) %>%  ##col_types = cols(calender_time=col_datetime())) %>%
  select(unix_sec)

var_lst = c("e4_id", unlist(str_split(FLAGS$var_lst,',')),"sleep")
modal = unlist(str_split(FLAGS$modality,','))

dt_ready = read_csv(paste0(proc_path[epoch_ty],sec[epoch_ty],"_ready_norm.csv")) %>%
  dplyr::select(all_of(var_lst)) %>% select(e4_id,starts_with(modal),sleep) %>% 
  mutate_at(c("hourofday","age"),funs(scale)) %>%
  bind_cols(time)  

#### get train/test split
###create train, validation and test set
# student index for train, validation and test
idx_files=1:25
set.seed(FLAGS$seed)
idx_all = sample(c(1,3:25),24)
idx_train = idx_all[1:15]   ## 15 files for training
idx_val = idx_all[16:20]    ## 5 for validation
idx_train_val=idx_all[1:20] ## 20 for training and validation
idx_test = sort(idx_all[21:24])   ## 4 for testing
print(idx_train)
print(idx_val)

#gen_split_dt = function(dt_ready,stepsize,var_lst,m,hr=T,LSTM=F)
gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m,LSTM=T) # handled 22, use hr=T here, actually var_lst already consider hr scenario

dt_train=gen_data$dt_train
dt_val=gen_data$dt_val
dt_test=gen_data$dt_test
dt_train_val=gen_data$dt_train_val

train_x = dt_train$dt_x
train_y = dt_train$dt_y

val_x = dt_val$dt_x
val_y = dt_val$dt_y

#checking the dimensions
dim(train_x) 
cat("No of training samples\t",dim(train_x)[[1]],
    "\tNo of validation samples\t",dim(val_x)[[1]])

# > dim(train_x) 
# [1] 4091   18   10
# > cat("No of training samples\t",dim(train_x)[[1]],
#       +     "\tNo of validation samples\t",dim(val_x)[[1]])
# No of training samples	 4091 	No of validation samples	 1305> 
  
# for test subject i in 1:5
test_x_lst = list()
test_y_lst = list()
test_time_lst = list()
len = rep(0,4)  # get total size of each subject's data
for (j in 1:4){
  test_x_lst = c(test_x_lst,list(dt_test[[j]]$dt_x))
  test_y_lst = c(test_y_lst,list(dt_test[[j]]$dt_y))
  # get time
  test_time_lst = c(test_time_lst,list(dt_test[[j]]$dt_time))
  
  len[j] = nrow(test_y_lst[[j]])
}
len
str(test_x_lst)
str(test_y_lst)
str(test_time_lst)


middle_value = function(x) apply(x,1,function(z) z[ceiling(FLAGS$m/2)])
test_time_lst1 = lapply(test_time_lst,middle_value)
str(test_time_lst1)
test_time=unlist(test_time_lst1)
str(test_time)

# > str(test_x_lst)
# List of 4
# $ : num [1:218, 1:18, 1:10] -0.459 -0.312 -0.312 -0.312 -0.312 ...
# $ : num [1:205, 1:18, 1:10] -0.459 -0.312 -0.312 -0.312 -0.312 ...
# $ : num [1:267, 1:18, 1:10] -0.312 -0.312 -0.312 -0.312 -0.312 ...
# $ : num [1:259, 1:18, 1:10] -0.312 -0.312 -0.312 -0.312 -0.312 ...
# > str(test_y_lst)
# List of 4
# $ : num [1:218, 1:18] 0 0 0 0 0 0 0 0 0 0 ...
# $ : num [1:205, 1:18] 0 0 0 0 0 0 0 0 0 0 ...
# $ : num [1:267, 1:18] 0 0 0 0 0 0 0 0 0 0 ...
# $ : num [1:259, 1:18] 0 0 0 0 0 0 0 0 0 0 ...
# > str(test_time_lst)
# List of 4
# $ : num [1:218, 1:18] 1.58e+09 1.58e+09 1.58e+09 1.58e+09 1.58e+09 ...
# $ : num [1:205, 1:18] 1.58e+09 1.58e+09 1.58e+09 1.58e+09 1.58e+09 ...
# $ : num [1:267, 1:18] 1.58e+09 1.58e+09 1.58e+09 1.58e+09 1.58e+09 ...
# $ : num [1:259, 1:18] 1.58e+09 1.58e+09 1.58e+09 1.58e+09 1.58e+09 ...

###### will be used in getting win_size, not sorted!!!!
train_val_x_lst = list()
train_val_y_lst = list()
len_train_val=rep(0,20)
for (j in 1:20){
  train_val_x_lst = c(train_val_x_lst,list(dt_train_val[[j]]$dt_x))
  train_val_y_lst = c(train_val_y_lst,list(dt_train_val[[j]]$dt_y))
  len_train_val[j] = nrow(train_val_y_lst[[j]])
}
len_train_val
cum_len=c(0,cumsum(len_train_val))
cum_len


#######
skip=T
if (!skip){


# train LSTM
# specify hyperparameter
if (epoch_ty==3){
    num_units = 16
}else num_units = 32

library(tensorflow)
library(keras)
lstm_model <- keras_model_sequential()
lstm_model %>%
  bidirectional(layer_lstm(batch_input_shape = dim(train_x),units = num_units,
                           dropout = 0.2,  return_sequences = TRUE
                           #units: size of the layer
                           #input_shape = c(251,18,10), # batch size, timesteps, features
  )) %>%
  layer_dense(units = 1, activation = "sigmoid")
opt<-optimizer_adam(lr=1e-5)    # lr must be this small
lstm_model %>% compile(loss="binary_crossentropy",
                       optimizer=opt,
                       metrics = list("AUC","accuracy","Recall","Precision"))

starttime=Sys.time()
history_lstm = lstm_model  %>% fit(
  x = train_x,y = train_y,
  validation_data = list(val_x, val_y),
  batch_size = FLAGS$batch_size,
  callbacks=callback_early_stopping(monitor = "val_accuracy",
                                    min_delta = 0,
                                    patience = FLAGS$patience),
  epochs = FLAGS$num_epoch
)
endtime=Sys.time()
comp_train_lstm = as.numeric(difftime(endtime,starttime,units="secs"))
summary(lstm_model)
lstm_model %>% save_model_hdf5(paste0("~/E4_CNN_2022/saved_test_models/lstm_",index,".h5"))

#lstm_model <- load_model_hdf5(paste0("~/E4_CNN_2022/saved_test_models/lstm_",index,".h5"))
summary(lstm_model)

all_x=abind::abind(train_x,val_x,along = 1)
all_y=rbind(train_y,val_y)

all_y = as.numeric(rowSums(all_y) > (FLAGS$m/2) )
str(all_y)

pred_train_values = lstm_model %>% predict(all_x,all_y, batch_size = 5)
# dim(pred_train_values)
# [1] 5396   18    1
pred_train_values = pred_train_values[,,1]
str(pred_train_values)
bin_pred_train_values = as.numeric(rowSums(pred_train_values>.5)>(FLAGS$m/2) )
str(bin_pred_train_values)

# train pred values
data.frame(pred_binary=bin_pred_train_values,true_y=all_y) %>% 
  write_csv(paste0("~/E4_CNN_2022/pred_values/lstm_trainval_",index,'.csv'))
}


all_x=abind::abind(train_x,val_x,along = 1)
all_y=rbind(train_y,val_y)
all_y = as.numeric(rowSums(all_y) > (FLAGS$m/2) )
str(all_y)
trainpred = read_csv(paste0("~/E4_CNN_2022/pred_values/lstm_trainval_",index,'.csv'))
bin_pred_train_values = trainpred$pred_binary


#### window size
####### get window size for majority voting      
# should use train_val to get win size
win_size = get_win_size(bin_pred_train_values,cum_len,all_y,grid=1:50,idx_train_val=idx_train_val) 
win_size
win_size %>% as_tibble%>%write_csv(paste0("~/E4_CNN_2022/window_size/lstm_",index,'window_full_ver.csv'))



if (!skip){
final_result_tran = NULL
for (i in 1:length(idx_test)){
  starttime=Sys.time()
  pred = lstm_model  %>% predict(test_x_lst[[i]],test_y_lst[[i]], batch_size = 5)
  endtime=Sys.time()
  comp_cost1 = as.numeric(difftime(endtime,starttime,units="secs"))
  pred_binary = as.numeric(rowSums(pred[,,1]>.5)>(FLAGS$m/2) )
  pred_result = data.frame(pred = apply(pred[,,1],1,mean),pred_binary,
                           time=test_time_lst1[[i]])
  
  #dt_ready_i = dt_ready %>% filter(e4_id==idx_test[i]) %>% 
  #  transmute(true_y=sleep,time=unix_sec)
  ### handle subject 22
  dt_ready_sub = dt_ready %>% filter(e4_id==idx_test[i]) 
  if (idx_test[i]==22){
    dt_ready_sub = dt_ready_sub %>% filter(unix_sec<=1582262785 |unix_sec>=1582316887)
  }
  dt_ready_i = dt_ready_sub %>% transmute(true_y=sleep,time=unix_sec)
  
  
  # get full result for all the epochs.
  final_result = left_join(dt_ready_i,pred_result,by="time") %>%
    mutate(pred_result_full = na.locf(na.locf(pred_binary,na.rm=F), fromLast = T))
  # get transition status
 # transition = get_transition(final_result$true_y,sec[epoch_ty])
  nr = nrow(final_result)
  final_result = final_result %>% transmute(id_test=rep(idx_test[i],nr), 
                                            seed_value=rep(FLAGS$seed,nr),
                                            true_y,pred,pred_binary,pred_result_full,
                                            test_time=time, #transition, 
                                            comp_cost = rep(comp_cost1,nr))
  
  final_result_tran = bind_rows(final_result_tran,final_result)
}

final_result_tran %>% write_csv(paste0("~/E4_CNN_2022/pred_values/lstm_test_",index,'.csv'))
}


final_result_tran = read_csv(paste0("~/E4_CNN_2022/pred_values/lstm_test_",index,'.csv'))

testids = unique(final_result_tran$id_test)
predfull = rep(NA,nrow(final_result_tran ))

set.seed(seed_split_val)
for (testid in testids){
    row_idx = which(final_result_tran$id_test==testid)
    b = a = final_result_tran$pred_binary[row_idx]
    na_idx = which(is.na(a))
# first handle leading and ending NAs
    nonNA_idx1 = min(which(!is.na(a)))
    b[na_idx[na_idx<nonNA_idx1]] = a[nonNA_idx1]
    nonNA_idx2 = max(which(!is.na(a)))
    b[na_idx[na_idx>nonNA_idx2]] = a[nonNA_idx2]
# handle gaps in middle (only work for stepsize=2)
    b1 = b
    na_idx = which(is.na(b))
    delta = b[na_idx-1] - b[na_idx+1]
    b1[na_idx[delta==0]] = b[na_idx[delta==0]+1]
    b1[na_idx[delta!=0]] = rbinom(length(na_idx[delta!=0]),1,prob=.5)
    #b1
    #a
    #rbind(a,b1)
    print(sum(is.na(b1)) == 0)
    predfull[row_idx] = b1
}

sum(is.na(predfull))
sum(final_result_tran$pred_result_full!=predfull)
final_result_tran$pred_result_full=predfull
sum(final_result_tran$pred_result_full!=predfull)
final_result_tran %>% write_csv(paste0("~/E4_CNN_2022/pred_values/lstm_test_",index,'.csv'))




transition_all = matrix(NA,nrow=nrow(final_result_tran),ncol=nrow(periods)) %>% as_tibble
for (pe in 1:nrow(periods)){
  transition_all[,pe] = get_transition(final_result_tran$true_y,sec[epoch_ty], periods$shorter[pe],periods$longer[pe])
}
colnames(transition_all) = paste0('transition_',periods$shorter,'_',periods$longer)
transition_all

final_result_tran = final_result_tran  %>% bind_cols(transition_all)



metrics_full = expand.grid(transition=c('full',paste0(c('static_','transition_'),rep(periods$shorter,each=2),'_',rep(periods$longer,each=2))),
                           mv=c(F,T),idx_test=idx_test)
n = nrow(metrics_full)
metrics_full = metrics_full %>% mutate(AUC=rep(NA,n),acc=rep(NA,n),wacc=rep(NA,n),specificity = rep(NA,n),precision=rep(NA,n),
                                       recall = rep(NA,n), F1 = rep(NA,n)) ## add comp cost


for (i in 1:nrow(metrics_full)){
  id = metrics_full$idx_test[i]
  tran = metrics_full$transition[i]
  mv = metrics_full$mv[i]
  sub = final_result_tran %>% filter(id_test==id)
  print(paste('id:',id,'; tran:',tran,'; mv:',mv))
  # no mv vs mv
  if (mv==T) sub$pred_result_full = majority_vote(sub$pred_result_full,w_size=win_size)
  
  # full vs static vs tran
  if (tran !='full'){
    tran_key = gsub('^([a-z]+)_[0-9]+_[0-9]+','\\1',tran)
    if (tran_key == 'static'){
      tran_id = gsub('^([a-z]+)(_[0-9]+_[0-9]+)','transition\\2',tran) # change the name of variable from static_#_# to transition_#_#
      sub = sub[pull(sub[, tran_id])==0,]  #sub[, tran_id] this is a tibble, so need to pull
    }else{ # tran_key == 'transition'
      sub = sub[pull(sub[, tran])==1,] 
    }
  }
  
  metrics_tb = compute_metrics(sub$pred_result_full,sub$true_y)
  print(metrics_tb)
  metrics_full[i,-(1:3)] = c(metrics_tb$AUC,metrics_tb$Accuracy,metrics_tb$wACC,metrics_tb$Specificity,
                             metrics_tb$Precision,metrics_tb$Recall, metrics_tb$F1score)
  
}

comp_cost_sub =final_result_tran%>% rename(idx_test=id_test)%>%group_by(idx_test) %>% 
  dplyr::select(starts_with('comp_'))%>%  slice(1) %>% ungroup

metrics_full1 = metrics_full %>% inner_join(comp_cost_sub, by='idx_test')# %>% mutate(comp_train_lstm)

metrics_full1 %>% write_csv(paste0("~/E4_CNN_2022/cnn_test_metrics/lstm_test_",index,".csv"))

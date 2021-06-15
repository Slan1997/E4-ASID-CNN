# Functions needed for the experiment.

###################### Workflow Implementation
### reshaping time series into image
# ts: original time series
# len_seg: m, the total number of epochs in a synthesized image
# stepsize: step size of the sliding window

build_matrix <- function(ts, len_seg,stepsize){ 
  sapply(seq(1,(length(ts) - len_seg + 1),stepsize), 
         function(x) ts[x:(x + len_seg - 1)])
}

reformat = function(dt_ready,idx_files,m,stepsize){
  fi_reform_all = NULL 
  # each subject's images are generated independently, 
  # then all subjects' images are combined into one matrix
  for (id in idx_files){ 
    fi =filter(dt_ready,e4_id==id)
    n = nrow(fi)
    idx = build_matrix(1:n,m,stepsize) %>% as.vector()
    dt = fi %>% slice(idx) 
    dt = dt %>% mutate(pt_idx = rep(1:m,nrow(dt)/m))
    fi_reform_all = bind_rows(fi_reform_all,dt)
  }
  return(fi_reform_all)
}

### split train and test
# input format must be: 
# the first column should be e4_id, and the last two column are: sleep, pt_idx.
gen_final_dt = function(fi_reform_all,idx_files,idx_trte,m,l){
  fi_reform = filter(fi_reform_all,e4_id %in% idx_files[idx_trte])
  n = nrow(fi_reform)/m

  dt_x = NULL  
  for (j in 1:m){
    dt_nl = fi_reform %>% filter(pt_idx==j) %>% 
      select(2:(ncol(fi_reform_all)-3) )%>% as.matrix() %>% as.vector()
    dt_x = c(dt_x,dt_nl)
  }

  dim(dt_x) = c(n,l,m,1)
  
  dt_y = fi_reform %>% select(sleep) %>%
    mutate(voter = rep(1:n,each=m)) %>%
    group_by(voter) %>% summarise(s=sum(sleep)) %>% pull(s)
  dt_y[dt_y<(m/2)] = 0 # 0 for wake
  dt_y[dt_y>=(m/2)] = 1  # 1 for sleep
  #dt_y

  dt_time = fi_reform %>% select(unix_sec) %>%
    mutate(time_chunk = rep(1:n,each=m)) %>%
    group_by(time_chunk) %>% 
    summarise(t = mean(unix_sec)) %>% pull(t)
  return(list(dt_x=dt_x,dt_y=dt_y,dt_time=dt_time))
}

### gen_split_dt(): final function for generating reformed train/test/validation dataset
gen_split_dt = function(dt_ready,stepsize,var_lst,m,hr=T){ # hr = 1 or 0 (have or not have)
  if (hr==F) dt_ready = dt_ready %>% select(-(hr_mean:hr_q3))

  l = ncol(dt_ready)-3
  fi_reform_all = reformat(dt_ready,idx_files,m,stepsize)
  dt_train = gen_final_dt(fi_reform_all,idx_files,idx_train,m,l)
  dt_val = gen_final_dt(fi_reform_all,idx_files,idx_val,m,l)

  # test should by individual
  dt_test = list()
  for (k in 1:length(idx_test)){
    dt_test = c(dt_test,list(gen_final_dt(fi_reform_all,idx_files,idx_test[k],m,l)))
  }
  names(dt_test) = paste0("subject_",idx_test)

  dt_train_val =  list()

  for (k in 1:length(idx_train_val)){
    dt_train_val = c(dt_train_val,list(gen_final_dt(fi_reform_all,idx_files,idx_train_val[k],m,l)))
  }
  names(dt_train_val) = paste0("subject_",idx_train_val)
  return(list(dt_train=dt_train,dt_val=dt_val,dt_test=dt_test,dt_train_val=dt_train_val))
}

########### Training => find the best Configuration j*

#### summary_over_seed(): find the max or mean across seed value for each metric
# summary_type="mean" or "max"
summary_over_seed = function(dt,num_seed,summary_type="mean"){   # default is mean
  comb_seed = dt %>% mutate(config_idx = rep(1: (nrow(dt)/num_seed),each=num_seed))
  if (summary_type=="max"){
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), max)
  }else{
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), mean)
  }
  return(over_seed)
}

#### Get training result table
# dt: metric_tb; 
# l: number of layer; 
# EDA_type="cat 3" or "median"; 
# summary_type="mean" or "max"
train_result_tb = function(dt,hyper,l,num_seed,EDA_type="cat 3",summary_type="mean"){
  if (l==1){
    hyper_seedset = hyper %>% select(-2) %>% slice(seq(1,nrow(hyper),num_seed)) %>% 
      mutate(config_idx = 1:(nrow(hyper)/num_seed)) %>% select(18,1:17)
  }else{
    hyper_seedset = hyper %>% select(-2) %>% slice(seq(1,nrow(hyper),num_seed)) %>% 
      mutate(config_idx = 1:(nrow(hyper)/num_seed)) %>% select(25,1:24)
  }
  comb_seed = dt %>% mutate(config_idx = rep(1: (nrow(dt)/num_seed),each=num_seed))
  if (summary_type=="max"){
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), max)
  }else{
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), mean)
  }
  criterias=c("val_accuracy","val_recall","val_precision","val_AUC")
  config_tb = NULL
  criteria_tb = NULL
  for (j in 2:5){
    config_num = which.max(over_seed[,j]%>%pull)
    best_config = hyper_seedset[config_num,]
    config_tb = bind_rows(config_tb,best_config)
    criteria_tb = bind_rows(criteria_tb,
                            over_seed[config_num,] %>% mutate(EDA_type,criteria=criterias[(j-1)]) )
  }
  return(list(best_configs=config_tb,metrics=criteria_tb))
}

#### Test

########### function for CNN's majority vote and weighted accuracy
majority_vote <- function(y_pred,w_size=1){    # window size = 2*w_size+1
  new_pred=y_pred
  for(i in c(1:length(y_pred))){
    new_pred[i]=round(mean(y_pred[max(0,i-w_size):min(i+w_size,length(y_pred))]))
  }
  return(new_pred)
}

weighted_acc <- function(y_pred,y_true){
  w1=sum(y_true)
  w0=length(y_true)-w1
  if (w1!=0){
    w_a=1/2*sum(y_pred[which(y_true==1)]==1)/w1+1/2*sum(y_pred[which(y_true==0)]==0)/w0
  }
  else w_a=mean(y_pred==y_true)
  return(w_a)
}


## Example with simulated 5min epoch data with Heart Rate (D-5min-HR)
setwd('~/E4-ASID-CNN/E4 ASID Example/')
library(pacman)
pacman::p_load(tibble,tidyr, dplyr, readr,lubridate,tfruns,
               rappdirs,scales,caret,magrittr)

### General setup
epoch_type = c("30s","1min","5min")
epo = 3  # "use epoch_type[epo]" to indicate "5min"
epoch = 300 # indicate "5min", unit: sec
source("functions.R") # load functions

### Create train, validation and test set by randomly sampling
# student index for train, validation and test, this variable will be used in functions of image reshaping (see function.R)
idx_files=1:25
# originally, there are 25 students' data, however student 2's data is
# not qualified for the experiment. So only 24 students' data were used in the experiment.
set.seed(890)
idx_all = sample(c(1,3:25),24)  
idx_train = idx_all[1:15]   ## 15 files for training
idx_val = idx_all[16:20]    ## 5 for validation
idx_train_val=idx_all[1:20] ## 20 for training and validation
idx_test = sort(idx_all[21:24])   ## 4 for testing

################### Training process of the ASID Workflow
# Hyperparameter tuning in this example is greatly simplified for both ASID Workflow (CNN) and competing algorithms
# i.e. less possible values of hyperparameters are tried.

# Get variable name list: only consider categorical EDA in this example; for median EDA, replace "eda_cat" by "eda_q2"
var_lst = c("e4_id","acc_mean","acc_sd","acc_q1","acc_q2","acc_q3",
            "temp_mean","temp_sd","temp_q1","temp_q2","temp_q3",
            "hr_mean","hr_sd","hr_q1","hr_q2","hr_q3",    # under "w/o Heart Rate" scenario, this line should be deleted
            "hourofday","eda_cat","age","sex","sleep" )
 
time = read_csv(paste0(epoch_type[epo],"_permute.csv")) %>% select(unix_sec)  # Get the exact time, will be used later for image plotting.

# Import normalized D-5min-HR.
dt_ready = read_csv(paste0(epoch_type[epo],"_permute_norm.csv")) %>%
  select(all_of(var_lst)) %>% bind_cols(time)

# Train the ASID workflow
source("Train_ASID.R")

#### Following results will be generated:
# metric table for ASID Workflow w/ 1-layer CNN
metric_tb
metric_tb %>% write_csv(paste0("metric_",epoch_type[epo],".csv")) 

# metric table for ASID Workflow w/ 2-layer CNN
metric_tb2
metric_tb2 %>% write_csv(paste0("metric_",epoch_type[epo],"_layer2_",".csv")) 

## Summarize training results
# configs of ASID Workflow (w/ 1-layer CNN) with maximal metrics
max_config_tb = train_result_tb(metric_tb,hyper,l=1,num_seed,summary_type="max")$best_configs
max_criteria_tb = train_result_tb(metric_tb,hyper,l=1,num_seed,summary_type="max")$metrics

# configs of ASID Workflow (w/ 2-layer CNN) with maximal metrics
max_config_tb2 = train_result_tb(metric_tb2,hyper2,l=2,num_seed2,summary_type="max")$best_configs
max_criteria_tb2 = train_result_tb(metric_tb2,hyper2,l=2,num_seed2,summary_type="max")$metrics

# reformat "max_config_tb" to be compatible with "max_config_tb2"
max_config_tb1 = max_config_tb %>% transmute(config_idx,epoch_type,m,l,step_size,hr,
                                    kernel_size_h1=kernel_size_h,kernel_size_w1=kernel_size_w,
                                    kernel_shape1=kernel_shape,num_kernel1=num_kernel,
                                    pool_size_h1=pool_size_h, pool_size_w1=pool_size_w,dropout1=dropout,
                                    
                                    kernel_size_h2=NA,kernel_size_w2=NA,kernel_shape2=NA,
                                    num_kernel2=NA,pool_size_h2=NA, pool_size_w2=NA,dropout2=NA,
                                    
                                    fcl_size,optimizer,learning_rate,batch_size,num_epoch)

# Combine above results (1-layer & 2-layer) in one table w.r.t both config and metric. 
# (config_idx may repeat, indicates that this configuration reaches the best result under different evaluation metrics.)

# combined config table
config_comb =  bind_rows(max_config_tb1%>%mutate(conv_layer="1 layer"),
                         max_config_tb2%>%mutate(conv_layer="2 layers"))
config_comb

# combined metric table
criteria_comb = bind_rows(max_criteria_tb,max_criteria_tb2)
criteria_comb

### compare 1-layer vs 2-layer for each metric and find the final best configuration under each metric
criteria_final = NULL
config_final = NULL
for (k in 1:4){
  sub_conf = config_comb%>% slice(c(k,k+4))
  sub_cri = criteria_comb %>% slice(c(k,k+4))
  best = which.max(sub_cri[,(k+1)]%>%pull)  # if equal, will choose layer1 config!
  criteria_final = bind_rows(criteria_final,sub_cri[best,])
  config_final = bind_rows(config_final,sub_conf[best,])
}
config_final
criteria_final

# rearrange the variables to make the table tidier.
out_criteria = criteria_final %>% select(-1) %>%
  mutate(conv_layer=config_final$conv_layer) %>% select(6:7,5,1:4)
out_config = config_final %>% select(-(1:2)) %>%
  mutate(EDA_type=criteria_final$EDA_type) %>% select(24:25,1:23)

out_criteria
out_config

## find config j* with the maximal AUC for test use, saved as a csv file.
hyper_star = out_config %>% slice(4)
hyper_star %>% write_csv(paste0("hyper_star_",epoch_type[epo],".csv"))


################## Test: Build best CNN classifier with config j* and get test result
source("Test_ASID.R")

#### following result will be generated:
## computational cost:
comp_cost1

## Before Majority Vote Metrics for 4 test subjects:
# test accuracy, weighted accuracy, AUC, recall, specificity, precision
test_acc1
mean(test_acc1)

test_acc_weighted
mean(test_acc_weighted)

test_AUC
test_recall
test_specificity
test_precision


## Post Majority Vote Metrics:
# window size for majority voting
win_size

# post MV test accuracy, weighted accuracy, recall, specificity, precision for 4 test subjects
# (note AUC was not calculated post majoirty voting)
mv_test_acc1
mean(mv_test_acc1)

weighted_test_acc1_mv
mean(weighted_test_acc1_mv)

test_recall_mv
mean(test_recall_mv)

test_specificity_mv
mean(test_specificity_mv)

test_precision_mv
mean(test_precision_mv)

# summarized all the results and saved in a csv file.
result = tibble(idx_test, accuracy = test_acc1, weighted_acc = test_acc_weighted,
                AUC=test_AUC, recall=test_recall, precision=test_precision, specificity=test_specificity,
                accuracy_mv = mv_test_acc1, weighted_acc_mv = weighted_test_acc1_mv,
                recall_mv=test_recall_mv, precision_mv=test_precision_mv, specificity_mv=test_specificity_mv, comp_cost = comp_cost1)
result %>% write_csv(paste0("ASID_test_",seed,".csv"))

################## Plot the test result (codes for generating paper's Figure 5)
library(ggplot2)
library(ggpubr)
## construct plot table
pred_labels = NULL
pred_labels_mv = NULL
for (i in 1:4){
  test_x=test_x_lst[[i]]
  test_y=test_y_lst[[i]]
  pred = model1 %>% predict(test_x,test_y, batch_size = 5)
  pred_labels = c(pred_labels,pred>.5)
  pred_labels_mv = c(pred_labels_mv,majority_vote(pred>.5,w_size=win_size))
}
str(pred_labels)
str(pred_labels_mv)

### construct plot table
timezone='America/Detroit'
tb_plot = tibble(subject = rep(idx_test,len),
                 time = as.POSIXct(test_time,origin = "1970-01-01",tz = timezone),
                 true_label = unlist(test_y_lst),
                 pred_label = as.numeric(pred_labels),
                 pred_label_mv = pred_labels_mv) %>%
  mutate(correct= factor(true_label==pred_label),
         correct_mv = factor(true_label==pred_label_mv))
cols <- names(tb_plot)[c(1,3:5)]
tb_plot = tb_plot %>%
  mutate_each_(funs(factor(.)),cols)
str(tb_plot)

tags = read_csv("sleep_tag.csv") # get sleep tags

for (a in 1:4){
  idx_sub = idx_test[a]
  tag_sub = t(tags[idx_sub,2:9]) %>% as_tibble %>% drop_na
  tag_sub = mutate(tag_sub,
                   Tag=as.POSIXct(V1,origin = "1970-01-01",tz = timezone),
                   event = rep(c("sleep starts","sleep ends"),nrow(tag_sub)/2))
  ### before mv
  p1 = ggplot(tb_plot%>% filter(subject==idx_sub),
              aes(x=time,y=pred_label,color=correct)) +
    geom_point() +
    theme_pubr() +
    scale_color_manual(name = "Prediction Validity",
                       labels = c("Incorrect","Correct"),
                       values = c("FALSE" = "#FF6666","TRUE" = "#99CCFF"))+
    geom_vline(data=tag_sub,mapping=aes(xintercept = Tag),color='grey',size=.25) +
    geom_text(data=tag_sub, mapping=aes(x=Tag, y=1,label=event),
              size=3, color="black",angle=90, vjust=-0.5, hjust=-1)+
    scale_x_datetime(date_breaks = "3 hours",
                     labels=label_time(format="%H:%M\n%m/%d",tz=timezone))+
    scale_y_discrete(breaks=c("0","1"),
                     labels=c("Awake", "Sleep"))+
    theme(axis.text.y = element_text(angle = 90, hjust = .5))+
    labs(x='Calender Time', y="Predicted Status")+
    ggtitle(paste("Subject",idx_sub," before mv")) +
    theme(plot.title=element_text()) # hjust = 0.5
  
  #print(p1)
  ggsave(paste0("subject",idx_sub,"_seed",seed,".png"),
         width = 24, height = 12, units = "cm")
  
  #### after mv
  p2 = ggplot(tb_plot%>% filter(subject==idx_sub),
              aes(x=time,y=pred_label_mv,color=correct_mv)) +
    geom_point() +
    theme_pubr() +
    scale_color_manual(name = "Prediction Validity",
                       labels = c("Incorrect","Correct"),
                       values = c("FALSE" = "#FF6666","TRUE" = "#99CCFF"))+
    geom_vline(data=tag_sub,mapping=aes(xintercept = Tag),color='grey',size=.25) +
    geom_text(data=tag_sub, mapping=aes(x=Tag, y=1,label=event),
              size=3, color="black",angle=90, vjust=-0.5, hjust=-1)+
    scale_x_datetime(date_breaks = "3 hours",
                     labels=label_time(format="%H:%M\n%m/%d",tz=timezone))+
    scale_y_discrete(breaks=c("0","1"),
                     labels=c("Awake", "Sleep"))+
    theme(axis.text.y = element_text(angle = 90, hjust = .5))+
    labs(x='Calender Time', y="Predicted Status")+
    ggtitle(paste0("Subject ",idx_sub,": Majority Vote (window=",win_size,")")) +
    theme(plot.title=element_text()) # hjust = 0.5
  #print(p2)
  ggsave(paste0("mv_","subject",idx_sub,"_seed",seed,".png"),
         width = 24, height = 12, units = "cm")
}


####################################### Competing Workflow
source("Competing_workflow.R")

## following result are generated and saved as csv files in the current working directory.
# Logistics regression's result
result_log
result_log %>% write_csv("test_result_logistic.csv")

# SVM linear's result
result_svml
result_svml %>% write_csv("test_result_svml.csv")

# SVM radial's result
result_svmr
result_svmr %>% write_csv("test_result_svmr.csv")

# Random Forest (w/ seed = 613)'s result
result_rf
result_rf %>% write_csv(paste0("test_result_rf_",seed,".csv"))



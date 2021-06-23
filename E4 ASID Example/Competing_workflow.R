# when use categorical EDA
dt_norm <- dt_ready %>% mutate_at(c("sex","eda_cat","sleep"),funs(as.factor))  # when use median EDA, replace "eda_cat" by "eda_q2"
dt_time <- dt_ready %>% select(e4_id,unix_sec)

# create dummy varaibles for sex 
dummies <- dummyVars(sleep ~ ., data = dt_norm)
dt_dum = data.frame(predict(dummies, newdata = dt_norm))%>%
  mutate(sleep=dt_norm$sleep)
dt_dum = dt_dum %>% select(ncol(dt_dum),1:ncol(dt_dum)-1)

# split data for train_val and test
dt_train =  dt_dum %>% filter(e4_id %in% idx_train_val)
dt_test = dt_dum %>% filter(e4_id %in% idx_test)

# create folds which will be used in cross-validation 
group_folds =  groupKFold(dt_train$e4_id, k = 10) 
TrainingParameters = trainControl(index=group_folds,
                                  method = "cv")

# model formula
f = DF2formula(dt_dum[,-2])

# modified these two functions to handle factor.
majority_vote <- function(y_pred,w_size=1){    # window size = 2*w_size+1
  y_pred = as.integer(y_pred)-1
  new_pred=y_pred
  for(i in c(1:length(y_pred))){
    new_pred[i]=round(mean(y_pred[max(0,i-w_size):min(i+w_size,length(y_pred))]))
  }
  new_pred=factor(new_pred,levels = c(0,1))
  return(new_pred)
}

weighted_acc <- function(y_pred,y_true){
  y_true = as.integer(y_true)-1
  y_pred = as.integer(y_pred)-1
  w1=sum(y_true)
  w0=length(y_true)-w1
  if (w1!=0){
    w_a=1/2*sum(y_pred[which(y_true==1)]==1)/w1+1/2*sum(y_pred[which(y_true==0)]==0)/w0
  }
  else{
    w_a=mean(y_pred==y_true)
  }
  return(w_a)
}

# pre majority vote
test_acc = rep(NA,4)  
test_recall = rep(NA,4)
test_precision = rep(NA,4)
test_specificity = rep(NA,4)
test_acc_weighted = rep(NA,4)

# post majority vote
test_acc_mv = rep(NA,4)
test_recall_mv = rep(NA,4)
test_precision_mv = rep(NA,4)
test_specificity_mv = rep(NA,4)
test_acc_weighted_mv = rep(NA,4)


LGModel = caret::train(f, data = dt_train,method = "glm",family = "binomial",
                trControl = TrainingParameters, na.action = na.omit)

# best model and  parameter 
LGModel$finalModel

starttime=Sys.time()
# tune window size of majority vote based on training data
pred_train = predict(LGModel, dt_train)
len_train_val = sapply(idx_train_val, function(x) nrow(filter(dt_train,e4_id==x)))
cum_len=c(0,cumsum(len_train_val))
all_y = dt_train$sleep
diff=NULL
for(i in 1:50){
  before=NULL
  after=NULL
  for (j in 1:20){
    before=c(before,weighted_acc(pred_train[(cum_len[j]+1):cum_len[j+1]],all_y[(cum_len[j]+1):cum_len[j+1]]))
    mv_pred=majority_vote(pred_train[(cum_len[j]+1):cum_len[j+1]],w_size=i)
    after=c(after,weighted_acc(mv_pred,all_y[(cum_len[j]+1):cum_len[j+1]]))
  }
  diff = c(diff, mean(after-before))
}
win_size=which.max(diff)
win_size

for(i in 1:length(idx_test)){
  subject_index = idx_test[i]
  subject_test = dt_test %>% filter(e4_id==subject_index)
  subject_time = dt_time %>% filter(e4_id==subject_index)
  pred = predict(LGModel, subject_test)
  cm = confusionMatrix(pred, subject_test$sleep)
  mv_pred = majority_vote(pred,w_size=win_size)
  cm_mv = confusionMatrix(mv_pred,subject_test$sleep)
  test_acc[i] = cm$overall[1]
  test_acc_mv[i] = cm_mv$overall[1]
    test_precision[i] = cm$byClass[5]
    test_precision_mv[i] = cm_mv$byClass[5]
    test_specificity[i] = cm$byClass[2]
    test_specificity_mv[i] = cm_mv$byClass[2]
    test_recall[i] = cm$byClass[1]
    test_recall_mv[i] = cm_mv$byClass[1]
    test_acc_weighted[i] = cm$byClass[11]
    test_acc_weighted_mv[i] = cm_mv$byClass[11]
}
endtime=Sys.time()
comp_cost=as.numeric(difftime(endtime,starttime,units='secs'))



result_log = tibble(idx_test=as.character(idx_test), accuracy = test_acc,accuracy_mv = test_acc_mv,
                    weighted_acc = test_acc_weighted,weighted_acc_mv = test_acc_weighted_mv,
                    recall=test_recall, recall_mv=test_recall_mv,
                    precision=test_precision, precision_mv=test_precision_mv, 
                    specificity=test_specificity,specificity_mv=test_specificity_mv,
                    comp_cost=comp_cost)
result_log = bind_rows(result_log,colMeans(result_log[,-1],na.rm = T))
result_log[5,1]="mean"


#SVM linear
SVML = caret::train(f, data = dt_train,
             method = "svmLinear",
             trControl= TrainingParameters,
             tuneLength = 10, na.action = na.omit)

#Best model and parameter
SVML$finalModel
SVML$bestTune

starttime=Sys.time()
pred_train = predict(SVML, dt_train)
diff=NULL
for(i in 1:50){
  before=NULL
  after=NULL
  for (j in 1:20){
    before=c(before,weighted_acc(pred_train[(cum_len[j]+1):cum_len[j+1]],all_y[(cum_len[j]+1):cum_len[j+1]]))
    mv_pred=majority_vote(pred_train[(cum_len[j]+1):cum_len[j+1]],w_size=i)
    after=c(after,weighted_acc(mv_pred,all_y[(cum_len[j]+1):cum_len[j+1]]))
  }
  diff = c(diff, mean(after-before))
}
win_size=which.max(diff)
win_size

for(i in 1:length(idx_test)){
  subject_index = idx_test[i]
  subject_test = dt_test %>% filter(e4_id==subject_index)
  pred = predict(SVML, subject_test)
  cm = confusionMatrix(pred, subject_test$sleep)
  mv_pred = majority_vote(pred,w_size=win_size)
  cm_mv = confusionMatrix(mv_pred,subject_test$sleep)
  test_acc[i] = cm$overall[1]
  test_acc_mv[i] = cm_mv$overall[1]
    test_precision[i] = cm$byClass[5]
    test_precision_mv[i] = cm_mv$byClass[5]
    test_specificity[i] = cm$byClass[2]
    test_specificity_mv[i] = cm_mv$byClass[2]
    test_recall[i] = cm$byClass[1]
    test_recall_mv[i] = cm_mv$byClass[1]
    test_acc_weighted[i] = cm$byClass[11]
    test_acc_weighted_mv[i] = cm_mv$byClass[11]
}
endtime=Sys.time()
comp_cost=as.numeric(difftime(endtime,starttime,units='secs'))


result_svml = tibble(idx_test=as.character(idx_test), accuracy = test_acc,accuracy_mv = test_acc_mv,
                     weighted_acc = test_acc_weighted,weighted_acc_mv = test_acc_weighted_mv,
                     recall=test_recall, recall_mv=test_recall_mv,
                     precision=test_precision, precision_mv=test_precision_mv, 
                     specificity=test_specificity,specificity_mv=test_specificity_mv,
                     comp_cost=comp_cost)

result_svml = bind_rows(result_svml,colMeans(result_svml[,-1],na.rm = T))
result_svml[5,1]="mean"


#SVM Radial
grid = expand.grid(sigma = c(0.01, 0.1), C = c(1,10))
SVMR =caret::train(f, data = dt_train,
             method = "svmRadial",
             trControl= TrainingParameters,
             tuneGrid = grid, na.action = na.omit)

starttime=Sys.time()
pred_train = predict(SVMR, dt_train)
diff=NULL
for(i in 1:50){
  before=NULL
  after=NULL
  for (j in 1:20){
    before=c(before,weighted_acc(pred_train[(cum_len[j]+1):cum_len[j+1]],all_y[(cum_len[j]+1):cum_len[j+1]]))
    mv_pred=majority_vote(pred_train[(cum_len[j]+1):cum_len[j+1]],w_size=i)
    after=c(after,weighted_acc(mv_pred,all_y[(cum_len[j]+1):cum_len[j+1]]))
  }
  diff = c(diff, mean(after-before))
}
win_size=which.max(diff)
win_size

for(i in 1:length(idx_test)){
  subject_index = idx_test[i]
  subject_test = dt_test %>% filter(e4_id==subject_index)
  subject_time = dt_time %>% filter(e4_id==subject_index)
  pred = predict(SVMR, subject_test)
  cm = confusionMatrix(pred, subject_test$sleep)
  mv_pred = majority_vote(pred,w_size=win_size)
  cm_mv = confusionMatrix(mv_pred,subject_test$sleep)
  test_acc[i] = cm$overall[1]
  test_acc_mv[i] = cm_mv$overall[1]
    test_precision[i] = cm$byClass[5]
    test_precision_mv[i] = cm_mv$byClass[5]
    test_specificity[i] = cm$byClass[2]
    test_specificity_mv[i] = cm_mv$byClass[2]
    test_recall[i] = cm$byClass[1]
    test_recall_mv[i] = cm_mv$byClass[1]
    test_acc_weighted[i] = cm$byClass[11]
    test_acc_weighted_mv[i] = cm_mv$byClass[11]
}
endtime=Sys.time()
comp_cost=as.numeric(difftime(endtime,starttime,units='secs'))

result_svmr = tibble(idx_test=as.character(idx_test), accuracy = test_acc,accuracy_mv = test_acc_mv,
                     weighted_acc = test_acc_weighted,weighted_acc_mv = test_acc_weighted_mv,
                     recall=test_recall, recall_mv=test_recall_mv,
                     precision=test_precision, precision_mv=test_precision_mv, 
                     specificity=test_specificity,specificity_mv=test_specificity_mv,
                     comp_cost=comp_cost)
result_svmr=bind_rows(result_svmr,colMeans(result_svmr[,-1],na.rm = T))
result_svmr[5,1]="mean"


# Random Forest (rf)
seed = 613 # tried 10 seeds in original experiment
grid = expand.grid(mtry = c(3:4))   
set.seed(seed)
RFModel =caret::train(sleep~., data = dt_train,method = "rf",
                trControl= TrainingParameters,
                tuneGrid = grid, na.action = na.omit)


starttime=Sys.time()
pred_train = predict(RFModel, dt_train)
len_train_val = sapply(idx_train_val, function(x) nrow(filter(dt_train,e4_id==x)))
cum_len=c(0,cumsum(len_train_val))
all_y = dt_train$sleep
diff=NULL
for(i in 1:50){
  before=NULL
  after=NULL
  for (j in 1:20){
    before=c(before,weighted_acc(pred_train[(cum_len[j]+1):cum_len[j+1]],all_y[(cum_len[j]+1):cum_len[j+1]]))
    mv_pred=majority_vote(pred_train[(cum_len[j]+1):cum_len[j+1]],w_size=i)
    after=c(after,weighted_acc(mv_pred,all_y[(cum_len[j]+1):cum_len[j+1]]))
  }
  diff = c(diff, mean(after-before))
}
win_size=which.max(diff)
win_size

for(i in 1:length(idx_test)){
  subject_index = idx_test[i]
  subject_test = dt_test %>% filter(e4_id==subject_index)
  subject_time = dt_time %>% filter(e4_id==subject_index)
  pred = predict(RFModel, subject_test)
  cm = confusionMatrix(pred, subject_test$sleep)
  mv_pred = majority_vote(pred,w_size=win_size)
  cm_mv = confusionMatrix(mv_pred,subject_test$sleep)
  test_acc[i] = cm$overall[1]
  test_acc_mv[i] = cm_mv$overall[1]
  test_precision[i] = cm$byClass[5]
  test_precision_mv[i] = cm_mv$byClass[5]
  test_specificity[i] = cm$byClass[2]
  test_specificity_mv[i] = cm_mv$byClass[2]
  test_recall[i] = cm$byClass[1]
  test_recall_mv[i] = cm_mv$byClass[1]
  test_acc_weighted[i] = cm$byClass[11]
  test_acc_weighted_mv[i] = cm_mv$byClass[11]
}
endtime=Sys.time()
comp_cost=as.numeric(difftime(endtime,starttime,units='secs'))


result_rf = tibble(idx_test=idx_test, accuracy = test_acc,accuracy_mv = test_acc_mv,
                   weighted_acc = test_acc_weighted,weighted_acc_mv = test_acc_weighted_mv,
                   recall=test_recall, recall_mv=test_recall_mv,
                   precision=test_precision, precision_mv=test_precision_mv, 
                   specificity=test_specificity,specificity_mv=test_specificity_mv,
                   comp_cost=comp_cost)



################### Alignment of modalities and variables
# The alignment of modalities (denoted as “Alignment”) is treated as a hyperparameter 
# with 5 randomly selected Alignment candidates defined. For “w/ HR”, they are: 
# A1, (EDA, ACC, TEMP, HR); 
# A2, (TEMP, ACC, EDA, HR); 
# A3, (ACC, HR, EDA, TEMP); 
# A4, (TEMP, HR, ACC, EDA); 
# A5, (HR, ACC, EDA, TEMP). 
# 
# Within one modality, the variables are ordered by their rank (termed as “Order”), 
# which is computed as the average of importance ranks given by RF classifiers 
# built separately on jointly permuted train and validation sets. 

seeds_table = expand.grid(splits= c(890, 9818,7,155,642), # seeds for 5 splits
                          models = c(18,29,1349,246,904), # every split have 5 seeds for rf.
                          epo=1:3,hr=1:2)
# this results in 150 cases in total

# seed_idx = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID")) # use "#SBATCH --array=1-150" if run on cluster
# here to show an example, only run one case: seed_idx = 51
seed_idx = 51
seeds_table[seed_idx,]
epo = seeds_table[seed_idx,'epo']  # "use epoch_type[epo]" to indicate "5min"
h = seeds_table[seed_idx,'hr']# # hr
seed_split = seeds_table[seed_idx,'splits']  # 890
seed_model = seeds_table[seed_idx,'models']


features = dt_train%>% dplyr::select(acc_mean:hourofday) 
f = DF2formula(dt_dum[,!(colnames(dt_dum) %in% c('e4_id'))]) # remove e4_id
f
### train RF classifiers to get importance rank
set.seed(seed_model)
RFModel <- randomForest(f, data = dt_train, importance=T)
importance(RFModel)
#varImpPlot(RFModel)
im = as.data.frame(importance(RFModel)) %>% arrange(desc(MeanDecreaseAccuracy))
im$feature = rownames(im)
im$rank = 1:nrow(im)
im %>% write_csv(paste0(folder_path,"/sample_result/","rf_full",seed_idx,".csv"))

####### If run all the 150 cases, there will be 150 rank tables rf_full1 - rf_full150
# following codes is to combine these tables, assign alignment of modalities and generate the final candidate alignments and orders. 

# combined_rank = tibble()
# for (i in 1:nrow(seeds_table)){
#   fi_name = paste0("rf_full",i,'.csv')   #paste0("rf_imp",i,'.csv')
#   temp_data = read_csv(paste0(path_fs,fi_name)) %>% 
#     transmute(epoch=epoch_type[seeds_table[i,'epo']],
#               hr = seeds_table[i,'hr'],
#               seed_split =seeds_table[i,'splits'] , 
#               seed_model =seeds_table[i,'models'],
#               feature,rank)
#   combined_rank <- bind_rows(combined_rank, temp_data)              
# }
# 
# combined_rank %>% write_csv(paste0(path_fs,'combined_rank_full_ver.csv') )

### result table from our original experiment are saved in the "original_result" folder.
combined_rank = read_csv(paste0(folder_path,"/original_result/",'combined_rank_full_ver.csv') )

# compute sum of rank across model seeds for all ranked features (results is the same as computing mean of rank)
align=tibble()
for (epo in 1:3){
  for (h in 1:2){
    sub_dt = combined_rank %>% filter(epoch==epoch_type[epo]&hr==h)
    align = bind_rows(align,get_final_rank(sub_dt)%>%
                        mutate(epoch=epoch_type[epo],hr=(h==1)))
  }
}
align
align %>% write_csv(paste0(folder_path,"/original_result/","alignment_full_ver.csv")) 




### Generate ASID Workflow w/ 2-layer CNN configuration table 
hyperpara_path = paste0(folder_path,"/original_result/hyperparameter/")
best_config = read_csv(paste0(hyperpara_path,"L1train_config_best_4metrics_full_ver.csv"))   

best_config_al = best_config %>% group_by(epoch_type,hr) %>% slice(which(duplicated(config_align)==F)) %>%  # remove replicates caused by different metrics.
  dplyr::select(align:config_align) %>% mutate(L1_config_align=1:n()) %>% ungroup %>%
  mutate(row_idx_L1=1:n()) %>%  # get all the config, regardless of epoch type and hr, to combine with L2's config
  rename(kernel_size_h1=kernel_size_h,kernel_size_w1=kernel_size_w, 
         kernel_shape1=kernel_shape,num_kernel1 = num_kernel, 
         pool_size_h1=pool_size_h, pool_size_w1=pool_size_w,
         dropout1=dropout) 


kernel_size_h2 = c(1,3)
kernel_size_w2 = c(1,3)
num_kernel2 = 16
# pool size layer2:   
pool_size_h2 = 1:2
pool_size_w2 = 1:2
dropout2 = c(0,0.5)

######################### Output
# fully-connected layer size
fcl_size = c(16)
optim = c("adam")   # optimizer
learning_rate = c(1e-2,1e-3)
batch_size = c(100)
num_epoch = c(150)

tb = expand.grid(seed_split=c(890, 9818,7,155,642),row_idx_L1=1:nrow(best_config_al),
                 #conv layer2
                 kernel_size_h2=kernel_size_h2,kernel_size_w2=kernel_size_w2,
                 num_kernel2=num_kernel2,pool_size_h2=pool_size_h2,pool_size_w2=pool_size_w2,
                 dropout2=dropout2,
                 #output layer
                 fcl_size=fcl_size,optimizer=optim,
                 learning_rate=learning_rate,batch_size=batch_size,
                 num_epoch=num_epoch) %>% as_tibble
tb1 = full_join(best_config_al %>% dplyr::select(align:pool_size_w1,dropout1,row_idx_L1),tb,by='row_idx_L1') %>%
  dplyr::select(-c('l','modal'))

al = read_csv(paste0(folder_path,"/original_result/","alignment_full_ver.csv")) %>% 
  rename(epoch_type=epoch) 
al # to get the corresponding var_list for each seed_split
# al contains hr

tb2 = full_join(tb1,al,by=c('seed_split','align','epoch_type','hr')) %>% 
  dplyr::select(seed_split,align,l:var_lst,epoch_type:row_idx_L1,kernel_size_h2:num_epoch) 

tb3 = tb2 %>% group_by(epoch_type,hr) %>% mutate(config_align_L2 = rep(1:(n()/5),each=5)) %>% ungroup

tb4 = tb3 %>% mutate(a1 = l%/%pool_size_h1, #- (kernel_size_h2-1),
                     b1 = m%/%pool_size_w1,# - (kernel_size_w2-1),
                     a2 = a1 %/% pool_size_h2,
                     b2 = b1 %/% pool_size_w2) %>%
  filter( ( pool_size_h2 <=  a1) & ( pool_size_w2 <= b1) & 
            # don't want pool size = 1*1 => meaningless
            !(pool_size_h2==1&pool_size_w2==1)  )  

config_align_L2_full = tb4  %>% group_by(config_align_L2 ) %>%
  summarize(n=n())%>% filter(n==30)%>% dplyr::select(config_align_L2)%>%pull
tb_full = tb4 %>% filter(config_align_L2 %in% config_align_L2_full) %>% 
  group_by(epoch_type,hr) %>% mutate(config_align_L2 = rep(1:(n()/5),each=5)) %>% ungroup



# add shape indicator
kernel_shape2 = 1:nrow(tb_full)
kernel_shape2[tb_full$kernel_size_h2 < tb_full$kernel_size_w2] = "wide rec"
kernel_shape2[tb_full$kernel_size_h2 == tb_full$kernel_size_w2] = "square"
kernel_shape2[tb_full$kernel_size_h2 > tb_full$kernel_size_w2] = "long rec"

tb_full1 = tb_full %>% mutate(kernel_shape2) %>% dplyr::select(seed_split:kernel_size_w2,kernel_shape2,
                                                        num_kernel2:config_align_L2)

#tb_full1 %>% write.csv(paste0(hyperpara_path,"hyperparameter_table_L2_align.csv"))
tb_full1 %>% write.csv(paste0(hyperpara_path,"hyperparameter_table_L2_align_full_ver.csv"))









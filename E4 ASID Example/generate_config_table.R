### Generate ASID Workflow w/ 1-layer CNN configuration table
# For simplicity, hyperparameters combinations used in this example are much less than
# those used in real experiment.
hyperpara_path = paste0(folder_path,"/original_result/hyperparameter/")

epoch_type = c("30s","1min","5min")
m_list = list(c(45, 60, 75),  # for 30s
              c(45, 60, 75),  # for 1min
              c(12, 15, 18))  # for 5min
### generate hyperparameter table for one-layer cnn

tb_final = NULL
for (j in 1:3){
  epoch = epoch_type[j]
  m_s = m_list[[j]]
  tb_full = NULL
  for (i in 1:length(m_s)){
    seed = c(890, 9818,7,155,642)
    m = m_s[i]
    step_size = 2
  
    ## kernel_size = h*w, 
    # if h<w, wide rectangular; if h>w, long rectangular; if h=w, square
    kernel_size_h = c(1,3)
    if (m<=15){
      kernel_size_w = c(1,3)
    }else{
      kernel_size_w = c(1,3,5)
    }
    
    # number of kernels
    num_kernel = c(16)
    
    # pool size
    pool_size_h = c(3:5)
    pool_size_w = c(3:5)
    
    # stride for kernel
    # stride for pool
    # not specify here! by default is 1*1
    
    # fully-connected layer size
    fcl_size = 16
    
    # dropout rate
    dropout = c(0.5)
    
    # optimizer
    optim = c("adam")  
    
    learning_rate = c(1e-3)
    
    batch_size = c(100)
    
    num_epoch = c(150)
    
    tb = expand.grid(epoch_type=epoch,seed_split=seed,m=m,step_size=step_size,   
                     kernel_size_h=kernel_size_h,kernel_size_w=kernel_size_w,
                     num_kernel=num_kernel,pool_size_h=pool_size_h,pool_size_w=pool_size_w,
                     fcl_size=fcl_size,dropout=dropout,optimizer=optim,
                     learning_rate=learning_rate,batch_size=batch_size,
                     num_epoch=num_epoch) %>% as_tibble
    
    # fcl_size should be less or equal to a1*b1*num_kernel
    # tb = tb%>% mutate(a1 = l%/%pool_size_h,
    #                   b1 = m%/%pool_size_w)# %>%
    #   #filter( fcl_size <= a1*b1*num_kernel  )
    
    tb_full = bind_rows(tb_full,tb)
  }
  
  # add shape indicator
  kernel_shape = 1:nrow(tb_full)
  kernel_shape[tb_full$kernel_size_h < tb_full$kernel_size_w] = "wide rec"
  kernel_shape[tb_full$kernel_size_h == tb_full$kernel_size_w] = "square"
  kernel_shape[tb_full$kernel_size_h > tb_full$kernel_size_w] = "long rec"
  tb_full1 = tb_full %>% mutate(kernel_shape,config = rep(1:(nrow(tb_full)/5),each=5)) %>% 
    dplyr::select(epoch_type,config,seed_split:kernel_size_w,kernel_shape,num_kernel:num_epoch)
  #tb_full1 %>% write.csv(paste0(hyperpara_path,"hyerpara_",epoch,".csv"))
  tb_final = bind_rows(tb_final,tb_full1)
}

tb_final %>% write.csv(paste0(hyperpara_path,"hyperparameter_table.csv"))

# seed_split,epoch
al = read_csv(paste0(folder_path,"/original_result/","alignment_full_ver.csv")) %>% 
  rename(epoch_type=epoch) 

hyper_al_final = full_join(al,tb_final,by=c('epoch_type','seed_split')) 

hyper_al_final %>% write.csv(paste0(hyperpara_path,"hyperparameter_table_L1_align_full_ver.csv")) #write.csv(paste0(hyperpara_path,"hyperparameter_table_L1_align.csv"))
dim(hyper_al_final )




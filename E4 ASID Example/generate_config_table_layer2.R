### Generate ASID Workflow w/ 2-layer CNN configuration table for 5min epoch data w/ HR (D-5min-HR)

## use mean across seed to decide the best 4 configs from layer1 and pass them to layer 2
mean_config_tb = train_result_tb(metric_tb,hyper,l=1,num_seed)$best_configs
hyper_layer1 = mean_config_tb %>% distinct() %>% # use distinct to avoid repeated best_configurations.
  select(m:pool_size_w) # only leave hyperparameters of first convolutional layer

tb_full = NULL # Create a table to store all the values later.
seed = c(394,838)  
num_seed2=length(seed)
for (i in 1:nrow(hyper_layer1)){
  m = hyper_layer1$m[i]
  l = hyper_layer1$l[i]
  step_size = hyper_layer1$step_size[i]
  hr = hyper_layer1$hr[i]
  
  ######################## 1st convolutional layer
  kernel_size_h1 = hyper_layer1$kernel_size_h[i]
  kernel_size_w1 = hyper_layer1$kernel_size_w[i]
  # number of kernels
  num_kernel1 = hyper_layer1$num_kernel[i]
  # pool size
  pool_size_h1 = hyper_layer1$pool_size_h[i]
  pool_size_w1 = hyper_layer1$pool_size_w[i]
  dropout1 = c(0.5)
  # default stride for kernel is (1,1)
  # default stride for pool is the same as pool size (h,w)
  
  ##### compute the size of the current output=a*b
  a = l%/%pool_size_h1 
  b = m%/%pool_size_w1 
  
  ######################## 2nd convolutional layer
  ## kernel_size2 = h*w, 
  # if h<w, wide rectangular; if h>w, long rectangular; if h=w, square
  # h should be <= a, w should be <= b.
  
  # po = c(1,3) # potential values for kernal size
  kernel_size_h2 = 1 # po[po<=a]
  kernel_size_w2 = 1 # po[po<=b]
  # number of kernels
  num_kernel2 = 16
  
  # pool size layer2:   
  pool_size_h2 = 2
  pool_size_w2 = 1
  
  dropout2 = 0
  
  ######################### Output
  # fully-connected layer size
  fcl_size = c(16)
  
  optim = c("adam")   # optimizer
  learning_rate = 1e-2
  batch_size = c(100)
  num_epoch = c(150)
  
  tb = expand.grid(epoch_type=epoch_type[epo],seed=seed,m=m,l=l,step_size=step_size,hr=hr, #input layer
                   #conv layer1
                   kernel_size_h1=kernel_size_h1,kernel_size_w1=kernel_size_w1,
                   num_kernel1=num_kernel1,pool_size_h1=pool_size_h1,pool_size_w1=pool_size_w1,
                   dropout1=dropout1,
                   #conv layer2
                   kernel_size_h2=kernel_size_h2,kernel_size_w2=kernel_size_w2,
                   num_kernel2=num_kernel2,pool_size_h2=pool_size_h2,pool_size_w2=pool_size_w2,
                   dropout2=dropout2,
                   #output layer
                   fcl_size=fcl_size,optimizer=optim,
                   learning_rate=learning_rate,batch_size=batch_size,
                   num_epoch=num_epoch) %>% as_tibble 
  ### Use following code to make sure:
  # pool_size_h2 should be <= l%/%pool_size_h1 -(kernel_size_h2-1)
  # pool_size_w2 should be <= m%/%pool_size_w1 -(kernel_size_w2-1) 
  # fcl_size should be <=
  # {[l%/%pool_size_h1-(kernel_size_h2-1)] %/% pool_size_h2} * 
  # {[m%/%pool_size_w1-(kernel_size_w2-1)] %/% pool_size_w2 }* num_kernel2
  tb = tb%>% mutate(a1 = l%/%pool_size_h1 - (kernel_size_h2-1),
                    b1 = m%/%pool_size_w1 - (kernel_size_w2-1),
                    a2 = a1 %/% pool_size_h2,
                    b2 = b1 %/% pool_size_w2) %>%
    filter( ( pool_size_h2 <=  a1) & ( pool_size_w2 <= b1)# don't want pool size = 1*1 => meaningless & 
             # !(pool_size_h2==1&pool_size_w2==1)  ) #&  (fcl_size <= a2*b2*num_kernel2)  ) 
  tb_full = bind_rows(tb_full,tb)
}
# add shape indicator
kernel_shape1 = 1:nrow(tb_full)
kernel_shape1[tb_full$kernel_size_h1 < tb_full$kernel_size_w1] = "wide rec"
kernel_shape1[tb_full$kernel_size_h1 == tb_full$kernel_size_w1] = "square"
kernel_shape1[tb_full$kernel_size_h1 > tb_full$kernel_size_w1] = "long rec"

kernel_shape2 = 1:nrow(tb_full)
kernel_shape2[tb_full$kernel_size_h2 < tb_full$kernel_size_w2] = "wide rec"
kernel_shape2[tb_full$kernel_size_h2 == tb_full$kernel_size_w2] = "square"
kernel_shape2[tb_full$kernel_size_h2 > tb_full$kernel_size_w2] = "long rec"

hyper2 = tb_full %>% mutate(kernel_shape1,kernel_shape2) %>% select(1:8,28,9:14,29,15:23)

hyper2 %>% write_csv(paste0(getwd(),"/hyerpara_",epoch_type[epo],"_layer2.csv"))  
  






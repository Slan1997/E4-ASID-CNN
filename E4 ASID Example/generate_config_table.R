### Generate ASID Workflow w/ 1-layer CNN configuration table for 5min epoch data w/ HR (D-5min-HR)
# For simplicity, hyperparameters combinations used in this example are much less than
# those used in real experiment.

m_s = c(12)  # should try different m values, for simplicity, only use one value here.

tb_full = NULL # Create a table to store all the values later.

seed = c(818,7)  # Try different initial number for CNN model parameter training.
num_seed=length(seed)

for (i in 1:length(m_s)){
  m = m_s[i]
  l = 19  # l = 19 for w/ HR, this line should be changed to l = 14 for w/o HR
  step_size = 2 # only use 2 in this example, other possible values can added as alternatives
  hr = TRUE  # this line should be changed to hr = FALSE for w/o HR
  
  ## kernel_size = h*w, 
  # if h<w, wide rectangular; if h>w, long rectangular; if h=w, square
  kernel_size_h = c(3,5)
  kernel_size_w = c(1,3) 
  
  # number of kernels
  num_kernel = c(16)
  
  # pool size
  pool_size_h = c(3:4)
  pool_size_w = c(3:4)
  

  # stride for kernel
  # stride for pool
  # not specify here! by default is 1*1
  
  # fully-connected layer size
  fcl_size = 32
  
  # dropout rate
  dropout = c(0.5)
  
  # optimizer
  optim = c("adam")  
  
  learning_rate = c(1e-3)
  
  batch_size = c(100)
  
  num_epoch = c(150)
  
  tb = expand.grid(epoch_type=epoch_type[epo],seed=seed,m=m,l=l,step_size=step_size, hr=hr,  
                   kernel_size_h=kernel_size_h,kernel_size_w=kernel_size_w,
                   num_kernel=num_kernel,pool_size_h=pool_size_h,pool_size_w=pool_size_w,
                   fcl_size=fcl_size,dropout=dropout,optimizer=optim,
                   learning_rate=learning_rate,batch_size=batch_size,
                   num_epoch=num_epoch) %>% as_tibble
  ## following code can be used to make sure that fcl_size is less or equal to a1*b1*num_kernel
  # tb = tb%>% mutate(a1 = l%/%pool_size_h,
  #                   b1 = m%/%pool_size_w) %>%
  #   filter( fcl_size <= a1*b1*num_kernel  )
  
  tb_full = bind_rows(tb_full,tb)
}

# add shape indicator
kernel_shape = 1:nrow(tb_full)
kernel_shape[tb_full$kernel_size_h < tb_full$kernel_size_w] = "wide rec"
kernel_shape[tb_full$kernel_size_h == tb_full$kernel_size_w] = "square"
kernel_shape[tb_full$kernel_size_h > tb_full$kernel_size_w] = "long rec"
hyper = tb_full %>% mutate(kernel_shape) %>% select(1:8,18,9:17)

hyper %>% write_csv(paste0(getwd(),"/hyerpara_",epoch_type[epo],".csv"))
 




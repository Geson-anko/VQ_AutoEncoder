"""
This is a hyper parameters of VQ_AutoEncoder for CIFAR100 
"""

max_view_imgs = 16
view_interval = 50

model_name:str = 'CIFAR100_afterVQ'
lr:float = 0.001
momentum:float = 0.9
num_quantizing:int = 500
quantizing_dim:int = 32

encoder_channels = (8,16,32)
res_kernel = 5
res_num_layers = 2

class encoder_hparams:
    model_name = model_name + '_encoder'
    quantizing_dim = quantizing_dim
    encoder_channels = encoder_channels
    ch0,ch1,ch2 = encoder_channels
    res_kernel,res_num_layers = res_kernel,res_num_layers

class decoder_hparams:
    model_name = model_name + '_decoder'
    quantizing_dim = quantizing_dim
    encoder_channels = encoder_channels
    res_kernel,res_num_layers = res_kernel,res_num_layers

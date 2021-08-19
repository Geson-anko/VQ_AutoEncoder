# %% importing
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from torch.utils import data as DataUtil

from MNIST_model import VQ_AutoEncoder
from hparams import MNIST_default as hparams
from datetime import datetime
def get_now(strf:str = '%Y-%m-%d_%H-%M-%S'):
    now = datetime.now().strftime(strf)
    return now

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# %% set dataset
data_set = MNIST('data',False,download=True,
    transform= transforms.Compose(
        [transforms.ToTensor(),]
    )
)
# %% set some settings
EPOCHS = 500
batch_size= 1024
model = VQ_AutoEncoder(hparams)
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)

#%%
model.encoder.load_state_dict(torch.load('params/MNIST_default_encoder_2021-08-18_21-11-47.pth'))
model.decoder.load_state_dict(torch.load('params/MNIST_default_decoder_2021-08-18_21-11-47.pth'))
model.set_quantizing_weight(data_loader)
#%%
if __name__ == '__main__':
    from pytorch_lightning import loggers as pl_loggers
    logger = pl_loggers.TensorBoardLogger('VQ_AutoEncoderLog',name='fromTraining')
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS,logger=logger,log_every_n_steps=5)
    trainer.fit(model,data_loader)
    torch.save(model.state_dict(),f'params/{model.model_name}_{get_now()}.pth')





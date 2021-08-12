# %% importing
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
import pytorch_lightning as pl
from torch.utils import data as DataUtil

from CIFAR100_model import VQ_AutoEncoder
from hparams import CIFAR100_afterVQ as hparams
from datetime import datetime
def get_now(strf:str = '%Y-%m-%d_%H-%M-%S'):
    now = datetime.now().strftime(strf)
    return now

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# %% set dataset
data_set = CIFAR100('data',False,download=True,
    transform= transforms.Compose(
        [transforms.ToTensor(),]
    )
)
# %% set some settings
EPOCHS = 1000
batch_size= 1024 *8
model = VQ_AutoEncoder(hparams)
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
#%%
model.encoder.load_state_dict(torch.load('params/CIFAR100_encoder.pth'))
model.decoder.load_state_dict(torch.load('params/CIFAR100_decoder.pth'))
#%%
if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,data_loader)
    torch.save(model.state_dict(),f'params/{model.model_name}_{get_now()}.pth')





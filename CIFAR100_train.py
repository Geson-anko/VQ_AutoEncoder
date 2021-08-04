# %% importing
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
import pytorch_lightning as pl
from torch.utils import data as DataUtil

from CIFAR100_model import VQ_AutoEncoder
from hparams import CIFAR100_default as hparams

# %% set dataset
data_set = CIFAR100('data',False,download=False,
    transform= transforms.Compose(
        [transforms.ToTensor(),]
    )
)
# %% set some settings
EPOCHS = 1000
batch_size= 1024 *4
model = VQ_AutoEncoder(hparams)
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)

#%%
if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,data_loader)





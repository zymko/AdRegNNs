import torch
from data import *
import matplotlib.pyplot as plt
# import pytorch_lightning as pl
from model import *
from util import *
from optimizer import *

### use fixed random seed
seed=123
torch.manual_seed(seed)
coeff_matrix= torch.randn(28, 28)
batch_size=64
# coeff_matrix

train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size=batch_size, coeff_matrix=coeff_matrix, num_workers=0, validation_fraction=0.20, std=0.1, mean=0)
len(train_loader), len(valid_loader), len(test_loader)

print('Training Set:\n')
for images, labels, noises in train_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print('Image noise dimensions:', noises.size())
    # print(labels[:10])
    break
    
# Checking the dataset
print('\nValidation Set:')
for images, labels, noises in valid_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print('Image noise dimensions:', noises.size())
    # print(labels[:10])
    break

# Checking the dataset
print('\nTesting Set:')
for images, labels, noises in test_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print('Image noise dimensions:', noises.size())
    # print(labels[:10])
    break

torch.manual_seed(seed)
fig, axs = plt.subplots(4,4,figsize=(8, 8))
fig_0, axes = plt.subplots(4,4,figsize=(8, 8))
no_images = axs.shape[0] * axs.shape[1]
axs = axs.reshape(-1)
axes = axes.reshape(-1)
end=16
for images, labels, noises in valid_loader:  
    for i, image in enumerate(images[0:end]):
        axs[i].imshow(image.reshape(image.shape[-2:]),cmap='gray')
        axs[i].set_title(labels[i].item(), fontsize=16)
        axs[i].get_xaxis().set_ticks([]), axs[i].get_yaxis().set_ticks([])
    for i, noise in enumerate(noises[0:end]):    
        axes[i].imshow(noise.reshape(noise.shape[-2:]),cmap='gray')
        axes[i].set_title(labels[i].item(), fontsize=16)
        axes[i].get_xaxis().set_ticks([]), axes[i].get_yaxis().set_ticks([])
    break  
fig.savefig("figures/MNIST_data_gr.png")
fig_0.savefig("figures/MNIST_data_measurments.png")


torch.manual_seed(seed)
instance_batch=get_batch_instance(valid_loader)
instance_batch['image'].size()
initial_guess = get_initial_guess(noises=instance_batch['noise'], coeff_matrix=coeff_matrix, grad_requires=False)
fig, axs = plt.subplots(4,4,figsize=(8, 8))
fig_0, axes = plt.subplots(4,4,figsize=(8, 8))
no_images = axs.shape[0] * axs.shape[1]
axs = axs.reshape(-1)
axes = axes.reshape(-1)
end=16
for i, image in enumerate(instance_batch['image'][0:end]):
    axs[i].imshow(image.reshape(image.shape[-2:]),cmap='gray')
    axs[i].set_title(instance_batch['label'][i].item(), fontsize=16)
    axs[i].get_xaxis().set_ticks([]), axs[i].get_yaxis().set_ticks([])
for i, noise in enumerate(initial_guess[0:end]):    
    axes[i].imshow(noise.reshape(noise.shape[-2:]),cmap='gray')
    axes[i].set_title(instance_batch['label'][i].item(), fontsize=16)
    axes[i].get_xaxis().set_ticks([]), axes[i].get_yaxis().set_ticks([])

fig.savefig("figures/MNIST_data_vs_gr.png")
fig_0.savefig("figures/MNIST_data_vs_lstsq.png")


optimizers = ["Adam", "SGD", "AdamW", "mSGD", "RMSprop"]
# optimizers = ["mSGD"]

# Learning rate recommendations for different optimizers
lr_dict = {
    "Adam": 1e-3,
    "RMSprop": 1e-3,
    "AdamW": 1e-3,
    "SGD": 1e-1,    # Plain SGD needs higher LR
    "mSGD": 1e-2    # Momentum SGD
}

for optimizer in optimizers:

    print(f"Training with {optimizer} optimizer")
    
    # Get specific LR or default to 1e-3
    current_lr = lr_dict.get(optimizer, 1e-3)
    
    h_params = {
        'input_channels':1,
        'epochs':3,
        'lambda':0.001,
        'coeff_matrix': coeff_matrix,
        'mu':1.5,
        'step_size':100,
        'lr':current_lr,
        'gamma':0.9,
        'batch_size':64,
        'opt_name': optimizer
    }

    model = Conv(h_params=h_params)

    import os
    os.environ["WANDB_API_KEY"] = "9168bec9c57359bde71178e40f0eb6d9e2daabdb"

    import wandb
    wandb.login()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    run_name = h_params['opt_name']
    model.train_regularization(train_loader, valid_loader, run_name=run_name, device=device)

    # torch.manual_seed(seed)
    # inputs, labels, noises = next(iter(valid_loader))
    # batch_instance ={'images': inputs,
    #                 'labels': labels,
    #                 'noises': noises}
                    

    # image, noise = inversion_solver(model=model,
    #                     lr=1e-2, 
    #                     batch=batch_instance, 
    #                     Lambda=80, epochs=1500, AdReg=True)

    # torch.manual_seed(1234)
    # fig, axs = plt.subplots(3,3,figsize=(6, 6))
    # axs=axs.reshape(-1)
    # for i, image_np in enumerate(image[0:9]):
    #     gr = batch_instance['images'][i][None,]
    #     axs[i].imshow(gr.reshape(gr.shape[-2:]),cmap='gray')
    #     axs[i].set_title(batch_instance['labels'][i].item(), fontsize=16)
    #     axs[i].get_xaxis().set_ticks([]), axs[i].get_yaxis().set_ticks([])
    # plt.show()
    # fig.savefig("figures/MNIST_gr_val.png")


    # fig, axs = plt.subplots(3,3,figsize=(6, 6))
    # axs=axs.reshape(-1)
    # for i, image_np in enumerate(image[0:9]):
    #     axs[i].imshow(image_np.detach().cpu().numpy().reshape(batch_instance['images'][i][None,].shape[-2:]),cmap='gray')
    #     axs[i].set_title(batch_instance['labels'][i].item(), fontsize=16)
    #     axs[i].get_xaxis().set_ticks([]), axs[i].get_yaxis().set_ticks([])


    # plt.show()


    # fig.savefig(f"figures/MNIST_recovered_{h_params['opt_name']}_val.png")


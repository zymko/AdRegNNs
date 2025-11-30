import torch
from util import *



# def myoptimizer(model, train_dataloader, criterion,iter_num, Lambda,lr):
#     for data in train_dataloader:
#         inputs, labels, noises = data
#         # initial_guess = get_initial_guess(model.coeff_matrix, noises)
#         # images_recover = initial_guess
#         images_recover = get_initial_guess(model.coeff_matrix, noises)
#         grads_sum = 1
#         counts = 0
#         while criterion <= grads_sum or iter_num <=counts:
#             loss_regularization = L2_regularization(model, images_recover, noises,Lambda)
#             # print(loss_regularization.size())
#             grads =[]
#             for index, guess in enumerate(images_recover):
#                 if not index==model.batch_size-1:
#                     print(loss_regularization[index,0])
#                     loss_regularization[index,0].backward(retain_graph=True)
#                 else:loss_regularization[index,0].backward()
#         # initial_guess_grad = torch.tensor([elem.grad for elem in initial_guess])
#                 # print(images_recover.grad)
#                 grad = images_recover.grad[index]

#                 images_recover[index] -=  lr*grad
#                 grads.append(grad)
#             grads_sum = np.array(grads).sum()
#             counts+=1
#         return images_recover

def inversion_solver(model, lr, batch,Lambda, epochs, AdReg=False):
    # Determine device from model parameters
    device = next(model.parameters()).device

    inputs = batch['images'].to(device)
    labels = batch['labels'].to(device)
    noises = batch['noises'].to(device)
    batch_size = inputs.size()[0]
    
    # get_initial_guess internally handles device consistency
    initial_guess = get_initial_guess(model.coeff_matrix, noises, grad_requires=False)
    # images_recover = initial_guess
    # optimizer = get_optimizer(images_recover,lr=0.001)
    image_recovers=[]
    _noises=[]
    for num in range(batch_size):
        # num=0
        noise = noises[num]
        noise = noise[None,]
        image_recover = initial_guess[num]
        image_recover=image_recover[None,]
        image_recover.requires_grad=True
        # print(image_recover.requires_grad)
        optimizer = get_optimizer(image_recover,lr=lr)
        scheduler = get_scheduler(optimizer, step_size=1000, gamma=0.8)
        
        for epoch in range(epochs):

            optimizer.zero_grad()

            loss = L2_regularization(model, image_recover, noise, Lambda=Lambda, AdReg=AdReg)
            
        

            print('epoch {}, train_loss {}'.format(epoch, loss.item()))

            loss.backward()

            optimizer.step()


            scheduler.step()
        # break
        image_recovers.append(image_recover)
        _noises.append(noise)
    
    return image_recovers, _noises

    

def get_optimizer(parameters,lr):
    # return torch.optim.SGD(self.model.parameters(), lr=self.lr)
    return torch.optim.Adam([parameters],lr=lr)

def get_scheduler(optimizer,step_size,gamma):
        return torch.optim.lr_scheduler.StepLR(optimizer,step_size,gamma)

    

def L2_regularization(model, inputs, noises, Lambda, AdReg):
    # Ensure coeff_matrix is on the same device as inputs
    if model.coeff_matrix.device != inputs.device:
        model.coeff_matrix = model.coeff_matrix.to(inputs.device)
        
    l2_loss = (model.coeff_matrix @ inputs - noises).square().sum(dim=(1,2,3))
    if AdReg:
        loss_reg = Lambda * model.forward(inputs).reshape(-1)
    else:
        loss_reg = Lambda * torch.square(inputs).sum(dim=(1,2,3))
    # print(l2_loss.size(), loss_reg.size())
    return l2_loss + loss_reg
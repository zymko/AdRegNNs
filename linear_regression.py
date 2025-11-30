import numpy as np
import torch
import matplotlib.pyplot as plt

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class QuadraticRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(QuadraticRegression, self).__init__()
        # self.linear = torch.nn.Linear(inputSize, outputSize)
        self.params = {'w1': torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(1))),
                       'w2': torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(1))),
                       'b':torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(1)))}
        
    def forward(self, x):
        # out = self.linear(torch.cat([x, x**2],dim=1))
        out = self.params['w1']*(x**2) + self.params['w2']*x + self.params['b']
        return out  

def loss_func(model, prediction, labels, lambda_l2):
    L2=0
    # for param in model.parameters():
    for param in [model.params['w1'],model.params['w2'], model.params['b']]:
        L2 += param**2
    loss = 0.5*torch.sum((prediction-labels)**2)/3000 + lambda_l2**2 *L2
    return loss

# def loss_regularization(prediction,  Y_train):
#     loss = 0.5*torch.sum((prediction-Y_train)**2)/3000
#     return loss

def loss_regularization(model):
    loss = (1-model.params['w1'])**2 + model.params['w2']**2 + model.params['b']**2
    return loss

# def train(model,epochs,optimizer, scheduler, X_train, Y_noise_train, X_val, Y_noise_val,lambda_l2):
#     for epoch in range(epochs):
#     # Converting inputs and labels to Variable
#         if torch.cuda.is_available():
#         #         inputs = Variable(torch.from_numpy(x_train).cuda())
#         #         labels = Variable(torch.from_numpy(y_train).cuda())
#             inputs = X_train
#             labels = Y_noise_train
#         else:
#             inputs = X_train
#             labels = Y_noise_train
#         # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
#         optimizer.zero_grad()

#         # get output from the model, given the inputs
#         outputs = model(inputs)

#         # get loss for the predicted output
#         loss = loss_func(model, outputs, labels, lambda_l2)
#         # loss = criterion(outputs, labels)
#         # print(loss)
#         print('epoch {}, train_loss {}'.format(epoch, loss.item()))
#         # get gradients w.r.t to parameters
#         with torch.no_grad(): # we don't need gradients in the testing phase
#             if torch.cuda.is_available():
#                 predicted = model(X_val)
#             else:
#                 predicted = model(X_val)
#             loss_val = loss_func(model, predicted, Y_noise_val, lambda_l2)
            
#         print('epoch {}, val_loss {}'.format(epoch, loss_val.item()))
#         loss.backward()

#         # update parameters
#         optimizer.step()
        
#         # print("learning rate of {}th epoch is {}".format(epoch, optimizer.param_groups[0]['lr']))
#         scheduler.step()
        
def train(model,epochs,optimizer, X_train, Y_noise_train, X_val, Y_noise_val,lambda_l2):
    for epoch in range(epochs):
    # Converting inputs and labels to Variable
        if torch.cuda.is_available():
        #         inputs = Variable(torch.from_numpy(x_train).cuda())
        #         labels = Variable(torch.from_numpy(y_train).cuda())
            inputs = X_train
            labels = Y_noise_train
        else:
            inputs = X_train
            labels = Y_noise_train
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = loss_func(model, outputs, labels, lambda_l2)
        # loss = criterion(outputs, labels)
        # print(loss)
        print('epoch {}, train_loss {}'.format(epoch, loss.item()))
        # get gradients w.r.t to parameters
        with torch.no_grad(): # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                predicted = model(X_val)
            else:
                predicted = model(X_val)
            loss_val = loss_func(model, predicted, Y_noise_val, lambda_l2)
            
        print('epoch {}, val_loss {}'.format(epoch, loss_val.item()))
        loss.backward()

        # update parameters
        optimizer.step()
        
        # print("learning rate of {}th epoch is {}".format(epoch, optimizer.param_groups[0]['lr']))
 
        

def train_with_regularization(model,iters,epochs, optimizer,optimizer_re, X_train,Y_noise_train,Y_train, X_val, Y_val,Y_noise_val, lambda_l2):
    for iter in range(iters):

        optimizer_re.zero_grad()

        train(model, epochs, optimizer, X_train, Y_noise_train, X_val, Y_noise_val,lambda_l2)
        # train_re(model, epochs_re,optimizer_re, X_train,Y_train, X_val, Y_val, lambda_l2)
        print(lambda_l2)


        prediction = model(X_train)
        loss_re = loss_regularization(model)
        with torch.no_grad(): # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                predicted = model(X_val)
            else:
                predicted = model(X_val)
            loss_val_re = loss_regularization(model)
            
        print('iter {}, train_loss_regularization {}'.format(iter, loss_re.item()))
        print('iter {}, val_loss_regularization {}'.format(iter, loss_val_re.item()))
        loss_re.backward()
        optimizer_re.step()

# def train_re(model, epochs_re,optimizer_re, X_train,Y_train, X_val, Y_val, lambda_l2):
#     for epoch_re in range(epochs_re):
#         optimizer_re.zero_grad()
        
#         print(lambda_l2)

#         prediction = model(X_train)
#         loss_re = loss_regularization(prediction, Y_train)
#         with torch.no_grad(): # we don't need gradients in the testing phase
#             if torch.cuda.is_available():
#                 predicted = model(X_val)
#             else:
#                 predicted = model(X_val)
#             loss_val_re = loss_regularization(predicted, Y_val)
            
#         print('iter {}, train_loss_regularization {}'.format(iter, loss_re.item()))
#         print('iter {}, val_loss_regularization {}'.format(iter, loss_val_re.item()))
#         loss_re.backward()
#         optimizer_re.step()        




def predicted_plotting(X_train, Y_train,model=None):
    if model:
        with torch.no_grad(): # we don't need gradients in the testing phase
            if torch.cuda.is_available():
        #         predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
                predicted = model(X_train).cpu().data.numpy()
            else:
        #         predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
                predicted = model(X_train)
            print(loss_regularization(model))
            # predicted = predicted.data.numpy()

    plt.clf()
    plt.plot(X_train, Y_train, 'go', label='True data', alpha=0.5)
    if model:
        plt.plot(X_train, predicted, 'ro', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()
    if model:
        return predicted
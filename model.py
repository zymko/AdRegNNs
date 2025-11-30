import time
import torch
import torch.nn as nn
from util import *
import torch.nn.functional as F
import wandb as _wandb

class Conv(nn.Module):
    # a stack of convolutional layers for image-to-image functionality
    def __init__(self, h_params): 
        super(Conv, self).__init__()
        self.input_channels = h_params['input_channels']
        self.epochs = h_params['epochs']
        self.Lambda = h_params['lambda']
        self.coeff_matrix = h_params['coeff_matrix']
        self.mu = h_params['mu']
        self.lr = h_params['lr']
        self.step_size = h_params['step_size']
        self.gamma = h_params['gamma']
        self.batch_size = h_params['batch_size']

        self.model = nn.Sequential(
                                   nn.Conv2d(h_params['input_channels'], out_channels=16, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(16, out_channels=32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, out_channels=32, kernel_size=5, stride=2,padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, out_channels=64, kernel_size=5, stride=2, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, out_channels=64, kernel_size=5, stride=2, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, out_channels=128, kernel_size=5, stride=2,  padding=2),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(4*128, 1)
        )
        self.opt_name = h_params['opt_name']
        
        # Apply Xavier initialization
        # self.model.apply(self._init_weights)
        
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)

    def train_regularization(self, train_dataloader, val_dataloader, device,run_name=None, seed=42):
        # Set seed at the start of training for reproducibility
        set_seed(seed)

        # --- Weights & Biases logging ---
        # 每次调用 train_regularization 都创建一个新的 run
        base_name = run_name if run_name is not None else getattr(self, "run_name", "conv_regularizer")
        run_name = f"{base_name}-{int(time.time())}"
        wandb_run = _wandb.init(project="adregnets", name=run_name, reinit=True)
        _wandb.config.update(
            {"epochs": self.epochs, "lr": self.optimizer.param_groups[0]["lr"]},
            allow_val_change=True,
        )


        for epoch in range(self.epochs):
        # Converting inputs and labels to Variable
            for data_train in train_dataloader:
                if torch.cuda.is_available():
                    inputs_train, labels_train, noises_train = data_train
                else:
                    inputs_train, labels_train, noises_train = data_train

                # move to target device and ensure float32 for image-like tensors
                inputs_train = inputs_train.to(device=device, dtype=torch.float32)
                noises_train = noises_train.to(device=device, dtype=torch.float32)
                labels_train = labels_train.to(device=device)
                   
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                self.optimizer.zero_grad()

                # get output from the model, given the inputs
                # outputs = self.model(inputs.float())
                outputs_gr_train = self.forward(inputs_train)
                
                initial_guess_train = get_initial_guess(self.coeff_matrix, noises_train)
                outputs_no_train = self.forward(initial_guess_train)
                
                init_gr_comb_train = samples_generate(ground_truth=inputs_train, 
                                                      initial_guess=initial_guess_train)
                                                      
                outputs_ini_train = self.forward(init_gr_comb_train)

                # get loss for the predicted output
                loss = self.loss_func(outputs_gr_train, 
                                      outputs_no_train, 
                                      outputs_ini_train, 
                                      init_gr_comb_train)

                print('epoch {}, train_loss {}'.format(epoch, loss.item()))
                # get gradients w.r.t to parameters
                with torch.no_grad(): # we don't need gradients in the validating phase
                    for data in val_dataloader:
                        if torch.cuda.is_available():
                            inputs_val, labels_val, noises_val = data
                        else:
                            inputs_val, labels_val, noises_val = data

                        inputs_val = inputs_val.to(device=device, dtype=torch.float32)
                        noises_val = noises_val.to(device=device, dtype=torch.float32)
                        labels_val = labels_val.to(device=device)

                        outputs_gr_val = self.forward(inputs_val)
                        
                        initial_guess_val = get_initial_guess(self.coeff_matrix, noises_val)
                        init_gr_comb_val = samples_generate(ground_truth=inputs_val, 
                                                        initial_guess=initial_guess_val)
                        outputs_no_val = self.forward(initial_guess_val)
                        outputs_ini_val = self.forward(init_gr_comb_val)

                        loss_val = self.loss_val(outputs_gr_val, 
                                                    outputs_no_val, 
                                                    outputs_ini_val, 
                                                    init_gr_comb_val)
                        break
                    
                print('epoch {}, val_loss {}'.format(epoch, loss_val.item()))
                # wandb logging (per-batch)
                if wandb_run is not None:
                    _wandb.log({"train/loss": float(loss.item()),
                                "val/loss": float(loss_val.item()),
                                "epoch": int(epoch)})


                loss.backward()

                # Calculate total gradient norm across all parameters
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                print(f"  > Total Grad norm: {total_norm:.6e}")

                if wandb_run is not None:
                    _wandb.log({"train/total_grad_norm": total_norm}, commit=False)

                # update parameters
                self.optimizer.step()
                
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.clamp_(-0.01, 0.01)
                
                
                print("learning rate of {}th epoch is {}".format(epoch, self.optimizer.param_groups[0]['lr']))
                if self.scheduler:
                    self.scheduler.step()
        # 结束当前 run，保证下次调用会新建 run
        if wandb_run is not None:
            wandb_run.finish()
    
    def loss_val(self, outputs_gr, outputs_no, outputs_ini_val, init_gr_comb_val):
        Lipschitz_contraints=0
        # Lipschitz_contraints=self.loss_grad(outputs_ini_val, init_gr_comb_val)
        return (outputs_gr.sum() - outputs_no.sum())/self.batch_size + self.mu * Lipschitz_contraints


    def loss_grad(self,outputs_ini, init_gr_comb):
        Lipschitz_contraints=0
        for i in range(self.batch_size):
            # outputs_ini[i].backward(retain_graph=True)
            if not i==self.batch_size-1:
                outputs_ini[i].backward(retain_graph=True)
            else:outputs_ini[i].backward()
        # initial_guess_grad = torch.tensor([elem.grad for elem in initial_guess])
            init_gr_comb_train_grad = init_gr_comb.grad[i]
            # print((init_gr_comb_train_grad**2).sum())
            Lipschitz_contraints += (F.relu((init_gr_comb_train_grad**2).sum() -1))**2
        return Lipschitz_contraints/self.batch_size

    def loss_func(self, outputs_gr, outputs_no, outputs_ini_train, init_gr_comb_train):
        Lipschitz_contraints=0
        # Lipschitz_contraints=self.loss_grad(outputs_ini_train, init_gr_comb_train)
        # print(Lipschitz_contraints)
        # for i in range(self.batch_size):
        #     # outputs_ini[i].backward(retain_graph=True)
        #     if not i==self.batch_size-1:
        #         outputs_ini[i].backward(retain_graph=True)
        #     else:outputs_ini[i].backward()
        # # initial_guess_grad = torch.tensor([elem.grad for elem in initial_guess])
        #     init_gr_comb_train_grad = init_gr_comb.grad[i]
        #     # print(init_gr_comb_train_grad)
        #     Lipschitz_contraints += (F.relu((init_gr_comb_train_grad**2).sum() -1))**2
        return (outputs_gr.sum() - outputs_no.sum())/self.batch_size + self.mu * Lipschitz_contraints

    def get_optimizer(self):
        if self.opt_name == "Adam":
            return torch.optim.Adam(self.model.parameters(),lr=self.lr)
        elif self.opt_name == "RMSprop":
            return torch.optim.RMSprop(self.model.parameters(),lr=self.lr)
        elif self.opt_name == "SGD":
            return torch.optim.SGD(self.model.parameters(),lr=self.lr)
        elif self.opt_name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.opt_name == "mSGD":
            return torch.optim.SGD(self.model.parameters(),lr=self.lr, momentum=0.8)
        else:
            raise ValueError(f"Invalid optimizer name: {self.opt_name}")

    def get_scheduler(self):
        # return torch.optim.lr_scheduler.StepLR(self.optimizer,self.step_size,self.gamma)
        return False



class Conv1d(nn.Module):
    # a stack of convolutional layers for image-to-image functionality
    def __init__(self, h_params): 
        super(Conv1d, self).__init__()
        self.input_channels = h_params['input_channels']
        self.epochs = h_params['epochs']
        self.Lambda = h_params['lambda']
        self.coeff_matrix = h_params['coeff_matrix']
        self.mu = h_params['mu']
        self.lr = h_params['lr']
        self.step_size = h_params['step_size']
        self.gamma = h_params['gamma']
        self.batch_size = h_params['batch_size']

        self.model = nn.Sequential(
                                   nn.Conv2d(h_params['input_channels'], out_channels=16, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(16, out_channels=32, kernel_size=5, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, out_channels=32, kernel_size=5, stride=2,padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, out_channels=64, kernel_size=5, stride=2, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, out_channels=64, kernel_size=5, stride=2, padding=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, out_channels=128, kernel_size=5, stride=2,  padding=2),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(4*128, 1)
        )
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()


    def forward(self, x):
        return self.model(x)

    def train_regularization(self, train_dataloader, val_dataloader):
        for epoch in range(self.epochs):
        # Converting inputs and labels to Variable
            for data_train in train_dataloader:
                if torch.cuda.is_available():
                    inputs_train, labels_train, noises_train = data_train
   
                else:
                    inputs_train, labels_train, noises_train = data_train
                   
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                self.optimizer.zero_grad()

                # get output from the model, given the inputs
                # outputs = self.model(inputs.float())
                outputs_gr_train = self.forward(inputs_train)
                
                initial_guess_train = get_initial_guess(self.coeff_matrix, noises_train)
                outputs_no_train = self.forward(initial_guess_train)
                
                init_gr_comb_train = samples_generate(ground_truth=inputs_train, 
                                                      initial_guess=initial_guess_train)
                                                      
                outputs_ini_train = self.forward(init_gr_comb_train)

                # get loss for the predicted output
                loss = self.loss_func(outputs_gr_train, 
                                      outputs_no_train, 
                                      outputs_ini_train, 
                                      init_gr_comb_train)

                print('epoch {}, train_loss {}'.format(epoch, loss.item()))
                # get gradients w.r.t to parameters
                with torch.no_grad(): # we don't need gradients in the validating phase
                    for data in val_dataloader:
                        if torch.cuda.is_available():
                            inputs_val, labels_val, noises_val = data
        
                        else:
                            inputs_val, labels_val, noises_val = data

                        outputs_gr_val = self.forward(inputs_val)
                        
                        initial_guess_val = get_initial_guess(self.coeff_matrix, noises_val)
                        init_gr_comb_val = samples_generate(ground_truth=inputs_val, 
                                                        initial_guess=initial_guess_val)
                        outputs_no_val = self.forward(initial_guess_val)
                        outputs_ini_val = self.forward(init_gr_comb_val)

                        loss_val = self.loss_val(outputs_gr_val, 
                                                    outputs_no_val, 
                                                    outputs_ini_val, 
                                                    init_gr_comb_val)
                        break
                    
                print('epoch {}, val_loss {}'.format(epoch, loss_val.item()))

                loss.backward()

                # update parameters
                self.optimizer.step()
                
                # with torch.no_grad():
                #     for param in self.model.parameters():
                #         param.clamp_(-0.01, 0.01)
                
                
                print("learning rate of {}th epoch is {}".format(epoch, self.optimizer.param_groups[0]['lr']))
                if self.scheduler:
                    self.scheduler.step()
    
    def loss_val(self, outputs_gr, outputs_no, outputs_ini_val, init_gr_comb_val):
        Lipschitz_contraints=0
        # Lipschitz_contraints=self.loss_grad(outputs_ini_val, init_gr_comb_val)
        return (outputs_gr.sum() - outputs_no.sum())/self.batch_size + self.mu * Lipschitz_contraints


    def loss_grad(self,outputs_ini, init_gr_comb):
        Lipschitz_contraints=0
        for i in range(self.batch_size):
            # outputs_ini[i].backward(retain_graph=True)
            if not i==self.batch_size-1:
                outputs_ini[i].backward(retain_graph=True)
            else:outputs_ini[i].backward()
        # initial_guess_grad = torch.tensor([elem.grad for elem in initial_guess])
            init_gr_comb_train_grad = init_gr_comb.grad[i]
            # print((init_gr_comb_train_grad**2).sum())
            Lipschitz_contraints += (F.relu((init_gr_comb_train_grad**2).sum() -1))**2
        return Lipschitz_contraints/self.batch_size

    def loss_func(self, outputs_gr, outputs_no, outputs_ini_train, init_gr_comb_train):
        Lipschitz_contraints=0
        # Lipschitz_contraints=self.loss_grad(outputs_ini_train, init_gr_comb_train)
        # print(Lipschitz_contraints)
        # for i in range(self.batch_size):
        #     # outputs_ini[i].backward(retain_graph=True)
        #     if not i==self.batch_size-1:
        #         outputs_ini[i].backward(retain_graph=True)
        #     else:outputs_ini[i].backward()
        # # initial_guess_grad = torch.tensor([elem.grad for elem in initial_guess])
        #     init_gr_comb_train_grad = init_gr_comb.grad[i]
        #     # print(init_gr_comb_train_grad)
        #     Lipschitz_contraints += (F.relu((init_gr_comb_train_grad**2).sum() -1))**2
        return (outputs_gr.sum() - outputs_no.sum())/self.batch_size + self.mu * Lipschitz_contraints

    def get_optimizer(self):
        if self.opt_name == "Adam":
            return torch.optim.Adam(self.model.parameters(),lr=self.lr)
        elif self.opt_name == "RMSprop":
            return torch.optim.RMSprop(self.model.parameters(),lr=self.lr)
        elif self.opt_name == "SGD":
            return torch.optim.SGD(self.model.parameters(),lr=self.lr)
        elif self.opt_name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.opt_name == "mSGD":
            return torch.optim.SGD(self.model.parameters(),lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Invalid optimizer name: {self.opt_name}")


    def get_scheduler(self):
        # return torch.optim.lr_scheduler.StepLR(self.optimizer,self.step_size,self.gamma)
        return False

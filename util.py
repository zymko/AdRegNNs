import random
import numpy as np
import torch
import copy

def set_seed(seed=123):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For exact reproducibility (may be slower)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Set seed immediately upon import
set_seed(123)

def get_initial_guess(coeff_matrix, noises, grad_requires=True):
    """Solve min_x || coeff_matrix @ x - noises ||_2 in least-squares sense.

    Uses torch.linalg.lstsq on CPU to avoid backends (e.g. MPS) where this op
    is not implemented, then moves result back to the input device.
    """
    # Preserve original device/dtype of noises
    noise_dev = noises.device
    noise_dtype = noises.dtype

    # Ensure coefficient matrix and RHS are CPU tensors with compatible shapes
    coef = torch.as_tensor(copy.copy(coeff_matrix), dtype=noise_dtype, device="cpu")
    coef = coef.unsqueeze(0).unsqueeze(1)  # match original [1,1,H,W] broadcasting
    rhs = noises.detach().to("cpu")

    # Solve least-squares on CPU; rcond ~ 1e-5 * largest singular value
    sol = torch.linalg.lstsq(coef, rhs, rcond=1e-5).solution

    # Move back to original device and set gradient requirement
    init_guess = sol.to(device=noise_dev)
    init_guess.requires_grad_(grad_requires)
    return init_guess

def samples_generate(ground_truth, initial_guess):
    """Interpolate between ground_truth and initial_guess on the *same device*.

    samples = eps * ground_truth + (1 - eps) * initial_guess
    where eps ~ U(0,1) and lives on the same device/dtype as ground_truth.
    """
    device = ground_truth.device
    dtype = ground_truth.dtype
    eps = torch.rand(1, device=device, dtype=dtype)
    samples = eps * ground_truth + (1.0 - eps) * initial_guess.to(device=device, dtype=dtype)
    if not samples.requires_grad:
        samples.requires_grad = True
    return samples

def get_batch_instance(dataloader):
    instance_batch={}
    for images, labels, noises in dataloader:  
        instance_batch['image']=images
        instance_batch['label']=labels
        instance_batch['noise']=noises
        break
    
    return instance_batch

def get_instance(dataloader):
    instance={}
    for images, labels, noises in dataloader:  
        instance['image']=images[0]
        instance['label']=labels[0]
        instance['noise']=noises[0]
        break
    
    return instance


def Frobenius_distance(batch_instance, recovered_images, batch_size): 
    loss=0
    for i in range(batch_size):
        gr = batch_instance['images'][i][None,]
        recovered_image = recovered_images[i]
        loss+=torch.sqrt(torch.square(gr - recovered_image).sum(dim=(1,2,3)))
    return loss/batch_size

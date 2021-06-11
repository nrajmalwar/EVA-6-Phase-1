from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, l1_reg=False):
    model.train()
    pbar = tqdm(train_loader)
    
    train_losses = []
    train_acc = []
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)

      # Calculate loss
      if l1_reg:
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
          if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
        
        loss = F.nll_loss(y_pred, target) + 10e-4 * L1_reg  
      else :  
        loss = F.nll_loss(y_pred, target)
      
      train_losses.append(loss)

      # Backpropagation
      loss.backward()
      optimizer.step()

      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)

      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      train_acc.append(100*correct/processed)

    return train_acc, train_losses
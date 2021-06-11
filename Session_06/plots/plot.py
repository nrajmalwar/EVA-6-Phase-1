import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def display_misclassified_images(model, data, device, num=10):
  print("\n********* Misclassified Images **************\n")
  model.eval()

  cuda = torch.cuda.is_available()
  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=len(data), num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=len(test))
  # test dataloader
  data_loader = DataLoader(data, **dataloader_args)

  with torch.no_grad():
      for data, target in data_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          # Get the indexes of images that are incorrectly classified
          indexes = (pred.view(-1,) != target.view(-1,)).nonzero()
          # Plot the misclassified images
          rows = np.ceil(len(indexes)/2)
          fig = plt.figure(figsize=(5, 18))
          for i, idx in enumerate(indexes[:num]):
              ax = fig.add_subplot(5, 2, i+1)
              ax.imshow(data[idx].cpu().numpy().squeeze(), cmap='gray_r')
              ax.set_title(f"Target = {target[idx].item()} \n Predicted = {pred[idx].item()}")
          
          plt.show()
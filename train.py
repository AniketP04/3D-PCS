import numpy as np 
import time 
import random 

import torch 

import matplotlib.pyplot as plt
from pathlib import Path

from modelnet10 import PointCloudDataset, default_transforms
from loss import pointnetloss
from segmentation import PointNet


def train(model, optimizer, dataloaders, epochs, plot=False): 
    """ Train the model """
    train_losses, train_accuracies = [],[]
    valid_losses, valid_accuracies = [],[]
    best_acc = 0
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-'*70)
        
        train_loss, train_acc = 0,0
        val_loss, val_acc = 0,0 
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                print('Training ...')
                t_start = time.time()
                model.train()        
            else: 
                print('Validation ...')
                t_start = time.time()
                model.eval()
                
            running_loss = 0
            total, correct = 0,0

            for step, data in tqdm(enumerate(dataloaders[phase])):
                if phase == 'train':
                    optimizer.zero_grad()
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
                    n_points = outputs.size()[1]
                    loss = pointnetloss(outputs.transpose(1,2), labels.unsqueeze(1).repeat(1,n_points), m3x3, m64x64)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() 
                    _, predicted = torch.max(outputs.transpose(1,2).data, 1)
                    total += labels.size(0) * n_points 
                    correct += (predicted == labels.unsqueeze(1).repeat(1,n_points)).sum().item()

                    if (step % 20 == 0) & (step != 0):
                        print("Batch {}/{} - Loss : {}".format(step, len(dataloaders[phase]), running_loss/step))
                    
                else: 
                    with torch.no_grad(): 
                        inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                        outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
                        n_points = outputs.size()[1]
                        loss = pointnetloss(outputs.transpose(1,2), labels.unsqueeze(1).repeat(1,n_points), m3x3, m64x64)

                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.transpose(1,2).data, 1)
                        total += labels.size(0) * n_points 
                        correct += (predicted == labels.unsqueeze(1).repeat(1,n_points)).sum().item()

                        if (step % 20 == 0) & (step != 0): 
                            print('Batch {}/{} -- Loss : {}'.format(step, len(dataloaders[phase]), running_loss/step))
        
            if phase == 'train':
                train_acc = 100. * correct / total
                train_loss = running_loss/len(dataloaders[phase])

                train_losses.append(train_loss)
                train_accuracies.append(train_acc) 
                print('  Average training loss : {}'.format(train_loss))
                print('  Average training accuracy : {} %'.format(train_acc))

                t_end = time.time() 
                print('  Elapsed Time : {}'.format(t_end-t_start))
                
            else: 
                val_acc = 100. * correct / total
                val_loss = running_loss/len(dataloaders[phase])

                valid_losses.append(val_loss)
                valid_accuracies.append(val_acc)
                print('  Average validation loss : {}'.format(val_loss))
                print('  Average validation accuracy : {} %'.format(val_acc))

                t_end = time.time() 
                print('  Elapsed Time : {}'.format(t_end-t_start))

                # save the model with best accuracy  
                if best_acc < val_acc: 
                    best_acc = val_acc 
                    torch.save(model.state_dict(), './models/model.pt')
                    print('model checkpoint saved !') 
        
    if plot == True: 
        plt.subplot(2,1,1)
        plt.plot(np.arange(epochs), train_losses, 'r')
        plt.plot(np.arange(epochs), valid_losses, 'b')
        plt.legend(['Train Loss','Validation Loss'])
        plt.show()
        
        plt.subplot(2,1,2)
        plt.plot(np.arange(epochs), train_accuracies, 'r')
        plt.plot(np.arange(epochs), valid_accuracies, 'b')
        plt.legend(['Train Accuracy','Validation Accuracy'])
        plt.show()
        
    print("Training Complete")
    return model 


if __name__ == "__main__":
    random.seed = 42
    path = Path("../input/Dataset")

    train_dataset = PointCloudDataset(path, start = 0, end = 90)
    valid_dataset = PointCloudDataset(path, start = 90, end = 120)
    datasets = {"train" : train_dataset, "validation" : valid_dataset}
    dataloaders = {x : torch.utils.data.DataLoader(datasets[x], batch_size=32, shuffle=True) for x in ['train', 'validation']}

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pointnet = PointNet().to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)

    train(pointnet, optimizer, dataloaders, epochs=100, plot=False)

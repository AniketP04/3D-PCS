import numpy as np 
import os
import time
from tqdm import tqdm
from pathlib import Path
import torch 
from dataloader import *
from model import PointNetSegmentation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

labels = dataloader.labels

def compute_stats(true_labels, pred_labels):
  unk     = np.count_nonzero(true_labels == 0)
  trav    = np.count_nonzero(true_labels == 1)
  nontrav = np.count_nonzero(true_labels == 2)

  total_predictions = labels.shape[1]*labels.shape[0]
  correct = (true_labels == pred_labels).sum().item()

  return correct, total_predictions

if __name__ == '__main__':
    pointnet = pt.PointNetSeg()
    model_path = os.path.join('', "pointnetmodel.yml")
    pointnet.load_state_dict(torch.load(model_path))
    pointnet.to(device)
    pointnet.eval()
    test_ds = dataloader.PointCloudData('dataset', start=120, end=150)
    test_loader = dataloader.DataLoader(dataset=test_ds, batch_size=1, shuffle=False)
    total_correct_predictions = total_predictions = 0

    start = time.time()

    for i, data in tqdm(enumerate(test_loader, 0)):
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, predicted = torch.max(outputs.data, 1)

        remapped_pred = dataloader.remap_to_bgr(predicted[0].cpu().numpy(), dataloader.remap_color_scheme)
        np_pointcloud = inputs[0].cpu().numpy()

        ground_truth_labels = labels.cpu()
        predicted_labels = predicted.cpu()
        correct, total = compute_stats(ground_truth_labels, predicted_labels)

        total_correct_predictions += correct
        total_predictions += total

    end = time.time()

    print()
    print()

    test_acc = 100. * total_correct_predictions / total_predictions
    tot_latency = end-start
    avg_latency = tot_latency / len(test_loader.dataset)

    print('Test accuracy:', test_acc, "%")
    print('total time:', tot_latency, " [s]")
    print('avg time  :', avg_latency, " [s]")


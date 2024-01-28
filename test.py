import numpy as np 
import random
import plotly.graph_objects as go

import torch 

from pathlib import Path

from modelnet10 import PointCloudDataset, default_transforms
from segmentation import PointNet


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig


def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


def evaluate_miou(model, dataloader):
    iou_scores = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print('Batch [%4d / %4d]' % (i+1, len(dataloader)))
                    
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            outputs, _, _ = model(inputs.transpose(1,2))
            n_points = outputs.size()[1]
            _, preds = torch.max(outputs.transpose(1,2).data, 1)

            classes = np.unique(labels.cpu().numpy())
            miou = []
            for cls in classes:
                I = np.sum(np.logical_and((labels == cls).view(-1,1).repeat(1,n_points).cpu().numpy(), (preds == cls).cpu().numpy()))
                U = np.sum(np.logical_or((labels == cls).view(-1,1).repeat(1,n_points).cpu().numpy(), (preds == cls).cpu().numpy()))
                iou = I/float(U)
                miou.append(iou)
            
            iou_scores.append(np.mean(miou))

    return iou_scores


if __name__ == "__main__":
    random.seed = 42
    path = Path("../input/Dataset")

    test_dataset = PointCloudDataset(path, start = 120, end = 150)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pointnet = PointNet().to(device)
    pointnet.load_state_dict(torch.load('./models/model.pth'))
    pointnet.eval()

    iou_scores = evaluate_miou(pointnet, test_dataloader)
    print("mIoU : {}".format(np.mean(iou_scores)))

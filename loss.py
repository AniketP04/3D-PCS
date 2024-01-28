import torch

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    batchsize = outputs.size(0)
    I = torch.eye(64, requires_grad=True).repeat(batchsize,1,1)
    if outputs.is_cuda:
        I = I.cuda()
    loss = criterion(outputs, labels) + alpha * torch.mean(torch.norm(I - torch.bmm(m64x64, m64x64.transpose(1,2)), dim=(1,2))) 
    return loss
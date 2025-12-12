import torch
import numpy as np

def evaluate(model, data_loader, loss_fun, device='cuda'):
    model.eval()
    loss_all = 0
    total = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            if isinstance(output, tuple):
                loss = loss_fun(output[1], target)
                loss = loss #+ 0.01*output[0]
            else:
                loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            if isinstance(output, tuple):
                pred = output[1].data.max(1)[1]
            else:
                pred = output.data.max(1)[1]
            accuracy += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), accuracy/total
    
def evaluate_mask(model, data_loader, loss_fun, device='cuda', class_mask=None):
    model.eval()
    loss_all = 0
    total = 0
    accuracy = 0
    len_data_loader = 0
    with torch.no_grad():
        for i in range(len(data_loader)):
            mask = class_mask[i]
            for data, target in data_loader[i]:
                data = data.to(device).float()
                target = target.to(device).long()
                output = model(data)
                if isinstance(output, tuple):
                    reduce_sim, logits = output
                    not_mask = np.setdiff1d(np.arange(model.num_classes), mask)
                    not_mask = torch.tensor(not_mask).to(device).long()
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    loss = loss_fun(logits, target)
                    loss = loss #+ 0.01*output[0]
                else:
                    loss = loss_fun(output, target)
                loss_all += loss.item()
                total += target.size(0)
                if isinstance(output, tuple):
                    pred = logits.data.max(1)[1]
                else:
                    pred = output.data.max(1)[1]
                accuracy += pred.eq(target.view(-1)).sum().item()
            len_data_loader += len(data_loader[i])
    return loss_all/len_data_loader, accuracy/total

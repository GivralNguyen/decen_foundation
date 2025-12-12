import torch
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(model, data_loader, optimizer, loss_fun, device='cuda', reduce_sim_scalar=0.01, mask=None):
    model.train()
    loss_all = 0
    total = 0
    accuracy = 0
    current_lr = get_lr(optimizer)
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        if isinstance(output, tuple):
            reduce_sim, logits = output
            if mask is not None:
                not_mask = np.setdiff1d(np.arange(model.num_classes), mask)
                not_mask = torch.tensor(not_mask).to(device).long()
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            loss = loss_fun(logits, target)
            loss = loss + reduce_sim_scalar*reduce_sim
        else:
            if mask is not None:
                not_mask = np.setdiff1d(np.arange(model.num_classes), mask)
                not_mask = torch.tensor(not_mask).to(device).long()
                output = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
            loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        if isinstance(output, tuple):
            pred = logits.data.max(1)[1]
        else:
            pred = output.data.max(1)[1]
        accuracy += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        #if isinstance(output, tuple):
            #torch.nn.utils.clip_grad_norm_(filter(lambda p : p.requires_grad, server_model.parameters()), 0.0)
        optimizer.step()
    return current_lr, loss_all/len(data_loader), accuracy/total



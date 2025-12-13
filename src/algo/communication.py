import torch
import torch.nn as nn
import copy
import numpy as np
from algo.nonparametric_aggregation import NonparametricAgg
from algo.nonparametric_aggregation import DecenNonparametricAgg

def communication(server_model, models, client_weights, aggregation_method='parametric', total_n_clietns=100, nonpara_hidden=128, device='cuda'):
    # Centralized FL aggregation step: sync all clients to the server using either
    # parametric (FedAvg on all trainable keys) or nonparametric agg.
    client_num = len(models)
    sum_weights = torch.tensor(np.sum(client_weights), dtype=torch.float, device=device)
    
    if aggregation_method == 'nonparametric':
        for key in server_model.trainable_keys:
            if 'prompt' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                temp = torch.div(temp, sum_weights)
                with torch.no_grad():
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(temp)
            else:
                temp = list()
                # prompt_length = server_model.state_dict()[key].shape[1]
                prompt_dim = server_model.state_dict()[key].shape[-1]
                if hasattr(server_model, 'trained_prompts_checklist'):
                    union_prompts_checklist = torch.zeros(server_model.state_dict()[key].shape[0],dtype=torch.int)
                    nonzero_index = torch.nonzero(server_model.trained_prompts_checklist).flatten()
                    union_prompts_checklist[nonzero_index] = 1
                    for client_idx in range(client_num):
                        nonzero_index = torch.nonzero(models[client_idx].trained_prompts_checklist).flatten()
                        union_prompts_checklist[nonzero_index] = 1
                    for client_idx in range(client_num):
                        temp.append(copy.deepcopy(models[client_idx].state_dict()[key][torch.nonzero(union_prompts_checklist).flatten()]))
                else:
                    for client_idx in range(client_num):
                        temp.append(copy.deepcopy(models[client_idx].state_dict()[key].squeeze()))
                temp = torch.stack(temp, dim=0) # temp is n_clients x prompt_length x 768
                agg = NonparametricAgg(prompt_dim, n_hidden=nonpara_hidden).to(device)
                temp = agg(temp)
                print(temp.shape)
                del agg
                with torch.no_grad():
                    if hasattr(server_model, 'trained_prompts_checklist'):
                        server_model.pool.prompt = nn.Parameter(temp, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].pool.prompt = nn.Parameter(temp, requires_grad=True)
                    else:
                        server_model.prompt_embeddings = nn.Parameter(temp, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].prompt_embeddings = nn.Parameter(temp, requires_grad=True)
    elif aggregation_method == 'parametric':
        for key in server_model.trainable_keys:   #server_model.state_dict().keys():
            '''if 'num_batches_tracked' in key:
                server_model.state_dict()[key].dataa.copy_(models[0].state_dict()[key])
            else:'''
            temp = torch.zeros_like(server_model.state_dict()[key])
            for client_idx in range(client_num):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            temp = torch.div(temp, sum_weights)
            with torch.no_grad():
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    
    return server_model, models


def decen_communication(models, G, client_weights=None,
                        aggregation_method='parametric', nonpara_hidden=128, device='cuda'):
    """
    # Decentralized FL aggregation step: each client aggregates only within its local
    # neighborhood in graph G (neighbors + self). Uses local FedAvg for parametric,
    # and DecenNonparametricAgg for prompt params when aggregation_method='nonparametric_aggregation'.
    ...

    """
    n_clients = len(models)
    assert set(G.nodes) == set(range(n_clients)), "Graph nodes must be 0..n_clients-1"

    # Default weights: uniform
    if client_weights is None:
        client_weights = [1.0 for _ in range(n_clients)]

    # Get device from first model
    first_state = models[0].state_dict()
    example_key = next(iter(first_state.keys()))
    dev = first_state[example_key].device

    # Container for new parameters (synchronous update)
    new_params = [dict() for _ in range(n_clients)]

    if aggregation_method == 'nonparametric_aggregation':
       
    # ---------------------------------------------------
    # Decentralized nonparametric_aggregation
    # (mimics your centralized nonparametric_aggregation
    #  but restricted to each node's neighborhood)
    # ---------------------------------------------------

        for i in range(n_clients):
            neighbors = list(G.neighbors(i))
            local_group = neighbors + [i]
            # print(i,neighbors)
            local_sum_w = sum(float(client_weights[j]) for j in local_group)

            for key in models[i].trainable_keys:
                # Non-prompt: local FedAvg
                if 'prompt' not in key:
                    temp = torch.zeros_like(models[i].state_dict()[key], device=dev)
                    for j in local_group:
                        w_j = float(client_weights[j])
                        temp += w_j * models[j].state_dict()[key].to(dev)
                    temp = temp / local_sum_w
                    new_params[i][key] = temp

                # Prompt keys: use NonparametricAgg over local neighborhood
                else:
                    temp_list = []
                    prompt_dim = models[i].state_dict()[key].shape[-1]

                    if hasattr(models[i], 'trained_prompts_checklist'):
                        # 1) Figure out pool sizes in the local neighborhood
                        pool_sizes = {
                            j: int(models[j].state_dict()[key].shape[0])
                            for j in local_group
                        }
                        max_n_prompts = max(pool_sizes.values())

                        # 2) Union checklist over the *max* pool size
                        #    (on the correct device)
                        union_prompts_checklist = torch.zeros(
                            max_n_prompts, dtype=torch.int, device=dev
                        )

                        for j in local_group:
                            checklist = models[j].trained_prompts_checklist

                            # In case checklist is not exactly same length as pool, be safe:
                            n_check = min(checklist.numel(), max_n_prompts)

                            if n_check == 0:
                                continue

                            # Active indices for client j, clipped to the union length
                            nz = torch.nonzero(checklist[:n_check]).flatten()
                            if nz.numel() == 0:
                                continue

                            union_prompts_checklist[nz] = 1

                        # 3) Union of active prompt indices across neighborhood
                        indices = torch.nonzero(union_prompts_checklist).flatten()

                        if indices.numel() == 0:
                            # No active prompts in this neighborhood -> nothing to aggregate
                            temp_list = []  # or just leave it empty
                        else:
                            # 4) For each client j, keep only indices that exist in its pool
                            for j in local_group:
                                param = models[j].state_dict()[key].to(dev)   # [n_prompts_j, ...]
                                n_j = param.shape[0]

                                # Only keep indices that are valid for this client
                                valid_idx = indices[indices < n_j]

                                if valid_idx.numel() == 0:
                                    # This client has no prompts corresponding to the union indices
                                    continue

                                temp_list.append(
                                    copy.deepcopy(param[valid_idx])
                                )

                    else:
                        # Use all prompts
                        for j in local_group:
                            temp_list.append(
                                copy.deepcopy(
                                    models[j].state_dict()[key].squeeze().to(dev)
                                )
                            )

                    # temp_stack = torch.stack(temp_list, dim=0)  # [local_clients, prompt_len(or subset), prompt_dim]
                    agg = DecenNonparametricAgg(prompt_dim, n_hidden=nonpara_hidden).to(device)
                    agg_out = agg(temp_list)  # should be [prompt_len(or subset), prompt_dim]
                    # print(agg_out.shape)
                    del agg

                    new_params[i][key] = agg_out

        # Assign back for nonparametric_aggregation
        with torch.no_grad():
            for i in range(n_clients):
                # First copy non-prompt params directly
                for key in models[i].trainable_keys:
                    if 'prompt' not in key:
                        models[i].state_dict()[key].data.copy_(new_params[i][key])

                # Then handle prompts like in your original function
                for key in models[i].trainable_keys:
                    if 'prompt' in key:
                        temp = new_params[i][key]
                        if hasattr(models[i], 'trained_prompts_checklist'):
                            models[i].pool.prompt = nn.Parameter(temp, requires_grad=True)
                        else:
                            models[i].prompt_embeddings = nn.Parameter(temp, requires_grad=True)
                        # Only need to set once per model
                        break
    else:
         # -----------------------------
        # Decentralized FedAvg
        # -----------------------------
        for i in range(n_clients):
            neighbors = list(G.neighbors(i))
            local_group = neighbors + [i]
            print(i,neighbors)
            local_sum_w = sum(float(client_weights[j]) for j in local_group)

            for key in models[i].trainable_keys:
                temp = torch.zeros_like(models[i].state_dict()[key], device=dev)
                for j in local_group:
                    w_j = float(client_weights[j])
                    temp += w_j * models[j].state_dict()[key].to(dev)
                temp = temp / local_sum_w
                new_params[i][key] = temp

        # Assign back
        with torch.no_grad():
            for i in range(n_clients):
                for key in models[i].trainable_keys:
                    models[i].state_dict()[key].data.copy_(new_params[i][key])

        return models
                    
    return models

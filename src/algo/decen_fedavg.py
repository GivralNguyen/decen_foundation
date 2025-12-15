import torch
from torch import nn
import numpy as np
from algo.communication import decen_communication
from algo.fedavg import fedavg
from training.eval import evaluate
from algo.topo import (
    # create_random_graph,
    create_random_ring_graph,
    create_regular_graph,
    # create_multi_star_graph,
    # create_exponential_graph,
    # create_double_ring_graph,
    # create_cluster_topology,
    # create_random_line_graph,
)
# Inherits from fedavg class, with graph as additional parameter
class decenfedavg(fedavg):
    def __init__(self, base_model, scenario, loss_fun, class_mask, G, aggregation_method='parametric', nonpara_hidden=128, device='cuda'):
        super(decenfedavg, self).__init__(base_model=base_model, scenario=scenario, loss_fun=loss_fun, class_mask=class_mask, aggregation_method=aggregation_method, nonpara_hidden=nonpara_hidden, device=device)
        self.base_model = base_model
        self.base_model.eval()
        self.graph = self._build_graph(G)  # topology graph (nodes: 0..n_clients_each_round-1)
    # aggregation with only neighbors
    def _build_graph(self, G):
        n = self.scenario.n_clients_each_round
        if isinstance(G, str):
            if G == 'random_ring':
                G = create_random_ring_graph(n)
            elif G == 'regular':
                degree = self.scenario.graph_degree
                G = create_regular_graph(n, degree)
            else:
                raise ValueError(f"Graph type {G} not recognized.")
        return G
    def agg(self):
        local_models = self.client_model[:self.scenario.n_clients_each_round]
        local_weights = self.selected_client_weights
        decen_communication(models=local_models, G=self.graph, client_weights=local_weights, aggregation_method=self.aggregation_method, nonpara_hidden=self.nonpara_hidden, device=self.device)
        self.client_model[:self.scenario.n_clients_each_round] = local_models
        if hasattr(self.client_model[0], 'trained_prompts_checklist'):
            for i in range(self.scenario.n_clients_each_round):
                self.client_model[i].reset_trained_pormpts_checklist()
    # average accuracy across all clients
    def global_eval_avg(self, testloader):
        total_loss, total_acc, n_clients = 0.0, 0.0, self.scenario.n_clients_each_round
        for i in range(n_clients):
            eval_loss, eval_acc = evaluate(self.client_model[i], testloader, self.loss_fun, self.device)
            total_loss += eval_loss
            total_acc += eval_acc
        avg_loss, avg_acc = total_loss / n_clients, total_acc / n_clients
        print(f'\n[Decentralized] Average across {n_clients} clients: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}%')
        return avg_loss, avg_acc


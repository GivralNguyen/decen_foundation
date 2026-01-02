import torch
import torch.nn as nn
from torch.optim import Adam
import copy
import numpy as np
import wandb
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from scipy.optimize import linear_sum_assignment

# Non parametric aggregation
class NonparametricAgg(nn.Module):
    def __init__(self, prompt_dim, n_hidden=128):
        super(NonparametricAgg, self).__init__()
        self.cov_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, prompt_dim),
            nn.Sigmoid()
        )
        self.bernoulli_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

    def prompt_likelihood(self, local_prompts, centroids, z):
        _z = torch.tensor(z).to(local_prompts.device)
        lik = 0.
        cost_mat = []
        for i in range(centroids.shape[0]):
            mean_i = centroids[i].view(1, -1)
            cov_i = self.cov_net(mean_i)
            prompt_dist = Independent(Normal(mean_i, cov_i),1)
            lp = prompt_dist.log_prob(local_prompts)
            cost_mat.append(lp)
            log_prob = _z[:, i] * cost_mat[-1] # n_local
            lik += log_prob.sum()
        return lik, torch.stack(cost_mat) # n_global x n_local

    def z_likelihood(self, centroids, z):
        _z = torch.tensor(z).to(centroids.device)
        c = torch.sum(_z, dim=0) # n_global
        lik = 0.
        cost_mat = []
        for i in range(centroids.shape[0]):
            prob_i = self.bernoulli_net(centroids[i].view(1, -1))
            prompt_dist = Independent(Bernoulli(prob_i),1)
            cost_mat.append(prompt_dist.log_prob(c[i] * torch.ones(_z.shape[0]).to(centroids.device)))
            log_prob = _z[:, i] * cost_mat[-1]   # n_local
            lik += log_prob.sum()
        return lik, torch.stack(cost_mat) # n_global x n_local

    # local_prompts: n_clients x n_prompts x 768
    def forward(self, local_prompts, outer_loop=50):
        n_clients, n_local = local_prompts.shape[0], local_prompts.shape[1]
        n_global = n_clients * n_local
        # Initialize z
        z = []
        for i in range(n_clients):
            perm = np.arange(n_global)
            np.random.shuffle(perm)
            zi = np.zeros((n_local, n_global))
            for j in range(n_local):
                zi[j][perm[j]] = 1
            z.append(zi)

        centroids = nn.ParameterList([copy.deepcopy(local_prompts.flatten(0, 1))]) # (n_clients x n_prompts) x 768
        opt = Adam([
            {'params': self.cov_net.parameters()},
            {'params': self.bernoulli_net.parameters()},
            {'params': centroids}
        ])

        # Alternate opt phi, z
        for i in range(outer_loop):
            for t in range(n_clients):
                opt.zero_grad()
                # Compute l1, l2
                l1, m1 = self.prompt_likelihood(local_prompts[t], centroids[0], z[t])
                l2, m2 = self.z_likelihood(centroids[0], z[t])
                #loss.append(l1 + l2)
                loss = -l1 -l2
                loss.backward()
                opt.step()

                # Solve for z
                m = (m1 + m2).t().detach().cpu().numpy()
                row_id, col_id = linear_sum_assignment(m, maximize=True)
                z[t] *= 0
                z[t][row_id, col_id] += 1

            #loss = torch.stack(loss).sum()
            #loss.backward()
            #opt.step()
        z = np.stack(z)
        z = np.sum(np.stack(z), axis=(0, 1), keepdims=False) # n_local x n_global
        print(z)
        global_prompts = centroids[0][np.where(z > 0)[0]]
        del z, centroids
        return global_prompts


class DecenNonparametricAgg(NonparametricAgg):
    def __init__(self, prompt_dim, n_hidden=128):
        super().__init__(prompt_dim=prompt_dim, n_hidden=n_hidden)

    # local_prompts_list: list of length n_clients; each is [n_local_t, prompt_dim]
    def forward(self, local_prompts_list, outer_loop=50):
        n_clients = len(local_prompts_list)
        device = local_prompts_list[0].device

        # number of prompts per client
        n_local_list = [p.size(0) for p in local_prompts_list]
        n_global = sum(n_local_list)

        # -------- init z for each client (ragged) ----------
        z = []
        for t in range(n_clients):
            n_local_t = n_local_list[t]
            perm = np.arange(n_global)
            np.random.shuffle(perm)

            z_t = np.zeros((n_local_t, n_global), dtype=np.float32)
            for j in range(n_local_t):
                z_t[j, perm[j]] = 1.0
            z.append(z_t)

        # -------- init centroids ----------
        all_prompts = torch.cat(local_prompts_list, dim=0)  # [n_global, prompt_dim]
        centroids = nn.ParameterList([nn.Parameter(all_prompts.clone())])

        opt = Adam([
            {'params': self.cov_net.parameters()},
            {'params': self.bernoulli_net.parameters()},
            {'params': centroids}
        ])

        # -------- alternate optimize and z update ----------
        for _ in range(outer_loop):
            for t in range(n_clients):
                opt.zero_grad()

                local_t = local_prompts_list[t]  # [n_local_t, prompt_dim]
                # z_t = z[t]                       # [n_local_t, n_global]

                # inherited from NonparametricAgg
                l1, m1 = self.prompt_likelihood(local_t, centroids[0], z[t])
                l2, m2 = self.z_likelihood(centroids[0], z[t])

                loss = -l1 - l2
                loss.backward()
                opt.step()

                # Hungarian update (client-specific)
                m = (m1 + m2).t().detach().cpu().numpy()
                row_id, col_id = linear_sum_assignment(m, maximize=True)

                z[t] *= 0.0
                # z_t[row_id, col_id] = 1.0
                # z[t] = z_t
                z[t][row_id, col_id] += 1

        # -------- aggregate z across clients ----------
        # z = np.stack(z)
        # z = np.sum(np.stack(z), axis=(0, 1), keepdims=False) # n_local x n_global
        # print(z)
        # global_prompts = centroids[0][np.where(z > 0)[0]]
        z_all = np.zeros(n_global, dtype=np.float32)
        for z_t in z:
            z_all += z_t.sum(axis=0)
        print(z_all)
        used_idx = np.where(z_all > 0)[0]
        used_idx_t = torch.from_numpy(used_idx).long().to(device)

        scores_used = torch.tensor(z_all, device=device)[used_idx_t]

        # K = min(30, scores_used.numel())
        K = scores_used.numel()
        top_scores, top_pos = torch.topk(scores_used, K)
        # print(top_scores, top_pos)
        final_idx_t = used_idx_t[top_pos]
        global_prompts = centroids[0][final_idx_t]  # [K, prompt_dim]

        del z, centroids
        return global_prompts
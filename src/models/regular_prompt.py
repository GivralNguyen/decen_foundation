import torch
from torch import nn
import torchvision.models as models
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from collections import Counter, OrderedDict
# ViT-B/32 model with prompt prepended, but no global prompt pool
class Prompted_ViT_B32(nn.Module):
    def __init__(self, weight_init, prompt_method, num_tokens, prompt_dropout_value=0.0,
                 classification_adaptor=True, frozen_pretrian=True, num_classes=10):
        super(Prompted_ViT_B32, self).__init__()
        self.weight_init = weight_init
        self.prompt_method = prompt_method
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        patch_size = _pair((32, 32))
        self.prompt_dropout = nn.Dropout(prompt_dropout_value)
        self.prompt_proj = nn.Identity()
        self.vit_b32 = models.vit_b_32(weights='IMAGENET1K_V1')
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.vit_b32.heads.head.out_features, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.vit_b32.heads.head.out_features, num_classes)
        hidden_size_each_layers = self.record_hidden_size_each_layers(self.vit_b32)
        self.trainable_keys = list()
        self.control = OrderedDict()
        self.delta_control = OrderedDict()
        self.delta_y = OrderedDict()

        if weight_init == 'random':
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + hidden_size_each_layers[0]))
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size_each_layers[0]))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_method == 'deep':
                self.deep_prompt_embeddings_list = nn.ParameterList()
                for i in range(1, len(hidden_size_each_layers)):
                    val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + hidden_size_each_layers[i]))
                    deep_prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size_each_layers[i]))
                    nn.init.uniform_(deep_prompt_embeddings.data, -val, val)
                    self.deep_prompt_embeddings_list.append(deep_prompt_embeddings)

        else:
            raise ValueError("Initiation is not supported")

        if frozen_pretrian:
            self.vit_b32.requires_grad_(False)

    def record_hidden_size_each_layers(self, origin_model):
        num_encoder_layers = len(origin_model.encoder.layers)
        hidden_size_record = list()
        for i in range(num_encoder_layers):
            for n, p in origin_model.encoder.layers[i].named_parameters():
                if 'ln_1.weight' in n:
                    hidden_size_record.append(p.shape[0])
        return hidden_size_record

    def embedding_input(self, x):
        x = self.vit_b32._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit_b32.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x += self.vit_b32.encoder.pos_embedding
        x = self.vit_b32.encoder.dropout(x)
        return x

    def incorporate_prompt(self, x):
        # print("prompted")
        batch_size = x.shape[0]
        x = self.embedding_input(x)
        # print(x.shape)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(batch_size, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        # print(x.shape)
        return x

    def forward_deep_prompt(self, embedding_output):
        batch_size = embedding_output.shape[0]
        for i in range(len(self.vit_b32.encoder.layers)):
            if i == 0:
                hidden_states = self.vit_b32.encoder.layers[i](embedding_output)
            else:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.deep_prompt_embeddings_list[i-1]).expand(batch_size, -1, -1))

                hidden_statese = torch.cat((
                    hidden_states[:, :1, :],
                    deep_prompt_emb,
                    hidden_states[:, (1+self.num_tokens):, :]
                ), dim=1)

                hidden_states = self.vit_b32.encoder.layers[i](hidden_states)
        encoded = self.vit_b32.encoder.ln(hidden_states)
        return encoded

    def build_trainable_keys(self):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                grad_keys.append(n)
        self.trainable_keys = grad_keys

    def init_contorl_parameter_for_scaffold(self, device='cuda'):
        if len(self.trainable_keys) == 0:
            raise ValueError("Forget initializing trainable keys list")
        for key in self.trainable_keys:
            self.control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_y[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)

    def forward(self, x):
        embedding_output = self.incorporate_prompt(x)
        if self.prompt_method == 'deep':
            encoded = self.forward_deep_prompt(embedding_output)
        elif self.prompt_method == 'shallow':
            hidden_states = self.vit_b32.encoder.layers(embedding_output)
            encoded = self.vit_b32.encoder.ln(hidden_states)

        encoded = encoded[:, 0]
        encoded = self.vit_b32.heads(encoded)
        logits = self.classification_head(encoded)
        return logits
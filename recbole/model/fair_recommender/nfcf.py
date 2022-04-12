from email.policy import strict
from tabnanny import check
import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType


class NFCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NFCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout = config['dropout']
        self.sst_attr = config['sst_attr_list'][0]
        self.fair_weight = config['fair_weight']
        self.load_pretrain_path = config['load_pretrain_path']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size + [1], self.dropout)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        if self.load_pretrain_path is not None:
            self.reset_params(self.load_pretrain_path, dataset.get_user_feature()[1:])

    def reset_params(self, pretrain_path, user_data):
        checkpoint = torch.load(pretrain_path)
        self.item_embedding.weight.data = torch.zeros_like(checkpoint['state_dict']['item_embedding.weight'])
        self.load_state_dict(checkpoint['state_dict'], strict=False)

        sst_value = user_data[self.sst_attr]
        sst_unique_value = torch.unique(sst_value)
        sst1_indices = sst_value == sst_unique_value[0]
        sst2_indices = sst_value == sst_unique_value[1]
        ncf_user_embedding = self.user_embedding.weight.data[1:].clone()

        sst_embedding1 = ncf_user_embedding[sst1_indices].mean(dim=0)
        sst_embedding2 = ncf_user_embedding[sst2_indices].mean(dim=0)
        vector_bias = (sst_embedding1 - sst_embedding2)/torch.linalg.norm(sst_embedding1 - sst_embedding2, keepdim=True)
        vector_bias = torch.mul(ncf_user_embedding, vector_bias).sum(dim=1, keepdim=True) * vector_bias        
        user_embedding = ncf_user_embedding - vector_bias

        self.user_embedding.weight.data[1:] = user_embedding
        self.user_embedding.weight.requires_grad = False
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

    def forward(self, user, item):
        user_mlp_e = self.user_embedding(user)
        item_mlp_e = self.item_embedding(item)
        output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        
        return self.sigmoid(output.squeeze(-1))

    def get_differential_fairness(self, interaction, score):
        sst_unique_values, sst_indices = torch.unique(interaction[self.sst_attr], return_inverse=True)
        iid_unique_values, iid_indices = torch.unique(interaction[self.ITEM_ID], return_inverse=True)
        score_matric = torch.zeros((len(iid_unique_values), len(sst_unique_values)), device=self.device)
        epsilon_values = torch.zeros(len(iid_unique_values), device=self.device)

        concentration_parameter = 1.0
        dirichlet_alpha = concentration_parameter/len(iid_unique_values)

        for i in range(len(iid_unique_values)):
            for j in range(len(sst_unique_values)):
                indices = (iid_indices==i)*(sst_indices==j)
                score_matric[i,j] = (score[indices].sum()+dirichlet_alpha)/(indices.sum()+concentration_parameter)

        for i in range(len(iid_unique_values)):
            epsilon = torch.tensor(0.,dtype=torch.float32)
            for j in range(len(sst_unique_values)):
                for k in range(j+1, len(sst_unique_values)):
                    epsilon = max(epsilon, abs(torch.log(score_matric[i,j])-torch.log(score_matric[i,k])))
            epsilon_values[i] = epsilon
        
        return epsilon_values.mean()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        rec_loss = self.loss(output, label)
        if self.load_pretrain_path is None:
            return rec_loss
        fair_loss = self.get_differential_fairness(interaction, output)

        return rec_loss + self.fair_weight * fair_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

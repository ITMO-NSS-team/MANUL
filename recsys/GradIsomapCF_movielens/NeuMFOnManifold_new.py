import torch
import torch.nn as nn


class NeuMFOnManifold(nn.Module):
    def __init__(self,
                 user_num: int,
                 latent_dim: int,
                 factor_num: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 model_type: str = 'NeuMF-end',
                 use_item_projection: bool = True):
        super().__init__()
        assert model_type in ['MLP', 'GMF', 'NeuMF-end']
        self.model_type = model_type
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_item_projection = use_item_projection
        self.latent_dim = latent_dim

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        mlp_user_dim = factor_num * (2 ** (num_layers - 1))
        self.embed_user_MLP = nn.Embedding(user_num, mlp_user_dim)
        self.mlp_user_dim = mlp_user_dim

        if self.use_item_projection:
            self.item_GMF_linear = nn.Linear(latent_dim, factor_num)
            self.item_MLP_linear = nn.Linear(latent_dim, mlp_user_dim)
        else:
            if latent_dim < max(factor_num, mlp_user_dim):
                raise ValueError(
                    f"use_item_projection=False requires latent_len >= "
                    f"max(factor_num, mlp_user_dim) = {max(factor_num, mlp_user_dim)}, "
                    f"got latent_len={latent_dim}."
                )
            self.item_GMF_linear = None
            self.item_MLP_linear = None

        mlp_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        if self.model_type in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        if self.use_item_projection:
            nn.init.xavier_uniform_(self.item_GMF_linear.weight)
            nn.init.zeros_(self.item_GMF_linear.bias)
            nn.init.xavier_uniform_(self.item_MLP_linear.weight)
            nn.init.zeros_(self.item_MLP_linear.bias)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user, item, item_Z):
        if self.model_type != 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            z_i = item_Z[item]
            if self.use_item_projection:
                embed_item_GMF = self.item_GMF_linear(z_i)
            else:
                embed_item_GMF = z_i[..., :self.factor_num]
            output_GMF = embed_user_GMF * embed_item_GMF
        if self.model_type != 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            z_i = item_Z[item]
            if self.use_item_projection:
                embed_item_MLP = self.item_MLP_linear(z_i)
            else:
                embed_item_MLP = z_i[..., :self.mlp_user_dim]
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
            output_MLP = self.MLP_layers(interaction)
        if self.model_type == 'GMF':
            concat = output_GMF
        elif self.model_type == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), dim=-1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

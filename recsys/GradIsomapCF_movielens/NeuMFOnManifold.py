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
                 freeze_item_projection: bool = False,
                 item_projection_init_data: "torch.Tensor | None" = None):
        super().__init__()
        assert model_type in ['MLP', 'GMF', 'NeuMF-end']
        self.model_type = model_type
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.freeze_item_projection = freeze_item_projection

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        mlp_user_dim = factor_num * (2 ** (num_layers - 1))
        self.embed_user_MLP = nn.Embedding(user_num, mlp_user_dim)

        self.item_GMF_linear = nn.Linear(latent_dim, factor_num)
        self.item_MLP_linear = nn.Linear(latent_dim, mlp_user_dim)

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

        self._init_weights(latent_dim, mlp_user_dim, item_projection_init_data)

    def _init_weights(self, latent_dim, mlp_user_dim, item_projection_init_data):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)

        if self.freeze_item_projection:
            # No learned re-interpretation of the manifold coordinates: the
            # MLP branch takes z_i unchanged (identity - latent_dim is
            # deliberately set equal to mlp_user_dim for exactly this, so no
            # information is discarded), and the GMF branch takes a FIXED
            # (not random - a random frozen compression can be arbitrarily
            # poorly conditioned) PCA projection of the actual item_Z data
            # down to factor_num dims. Only the user-side embeddings and the
            # downstream MLP tower/predict layer remain trainable. See
            # docs/recsys_paper_diary.md, 2026-09-09.
            assert latent_dim == mlp_user_dim, (
                f"freeze_item_projection assumes latent_dim ({latent_dim}) == "
                f"mlp_user_dim ({mlp_user_dim}) so the MLP branch can use an "
                f"identity (information-preserving) map."
            )
            with torch.no_grad():
                self.item_MLP_linear.weight.copy_(torch.eye(mlp_user_dim))
                self.item_MLP_linear.bias.zero_()

            if item_projection_init_data is None:
                raise ValueError(
                    "freeze_item_projection=True requires item_projection_init_data "
                    "(the item_Z matrix) to fit a fixed, well-conditioned GMF "
                    "projection - a random frozen projection can compress poorly."
                )
            with torch.no_grad():
                z = item_projection_init_data.detach().to(torch.float32)
                z_centered = z - z.mean(dim=0, keepdim=True)
                # Top-`factor_num` principal directions via SVD - a fixed,
                # data-informed (not random) dimensionality reduction.
                _, _, Vt = torch.linalg.svd(z_centered, full_matrices=False)
                projection = Vt[:self.factor_num]  # (factor_num, latent_dim)
                self.item_GMF_linear.weight.copy_(projection)
                self.item_GMF_linear.bias.zero_()

            self.item_MLP_linear.weight.requires_grad_(False)
            self.item_MLP_linear.bias.requires_grad_(False)
            self.item_GMF_linear.weight.requires_grad_(False)
            self.item_GMF_linear.bias.requires_grad_(False)
        else:
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
            embed_item_GMF = self.item_GMF_linear(z_i)
            output_GMF = embed_user_GMF * embed_item_GMF
        if self.model_type != 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            z_i = item_Z[item]
            embed_item_MLP = self.item_MLP_linear(z_i)
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

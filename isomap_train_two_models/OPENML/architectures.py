from torch import nn, float32


def regres_fnn(latent_len):
    model_seq = [nn.Linear(latent_len, 512, dtype=float32),
     nn.Linear(512, 256, dtype=float32),
     nn.Linear(256, 64, dtype=float32),
     nn.Linear(64, 1, dtype=float32)
     ]
    return nn.Sequential(*model_seq)


def binary_fnn(latent_len):
    model_seq = [nn.Linear(latent_len, 512, dtype=float32),
                 nn.Linear(512, 256, dtype=float32),
                 nn.Linear(256, 64, dtype=float32),
                 nn.Linear(64, 1, dtype=float32),
                 nn.Sigmoid()]
    return nn.Sequential(*model_seq)
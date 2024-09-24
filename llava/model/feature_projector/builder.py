import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_face_feature_projector(config):
    projector_type = getattr(config, 'face_feature_projector_type', 'linear')
    
    if projector_type == 'linear':
        return nn.Linear(config, 'face_feature_hidden_size', config.hidden_size)
    
    mlp_gelu_match = re.match(r'mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.face_feature_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type=='identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_landmark_feature_projector(config):
    projector_type = getattr(config, 'landmark_feature_projector_type', 'linear')
    
    if projector_type == 'linear':
        return nn.Linear(config, 'landmark_feature_hidden_size', config.hidden_size)
    
    mlp_gelu_match = re.match(r'mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.landmark_feature_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type=='identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

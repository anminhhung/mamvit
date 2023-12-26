import torch
import time
from torch import nn
import random
import torch.nn.functional as F


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


######################################################## Radius
def ft_aug(features, repeat=1, cosine_sim=0.8, random_rate=0.2):
    features_aug = []
    
    for i in range(repeat):
        with torch.no_grad():
            features_aug.append(generate_similar_features(features, cosine_sim, random_rate))
            
    # Concat
    features_aug = torch.cat(features_aug)
        
    return torch.cat((features, features_aug), dim=0)


def generate_similar_features(features, cosine_sim, random_rate):
    # create a random vector of the same size as the input tensor
    rand_features = features + torch.randn_like(features)*random_rate
    
    for idx in range(len(features)):
        count = 0

        feature = features[idx]
        
        # calculate the cosine similarity between the input tensor and the random vector
        rand_feature = rand_features[idx]
        
        cs = torch.dot(feature, rand_feature) / (torch.norm(feature) * torch.norm(rand_feature))
        
        # if the cosine similarity is less than cosine_similarity, repeat steps 2 and 3
        while cs < cosine_sim:
            rand_feature = feature + torch.randn_like(feature)*random_rate
            cs = torch.dot(feature, rand_feature) / (torch.norm(feature) * torch.norm(rand_feature))
            
            count += 1
            if count > 500:
                break
            
        rand_features[idx] = rand_feature
    
    return rand_features

######################################################## Radius
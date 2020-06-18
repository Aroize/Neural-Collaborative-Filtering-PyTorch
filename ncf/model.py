import torch
import torch.nn as nn


"""
Generalized Matrix Factorization
NCF interpretation of MF. This part of framework
is used to endow model of linearity to learn interactions
between users and items
"""
class GMF(nn.Module):
    
    def __init__(self, config):
        super(GMF, self).__init__()
        
        self.user_embeddings = nn.Embedding(config.user_count, config.gmf_dim)
        self.item_embeddings = nn.Embedding(config.item_count, config.gmf_dim)
        self.out = nn.Sequential(nn.Linear(config.gmf_dim, 1), nn.Sigmoid())
        
    def forward(self, users, items):
        user_vector = self.user_embeddings(users)
        item_vector = self.item_embeddings(items)
        pointwise_product = user_vector * item_vector
        prob = self.out(pointwise_product)
        return prob

"""
Multy-Layer Perceptron
Simply a vector concatenation does not account for any interactions
between user and item latent features, which is insufficient
for modelling the collaborative filtering effect. To address
this issue, we propose to add hidden layers on the concatenated vector, using a standard MLP to learn the interaction
between user and item latent features. In this sense, we can
endow the model a large level of flexibility and non-linearity
to learn the interactions between user vector and item vector, rather than the
way of GMF that uses only a fixed element-wise product
on them. 
"""
class MLP(nn.Module):
    
    def __init__(self, config):
        
        super(MLP, self).__init__()
        self.config = config
        
        self.user_embeddings = nn.Embedding(config.user_count, config.mlp_dim * 2**(config.layers_count - 1))
        self.item_embeddings = nn.Embedding(config.item_count, config.mlp_dim * 2**(config.layers_count - 1))
        
        self.hidden = MLP.create_hidden_layers(config)
        
        self.out = nn.Sequential([nn.Linear(config.mlp_dim, 1), nn.Sigmoid()])


    @staticmethod 
    def create_hidden_layers(config):
        hidden_layers = []
        for i in range(config.layers_count):
            input_size = config.mlp_dim * 2**(config.layers_count - i)
            if i == 0:
                input_size += config.user_features
                input_size += config.item_features
            hidden_layers.extend([nn.Linear(input_size, input_size // 2), nn.LeakyReLU(config.slope), nn.Dropout(config.dropout)])
        hidden = nn.Sequential(*hidden_layers)
        return hidden


    def forward(self, *net_input):
        input_vector = MLP.parse_input(net_input, self.user_embeddings, self.item_embeddings)
        hidden_output = self.hidden(input_vector)
        prob = self.hidden(hidden_output)
        return prob


    @staticmethod
    def parse_input(*net_input, user_embeddings_layer, item_embeddings_layer):
        assert len(net_input) >= 2, "Input must contain at least user and item"
        users = net_input[0]
        items = net_input[1]
        user_vector = user_embeddings_layer(users)
        item_vector = item_embeddings_layer(items)
        
        input_vector = torch.cat((user_vector, item_vector), dim=1)
        if len(net_input) > 2:
            user_features = net_input[2]
            input_vector = torch.cat((input_vector, user_features), dim=1)
        if len(net_input) > 3:
            item_features = net_input[3]
            input_vector = torch.cat((input_vector, item_features), dim=1)
        return input_vector

"""
Enseble of GMF and MLP
"""
class NeuMF(nn.Module):
    
    def __init__(self, config):
        self.user_embeddings_gmf = nn.Embedding(config.user_count, config.gmf_dim)
        self.item_embeddings_gmf = nn.Embedding(config.item_count, config.gmf_dim)
        
        self.user_embeddings_mlp = nn.Embedding(config.user_count, config.mlp_dim * 2**(config.layers_count - 1))
        self.item_embeddings_mlp = nn.Embedding(config.item_count, config.mlp_dim * 2**(config.layers_count - 1))
        self.hidden = MLP.create_hidden_layers(config)
        
        self.out = nn.Sequential(nn.Linear(config.gmf_dim + config.mlp_dim, 1), nn.Sigmoid())


    def forward(self, *net_input):
        assert len(net_input) >= 2, "Input must contain at least user and item"
        gmf_output = self.forward_gmf(net_input)
        mlp_output = self.forward_mlp(net_input)
        last_layer_input = torch.cat((gmf_output, mlp_output), dim=1)
        prob = self.out(last_layer_input)
        return prob
        
        
    def forward_gmf(self, *net_input):
        users, items = net_input[0], net_input[1]
        user_vector = self.user_embeddings_gmf(users)
        item_vector = self.item_embeddings_gmf(items)
        pointwise_product = user_vector * item_vector
        return pointwise_product


    def forward_mlp(self, *net_input):
        input_tensor = MLP.parse_input(net_input, self.user_embeddings_mlp, self.item_embeddings_mlp)
        output_tensor = self.hidden(input_tensor)
        return output_tensor
        
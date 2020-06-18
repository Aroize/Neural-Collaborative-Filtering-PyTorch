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
        
        self.out = nn.Sequential(nn.Linear(config.mlp_dim, 1), nn.Sigmoid())


    @staticmethod 
    def create_hidden_layers(config):
        hidden_layers = []
        for i in range(config.layers_count):
            input_size = config.mlp_dim * 2**(config.layers_count - i)
            output_size = input_size // 2
            if i == 0:
                input_size += config.user_features
                input_size += config.item_features
            hidden_layers.extend([nn.Linear(input_size, output_size), nn.LeakyReLU(config.slope), nn.Dropout(config.dropout)])
        hidden = nn.Sequential(*hidden_layers)
        return hidden


    def forward(self, *net_input):
        input_vector = MLP.parse_input(*net_input, user_embeddings_layer=self.user_embeddings, item_embeddings_layer=self.item_embeddings)
        hidden_output = self.hidden(input_vector)
        prob = self.out(hidden_output)
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
        super(NeuMF, self).__init__()
        
        self.user_embeddings_gmf = nn.Embedding(config.user_count, config.gmf_dim)
        self.item_embeddings_gmf = nn.Embedding(config.item_count, config.gmf_dim)
        
        self.user_embeddings_mlp = nn.Embedding(config.user_count, config.mlp_dim * 2**(config.layers_count - 1))
        self.item_embeddings_mlp = nn.Embedding(config.item_count, config.mlp_dim * 2**(config.layers_count - 1))
        self.hidden = MLP.create_hidden_layers(config)
        
        self.out = nn.Sequential(nn.Linear(config.gmf_dim + config.mlp_dim, 1), nn.Sigmoid())


    def forward(self, *net_input):
        assert len(net_input) >= 2, "Input must contain at least user and item"
        gmf_output = self.forward_gmf(*net_input)
        mlp_output = self.forward_mlp(*net_input)
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
        input_tensor = MLP.parse_input(*net_input, user_embeddings_layer=self.user_embeddings_mlp, item_embeddings_layer=self.item_embeddings_mlp)
        output_tensor = self.hidden(input_tensor)
        return output_tensor
    
    @staticmethod
    def load_pretrained(gmf_model, mlp_model, config):
        neu_mf_model = NeuMF(config)
        
        # Load GMF embeddings
        neu_mf_model.user_embeddings_gmf.weight = gmf_model.user_embeddings.weight
        neu_mf_model.item_embeddings_gmf.weight = gmf_model.item_embeddings.weight
        
        # Load MLP embeddings
        neu_mf_model.user_embeddings_mlp.weight = mlp_model.user_embeddings.weight
        neu_mf_model.item_embeddings_mlp.weight = mlp_model.item_embeddings.weight
        
        # Load hidden layers from MLP
        neu_mf_state_dict = neu_mf_model.state_dict()
        hidden_mlp_state_dict = mlp_model.hidden.state_dict()
        hidden_state_dict = { "hidden."+k:v for k,v in hidden_mlp_state_dict.items() }
        neu_mf_state_dict.update(hidden_state_dict)
        neu_mf_model.load_state_dict(neu_mf_state_dict)
        
        # Last output layer is a concatenation of weights of GMF and MLP output layers respectively
        # There is a trade-off between weights of small models, which is tuned by alpha (hyperparameter)
        # alpha - coefficient for GMF weights, (1 - alpha) - coefficient for MLP Weights
        alpha = config.alpha
        gmf_out_layer_weights, gmf_out_layer_bias = alpha*gmf_model.out[0].weight, alpha*gmf_model.out[0].bias
        mlp_out_layer_weights, mlp_out_layer_bias = (1.0 - alpha)*mlp_model.out[0].weight, (1.0 - alpha)*mlp_model.out[0].bias
        
        out_weights = torch.cat((gmf_out_layer_weights, mlp_out_layer_weights), dim=1)
        out_bias = gmf_out_layer_bias + mlp_out_layer_bias
        
        neu_mf_model.out[0].weight = nn.Parameter(out_weights)
        neu_mf_model.out[0].bias = nn.Parameter(out_bias)
        return neu_mf_model
    
    
if __name__ == "__main__":
    from torchsummaryX import summary
    from hparams.utils import Hparam
    import sys
    
    config = Hparam(sys.argv[1])
    
    gmf_model = GMF(config.gmf)
    mlp_model = MLP(config.mlp)
    
    BATCH_SIZE = 1024
    user, item = [torch.zeros(BATCH_SIZE).long() for _ in range(2)]
    print('===============================================================')
    print('                             GMF                               ')
    _ = summary(gmf_model, user, item)
    print('===================================================================')
    print('                               MLP                                 ')
    _ = summary(mlp_model, user, item)
    print('====================================================================')
    print('                                NeuMF                               ')
    neu_mf_model = NeuMF.load_pretrained(gmf_model, mlp_model, config.neu_mf)
    _ = summary(neu_mf_model, user, item)
import torch
from torch.utils.data.dataset import Dataset as BaseDataset
import numpy as np


"""
Class, representing observed items by user
UserItemDataset takes as argument array-like element,
each element contains tuple (user_id, item_id)
By default, class will reindex user_ids and item_ids to match
GMF, MLP, NeuMF embeddings' vocablulary
Also, class will flood data with negative samples
All parameters can be tuned via config
"""
class UserItemDataset(BaseDataset):
    def __init__(self, data_dict, config):
        self.config = config
        user_mapper = None
        item_mapper = None
        self.user_features = None
        self.item_features = None
        
        if not config.is_reindexed:
            data, user_mapper, item_mapper = reindex_data(data_dict['samples'])
            self.user_mapper = user_mapper
            self.item_mapper = item_mapper
            self.data = data
        else:
            self.data = data_dict['samples']
        
        if config.flood_negative:
            negative_samples = flood_negative(self.data, config)
            #positive data should be provided with no ratings, because in NCF
            #we mark interaction as 1 if user knows about item's existence
            self.data = list(map(lambda x: (x[0], x[1], 1.0), self.data))
            #otherwise, we mark non-observed items as zero
            self.data.extend(negative_samples)
            
        if 'user_features' in data_dict:
            if user_mapper is not None:
                self.user_features = reindex_features(data['user_features'], user_mapper)
            else:
                self.user_features = data['user_features']
        
        if 'item_features' in data_dict:
            if item_mapper is not None:
                self.item_features = reindex_features(data['item_features'], item_mapper)
            else:
                self.item_features = data['item_features']


    def __len__(self):
        return len(self.data)
        
        
    def __getitem__(self, idx):
        user, item, label = list(map(lambda x: torch.tensor(x), self.data[idx]))
        user, item = user.long(), item.long()
        sample = {}
        sample['user'] = user
        sample['item'] = item
        sample['label'] = label
        if self.user_features is not None:
            sample['user_features'] = self.user_features[user.item()]
        if self.item_features is not None:
            sample['item_features'] = self.item_features[item.item()]
        return sample
    
    
    
def insert(idx, idx_dict):
    if idx not in idx_dict:
        new_idx = len(idx_dict)
        idx_dict[idx] = new_idx
    else:
        new_idx = idx_dict[idx]
    return new_idx
        
        

def reindex_data(data_samples):
    user_mapper = {}
    item_mapper = {}
    reindexed_data = []
    for user, item in data_samples:
        uid = insert(user, user_mapper)
        iid = insert(item, item_mapper)
        reindexed_data.append((uid, iid))
    return (reindexed_data, user_mapper, item_mapper)


def flood_negative(positive_data_samples, config):
    per_user = config.negatives_per_user
    #For fast search we need to create set of tuples like (user, item)
    positive_set = set(map(lambda x: (x[0], x[1]), positive_data_samples))
    #To iterate over each user once (and don't search in already created negative samples)
    user_set = set(map(lambda x: x[0], positive_data_samples))
    
    #Cause items have been already reindexed, MAX_ITEM_ID = len(set(ITEMS)) - 1
    item_next_idx = len(set(map(lambda x: x[1], positive_data_samples)))
    negative_samples = []
    for user in user_set:
        sample = set()
        while len(sample) != per_user:
            item_idx = np.random.randint(0, item_next_idx)
            pair = (user, item_idx)
            if pair not in sample and pair not in positive_set:
                sample.add(pair)
        #Generate samples with zero labels
        negative_samples.extend(list(map(lambda x: (x[0], x[1], 0.0) ,sample)))
    return negative_samples
    
if __name__ == "__main__":
    
    from hparams.utils import Hparam
    from torch.utils.data.dataloader import DataLoader
    import sys
    
    config = Hparam(sys.argv[1])
    data = [[420, 606], [420, 101], [69, 101], [69, 202]]
    data_dict = { "samples" : data }
    
    dataset = UserItemDataset(data_dict, config.dataset)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print('to convert user indicies back: ', dataset.user_mapper)
    print('to convert item indicies back: ', dataset.item_mapper)
    # Ensure, that dataset supports batches
    for _, data in enumerate(dataloader, 0):
        print(data)
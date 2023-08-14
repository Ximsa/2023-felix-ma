import torch
import wandb
import torch.nn.functional as F

from torch_geometric.nn import LabelPropagation
from botorch.utils.sampling import sample_hypersphere
from utils import euclidean_distance

class OpenGraph(torch.nn.Module):
    def __init__(self, ssl, hidden_channels, num_protos, known_classes, unknown_classes, 
                 device, sup_loss_weight, pseudo_loss_weight, unsup_loss_weight, entropy_loss_weight,
                 ood_percentile, lp_hop = 1,
                 log_all=True, proto_type="param", pseudo_label_method = "none"):
        
        super(OpenGraph, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_protots = num_protos

        self.known_classes = known_classes
        self.unknown_classes = unknown_classes
        self.offset_array = self.calc_offset_array(torch.arange(num_protos), unknown_classes).to(device)

        self.sup_temp = 0.1
        self.pseudo_temp = 0.7
        self.lp_hop = lp_hop
        
        self.ood_percentile = ood_percentile
        self.pseudo_label_method = pseudo_label_method

        self.sup_loss_weight = sup_loss_weight
        self.pseudo_loss_weight = pseudo_loss_weight
        self.unsup_loss_weight = unsup_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        
        
        self.ssl = ssl
        
        #self.encoder = encoder

        self.proto_type = proto_type
        self.normalizer = lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10

        self.device = device
        self.eps = 1e-10
        self.log_all = log_all


        if proto_type == "param":
            self.prototypes = torch.nn.ModuleList()
    
            for i in range(num_protos):
                proto_i = ProtoRepre(hidden_channels, proto_type)
                self.prototypes.append(proto_i)

        elif proto_type == "mean":
            self.prototypes = []
            for i in range(num_protos):
                #self.prototypes.append(sample_hypersphere(hidden_channels, 1).to(device))
                self.prototypes.append(self.normalizer(torch.normal(mean=0.0, std=10.0, size=(hidden_channels,)).to(device)))


    def calc_offset_array(self, known_classes, unknown_classes):
        result = [torch.sum(elem > unknown_classes) for elem in known_classes]
        return torch.Tensor(result).type(torch.long)
            
    
    def forward(self, x, edge_index):
        features = self.ssl.encoder(x, edge_index)
        
        return features


    def inference(self, x, edge_index):
        
        features = self.ssl.encoder(x, edge_index)
        
        if self.proto_type == "mean":
            prototypes_tensor = torch.stack(self.prototypes)
            distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
        else:
            prototypes_tensor = torch.stack([proto() for proto in self.prototypes]) # move init, check copy, extra method to return embeddin
            distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
            
        probas = F.softmax(-distances, dim=1)

        return probas

    def update_proto(self, features, y):
        if self.proto_type == "mean":
            labels = torch.unique(y)
            for i in labels:
                self.prototypes[i] = self.normalizer(torch.mean(features[y==i,:], dim=0).to(self.device))

    def final_prototypes(self, data):
        features = self.ssl.encoder(data.x, data.edge_index)
            
      
        id_mask, pseudo_labels = self.gen_pseudo_labels(features, data.edge_index, data)
        update_mask = data.labeled_mask | id_mask
        all_labels = torch.zeros(data.y.size(0), dtype=torch.long).to(self.device)
        all_labels[id_mask] = pseudo_labels[id_mask]
        all_labels[data.labeled_mask] = data.y[data.labeled_mask]
            
        if self.proto_type == "mean":
            self.update_proto(features[update_mask], all_labels[update_mask])


    def train_one_epoch(self, data):
        
        features = self.ssl.encoder(data.x, data.edge_index)
        
        id_mask, pseudo_labels = self.gen_pseudo_labels(features, data.edge_index, data)
       
        pseudo_mask = data.unlabeled_mask & id_mask
        
        if self.proto_type == "mean":
            proto_update_mask = data.labeled_mask | id_mask 
            all_labels = torch.zeros(data.y.size(0), dtype=torch.long).to(self.device)
            all_labels[id_mask] = pseudo_labels[id_mask]
            all_labels[data.labeled_mask] = data.y[data.labeled_mask]
            
            self.update_proto(features[proto_update_mask], all_labels[proto_update_mask])


        open_loss = self.open_loss(features, pseudo_labels, pseudo_mask, data)
        open_loss.backward(retain_graph=True)
        
        return open_loss



    def open_loss(self, features, pseudo_labels, pseudo_mask, data):

        if self.proto_type == "mean":
            prototypes_tensor = torch.stack(self.prototypes)
        else:
            prototypes_tensor = torch.stack([proto() for proto in self.prototypes])
        
        
        unsup_loss = self.unsupervised_loss(data.x, data.edge_index, data.unlabeled_mask)
        
        entropy_loss = self.entropy_regularizer(features, prototypes_tensor, data)

        if self.pseudo_label_method == "none" or not torch.any(pseudo_mask):
            pseudo_sup_loss = 0
        else:
            pseudo_sup_loss = self.supervised_loss(features[pseudo_mask], pseudo_labels[pseudo_mask], self.pseudo_temp)
        
        if self.proto_type == "mean":
            sup_loss = self.supervised_loss(features[data.labeled_mask], data.y[data.labeled_mask], self.sup_temp)

        elif self.proto_type == "param":
            sup_loss = self.proto_loss(features[data.labeled_mask], 
                                       prototypes_tensor, 
                                       data.y[data.labeled_mask], 
                                       self.sup_temp)

        if self.log_all:        
            wandb.log({'unsup_loss': unsup_loss, 'entropy_loss': entropy_loss, 'pseudo_loss': pseudo_sup_loss,'sup_loss': sup_loss})
        
        return self.sup_loss_weight*sup_loss + self.pseudo_loss_weight*pseudo_sup_loss + self.unsup_loss_weight*unsup_loss + self.entropy_loss_weight*entropy_loss 
        


    def proto_loss(self, features, prototypes_tensor, y, temperature):
        #ToDO: rewrite/check

        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor[self.known_classes]), p=2.0)
        distances = torch.div(distances, temperature)
        
        probabilities_for_training = torch.nn.Softmax(dim=1)(-distances)

     
        y-=self.offset_array[y]
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), y]

        loss = -torch.log(probabilities_at_targets).mean()

        return loss
        

    def supervised_loss(self, features, target, temperature):
        cosine_dist = features @ features.t()
        cosine_mat = torch.div(cosine_dist, temperature)
        mat_max, _ = torch.max(cosine_mat, dim=1, keepdim=True)
        
        cosine_mat = cosine_mat - mat_max.detach()

        target_ = target.contiguous().view(-1, 1)
        mask_pos = torch.eq(target_, target_.T)

        mask_neg_base = 1 - torch.diag(torch.ones(features.size(0))).to(self.device)

        pos_term = (cosine_mat * mask_pos).sum(1) / (mask_pos.sum(1) + self.eps)
        neg_term = (torch.exp(cosine_mat) * mask_neg_base).sum(1)
        
        log_term = (pos_term - torch.log(neg_term + self.eps))
        return -log_term.mean()
        
        
        

    def unsupervised_loss(self, x, edge_index, unlabeled_mask):
        
        pos_z, neg_z, summary = self.ssl.forward(x, edge_index, mask=unlabeled_mask)
        
        loss = self.ssl.loss(pos_z, neg_z, summary) 

        return loss

    def entropy_regularizer(self, features, prototypes_tensor, data):

        distances = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)

        #Assumption: Label distribution of test/train is similar - to drop this assumption use uniform distr.
        labeled_classes, count_labels = torch.unique(data.y[data.labeled_mask], return_counts=True)
        n_remaining_protos = self.num_protots - labeled_classes.size(0)

        

        #set minus prorotypes, labeled classes
        combined = torch.cat((torch.arange(start=0, end=self.num_protots).to(self.device), labeled_classes))
        uniques, counts = combined.unique(return_counts=True)
        unlabeled_classes = uniques[counts == 1].type(torch.LongTensor)

        labeled_proto_ratio = labeled_classes.size(0) / self.num_protots

        prior = torch.zeros(self.num_protots).to(self.device)
        
        if labeled_classes.size(0) == 0:
            prior[unlabeled_classes] = (1/unlabeled_classes.size(0))*(1-labeled_proto_ratio)
        elif unlabeled_classes.size(0) == 0:
            prior[labeled_classes] = (count_labels/count_labels.sum(0))*labeled_proto_ratio
        else:
            prior[labeled_classes] = (count_labels/count_labels.sum(0))*labeled_proto_ratio
            prior[unlabeled_classes] = (1/unlabeled_classes.size(0))*(1-labeled_proto_ratio)

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        probas = F.log_softmax(-distances, dim=1).max()

        loss =  kl_loss(probas, prior)

        return distances.mean()

    def select_threshold(self, closest, percentile):
        q = torch.quantile(closest, percentile)
        return q
        

    def gen_pseudo_labels(self, features, edge_index, data):
        #pseudo label all data points as id that are close to a known class prototype
        
        if self.proto_type == "mean":
            prototypes_tensor = torch.stack(self.prototypes)
        else:
            prototypes_tensor = torch.stack([proto() for proto in self.prototypes])

        if self.pseudo_label_method == "closest":

            distances_id = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor[self.known_classes]), p=2.0)
            min_dist_id, _ = torch.min(distances_id, dim=1)
    
            ood_threshold = self.select_threshold(min_dist_id, self.ood_percentile)
            id_mask = min_dist_id <= ood_threshold
            
    
            distances_all = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor), p=2.0)
            _ , pseudo_labels = torch.min(distances_all, dim=1)

        elif self.pseudo_label_method == "lp":

            distances_id = torch.cdist(F.normalize(features), F.normalize(prototypes_tensor[self.known_classes]), p=2.0)
            min_dist_id, _ = torch.min(distances_id, dim=1)
    
            ood_threshold = self.select_threshold(min_dist_id, self.ood_percentile)
            id_mask = min_dist_id <= ood_threshold
            
            src = data.edge_index[0,:]
            dst = data.edge_index[1,:]
            
            edge_weights_nodes =  euclidean_distance(F.normalize(features[src,:]), F.normalize(features[dst,:]))
            edge_weight_proto = torch.cdist(F.normalize(features)[src,:],
                                            F.normalize(prototypes_tensor[self.known_classes]), p=2.0)

            edge_weight_proto = edge_weight_proto.min(dim=1).values
            
            edge_weights = 1/(edge_weights_nodes + edge_weight_proto + self.eps)
            

            lp = LabelPropagation(num_layers=self.lp_hop, alpha=0.9)
            pseudo_labels = lp(data.y, data.edge_index, data.labeled_mask, edge_weight=edge_weights)
        
            scores, pseudo_labels = pseudo_labels.max(dim=1)
            id_mask = id_mask | (scores > 0.01)

        #pseudo_labels+=self.offset_array[pseudo_labels]

        #id_unlabeld_mask = id_mask & data.unlabeld_mask

        
        return id_mask, pseudo_labels

# ToDO: Neighbor versus random batch versus full batch
    def sampler(self, data, n_nodes, batch_size):
        indices = torch.arange(0,n_nodes)
        perm = torch.randperm(n_nodes)
        sample = perm[:batch_size]

        sampled_data = data.subgraph(sample)

        return sampled_data
        


class ProtoRepre(torch.nn.Module):
    def __init__(self, hidden_channels, proto_type):
        
        super(ProtoRepre, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.proto_type = proto_type
        
        if proto_type == "param":
            self.prototype = torch.Tensor(hidden_channels)
            torch.nn.init.normal_(self.prototype, mean=0.0, std=10.0)
            self.prototype = torch.nn.functional.normalize(self.prototype, p=2.0, dim=0)
            self.prototype = torch.nn.Parameter(self.prototype)

        # self.gat_layer = GATConv(in_channels=hidden_state, out_channels=hidden_state, heads=1, concat=False) #paper variant?
        #self.atten = nn.MultiheadAttention(hidden_state, 2) #code variant
    
    def forward(self):#, spt_embedding_i, degree_list_i=None):
        if self.proto_type == "param":
            return self.prototype
        else:
            if degree_list_i == None:
                avg_proto_i = torch.sum(spt_embedding_i, 0) / spt_embedding_i.shape[0]
                attn_input = torch.cat((avg_proto_i.unsqueeze(0), spt_embedding_i), dim=0).unsqueeze(1)
            else:
                norm_degree = degree_list_i / torch.sum(degree_list_i)
                norm_degree = norm_degree.unsqueeze(1)
                avg_proto_i = torch.sum(torch.mul(spt_embedding_i, norm_degree), 0)
                attn_input = torch.cat((avg_proto_i.unsqueeze(0), spt_embedding_i), dim=0).unsqueeze(1)
                
            attn_output, attn_output_weights = self.atten(attn_input, attn_input, attn_input)
            proto_embedding_i = attn_output[0] + avg_proto_i.unsqueeze(0)
            return proto_embedding_i, attn_output_weights  

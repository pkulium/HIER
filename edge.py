# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from utils.average_weights import average_weights
from utils.average_weights import average_weights_edge
from fednn.intialize_model import initialize_model
from utils.average_weights import average_weights_cloud
import torch
from utils.quantization import quantization_nne

class Edge():

    def __init__(self, id, cids, shared_layers, args):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.update_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.model = shared_layers
        # self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.G = {}
        self.cos_client_ref = {}
        self.args = args

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict):
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        # self.update_state_dict = average_weights(w = received_dict,
        #                                          s_num= sample_num)
        client_ids = [key for key in self.receiver_buffer.keys()]
        self.update_state_dict = average_weights_edge(w = received_dict,
                                                 s_num= sample_num,
                                                 client_learning_rate = [args.client_learning_rate[client_id] for client_id in client_ids],
                                                 edge_learning_rate = args.edge_learning_rate[self.id]
                                                 )
        sd = self.model.state_dict()
        for key in sd.keys():
            sd[key]= torch.add(self.model.state_dict()[key], self.update_state_dict[key])
        self.model.load_state_dict(sd)
        # print('edge after update')
        # print(self.model.state_dict()['stem.0.conv.weight'])
        for i in range(len(client_ids)):
            client_id = client_ids[i]
            if args.model == 'lenet':
                last_layer = torch.flatten(received_dict[i]['fc2.weight'])
            elif args.model == 'cnn_complex':
                last_layer = torch.flatten(received_dict[i]['fc_layer[-1].weight'])
            elif args.model == 'resnet18':
                last_layer = torch.flatten(received_dict[i]['linear.weight'])
            if torch.linalg.norm(last_layer) > 1:
                last_layer /= torch.linalg.norm(last_layer)
            self.G[client_id] = self.G.get(client_id, 0) + last_layer
            self.cos_client_ref[client_id] = args.reference.matmul(self.G[client_id])
        return self.cos_client_ref


    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.model.state_dict()))
        return None

    def send_to_cloudserver(self, cloud, compression_ratio, q_method):
        for key in self.update_state_dict.keys():
            self.update_state_dict[key] = torch.add(self.model.state_dict()[key],-cloud.model.state_dict()[key])
        self.model.load_state_dict(self.update_state_dict)
        # Now we decomment the random sparsification first
        quantization_nne(self.model, compression_ratio, q_method)
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.model.state_dict()))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.model.load_state_dict(shared_state_dict)
        return None


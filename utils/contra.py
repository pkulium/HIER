import torch
def contra(args):
        for i in range(args.num_clients):
            args.cos_client_ref0[i] += args.cos_client_ref[i]
        cos_client_ref = args.cos_client_ref0
        epsilon = 1e-9
        n = len(cos_client_ref)
        args.client_learning_rate = torch.ones(len(cos_client_ref)) 

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)
        tao = torch.zeros(n)
        topk = n // 5
        t = 0.5
        delta = 0.1

        cs = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cs[i][j] = cos(cos_client_ref[i], cos_client_ref[j]).item()

        maxcs = torch.max(cs, dim = 1).values + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        for i in range(n):
            tao[i] = torch.mean(torch.topk(cs[i], topk).values)
        #     if tao[i] > t:
        #         client_reputation[i] -= delta 
        #     else:
        #         client_reputation[i] += delta 

        #  Pardoning: reweight by the max value seen
        args.client_learning_rate = torch.ones((n)) - tao
        args.client_learning_rate /= torch.max(args.client_learning_rate)
        args.client_learning_rate[args.client_learning_rate==1] = 0.99
        args.client_learning_rate = (torch.log((args.client_learning_rate / (1 - args.client_learning_rate)) + epsilon) + 0.5)
        args.client_learning_rate[(torch.isinf(args.client_learning_rate) + args.client_learning_rate > 1)] = 1
        args.client_learning_rate[(args.client_learning_rate < 0)] = 0
        args.client_learning_rate /= torch.sum(args.client_learning_rate)
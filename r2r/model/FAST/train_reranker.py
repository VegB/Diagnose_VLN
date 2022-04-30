from env import R2RBatch
from utils import Tokenizer, read_vocab, check_dir
from vocab import TRAINVAL_VOCAB, TRAIN_VOCAB

import os
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return x


def average(_l):
    return float(sum(_l)) / len(_l)

def count_prefix_len(l1,l2):
    res = 0
    while(res < len(l1) and res < len(l2) and l1[res] == l2[res]):
        res += 1
    return res

def get_path_len(env, scanId, path):
    path_len = 0
    prev = path[0]
    for curr in path[1:]:
        path_len += env.distances[scanId][prev][curr]

def load_data(env, filenames):
    all_data = []
    for fn in filenames:
        with open(fn,'r') as f:
            train_file = json.loads(f.read())
        train_instrs = list(train_file.keys())
        train_data = {}
        
        for instr_id in train_instrs:
            path_id = int(instr_id.split('_')[0])
            scanId = env.gt[path_id]['scan']
            new_data = {
                'instr_id': instr_id,
                'candidates': [],
                'candidates_path': [],
                'reranker_inputs': [],
                'distance': [],
                'gt': env.gt[path_id],
                'gold_idx': -1,
                'goal_viewpointId': env.gt[path_id]['path'][-1],
                'gold_len': get_path_len(env, scanId, env.gt[path_id]['path']),
            }
            self_len = 0
            for i, candidate in enumerate(train_file[instr_id]):
                _, world_states, actions, sum_logits, mean_logits, sum_logp, mean_logp, pm, speaker, scorer = candidate
                new_data['candidates'].append(candidate)
                new_data['candidates_path'].append([ws[1] for ws in world_states])
                new_data['reranker_inputs'].append([len(world_states), sum_logits, mean_logits, sum_logp, mean_logp, pm, speaker] * 4)
                new_data['distance'].append(env.distances[scanId][world_states[-1][1]][new_data['goal_viewpointId']])
                my_path = [ws[1] for ws in world_states]
                if my_path == env.gt[path_id]['path']:
                    new_data['gold_idx'] = i
                
            new_data['self_len'] = self_len
            train_data[instr_id] = new_data
            
        print(fn)
        print('gold',average([d['gold_idx'] != -1 for d in train_data.values()]))
        print('oracle',average([any([dis < 3.0 for dis in d['distance']]) for d in train_data.values()]))
        all_data.append(train_data)
        
    return all_data


def main(args):
    check_dir(args.candidate_dir)
    candidate_dst_file = args.load_reranker
    if os.path.exists(candidate_dst_file):
        print(f'{candidate_dst_file} exists!')
        return
    else:
        print(f'Start training for {candidate_dst_file}')

    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab)
    env = R2RBatch(['none'], batch_size=64, splits=['train','val_seen','val_unseen'],tokenizer=tok, args=args)

    cache_files = [os.path.join(args.cache_result_dir, f'cache_{args.experiment_name}_{split}.json') for split in ['train','val_seen','val_unseen']]
    [train_data, val_seen, val_unseen] = load_data(env, cache_files)

    net = Net(28).cuda()

    batch_labels = []
    valid_points = 0

    for training_point in train_data.values():
        labels = training_point['distance']
        gold_idx = np.argmin(labels)
        ac_len = len(labels)
        choice = 1
        x_1 = []
        x_2 = []
        if choice == 1:
            for i in range(ac_len):
                for j in range(ac_len):
                    if labels[i] <= 3.0 and labels[j] > 3.0:
                        x_1.append(i)
                        x_2.append(j)
                        valid_points += 1
        else:
            for i in range(ac_len):
                if labels[i] > 3.0:
                    x_1.append(gold_idx)
                    x_2.append(i)
                    valid_points += 1
        batch_labels.append((x_1, x_2))

    x_1 = []
    x_2 = []
    optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.6)
    best_performance = 0.0
    for epoch in range(30):  # loop over the dataset multiple times
        epoch_loss = 0
        for i, (instr_id, training_point) in enumerate(train_data.items()):
            inputs = training_point['reranker_inputs']
            labels = training_point['distance']
            ac_len = len(labels)
            
            inputs = torch.stack([torch.Tensor(r) for r in inputs]).cuda()
            labels = torch.Tensor(labels)
            scores = net(inputs)
            
            if i%10 == 0 and len(x_1):
                x1 = torch.cat(x_1, 0)
                x2 = torch.cat(x_2, 0)
                loss = F.relu(1.0 - (x1 - x2)).mean()
                #s = x1-x2
                #loss = (-s + torch.log(1 + torch.exp(s))).mean()
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                x_1 = []
                x_2 = []
        
            if len(batch_labels[i][0]) > 0:
                x_1.append(scores[batch_labels[i][0]])
                x_2.append(scores[batch_labels[i][1]])
      
        print('epoch', epoch, 'loss', epoch_loss)

    print('Finished Training')

    for env_name, data_dict in zip(['train','val_seen','val_unseen'],[train_data,val_seen,val_unseen]):
        successes = []
        inspect = [1,2,3,4,5,6]
        other_success = [[] for _ in range(len(inspect))]
        spl = []
        for instr_id, point in data_dict.items():
            inputs = point['reranker_inputs']
            labels = point['distance']
            inputs = torch.stack([torch.Tensor(r) for r in inputs]).cuda()
            labels = torch.Tensor(labels)
            scores = net(inputs)
            pred = scores.max(0)[1].item()
            successes.append(int(labels[pred] < 3.0))
            
            if (int(labels[pred] < 3.0)):
                for i in range(len(point['distance'])):
                    pass
            
            for idx,i in enumerate(inspect):
                pred = np.argmax([_input[i] for _input in point['reranker_inputs']])
                other_success[idx].append(int(labels[pred] < 3.0))

        print(env_name, average(successes))
        for idx in range(len(inspect)):
            print(average(other_success[idx]))
    
    torch.save(net.state_dict(), candidate_dst_file)
    print(f'Candidate reranker saved to {candidate_dst_file}')


if __name__ == '__main__':
    from test_with_reranker import make_arg_parser
    arg_parser = make_arg_parser()
    args = arg_parser.parse_args()
    print(args)
    main(args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'Petros'

my_seed = 71093
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import os
import sys
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import random
import json
from pprint import pprint

random.seed(my_seed)

cudnn.benchmark = True

setting_entities = 'A'

embedding_dim = 30
hidden_dim = 50

if len(sys.argv) != 8:
    learning_rate = 0.001
    b_size = 100
    resume_from = 0
    shuffle_train = False
    print('Setting A for entities')
else:
    embedding_dim = int(sys.argv[1])
    hidden_dim = int(sys.argv[2])
    b_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    resume_from = int(sys.argv[5])
    shuffle_train = (str(sys.argv[6]) == 'True' or str(sys.argv[6]) == 'true' or str(sys.argv[6]) == '1')
    setting_entities = str(sys.argv[7])
    if setting_entities == 'A':
        print('Setting A for entities')
    elif setting_entities == 'B':
        print('Setting B for entities')
    else:
        print('Error setting_entities argument invalid!')
        exit(1)

gpu_device = 0

od = 'bioread_with_pn_aoareader'
odir = 'drive/My Drive/BioGroup/AOAReaderTest/bioread_pn_output/{}/'.format(od)
if not os.path.exists(odir):
    os.makedirs(odir)

import logging
import datetime
now = datetime.datetime.now()

logger = logging.getLogger(od)
hdlr = logging.FileHandler(odir + ('model_%s.log' % now.strftime("%Y_%d_%m_%H_%M")))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

if resume_from == 0:
    resume_from = None
elif resume_from == 1:
    resume_from = 'drive/My Drive/BioGroup/AOAReaderTest/bioread_pn_output/bioread_with_pn_aoareader/checkpoint_small.pth.tar'
elif resume_from == 2:
    resume_from = 'drive/My Drive/BioGroup/AOAReaderTest/bioread_pn_output/bioread_with_pn_aoareader/best_checkpoint_small.pth.tar'

start_epoch = 0

torch.manual_seed(my_seed)
print(torch.get_num_threads())
print(torch.cuda.is_available())
print(torch.cuda.device_count())

use_cuda = torch.cuda.is_available()
if (use_cuda):
    torch.cuda.manual_seed(my_seed)
    print("Using GPU")

if setting_entities == 'A':
    with open('drive/My Drive/BioGroup/ASReaderTest/vocabDictWithStop_a_small_onlyTrain.dat', 'rb') as openfile:
        vocab = pickle.load(openfile)
else:
    with open('drive/My Drive/BioGroup/ASReaderTest/vocabDictWithStop_b_small_onlyTrain.dat', 'rb') as openfile:
        vocab = pickle.load(openfile)

vocab_size = len(vocab)

print('Loaded Vocab\nVocab Size: %d' % vocab_size)

# Initialize data_size to 0
data_size = 0

# Initialize part indices
train_arr = [1, 2, 3, 4, 5, 6, 7]
valid_arr = [8]
test_arr = [9]
train_part = 0
valid_part = 0
test_part = 0

# for sorting
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_preprocess_data(part):
    global data_size

    print('Loading Part %d' % part)

    # bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '')
    #                             .replace("'", '').strip().lower()).split()

    with open('drive/My Drive/BioGroup/ASReaderTest/dataset_final_json_withstopwords_a_part%d_small.json' % part) as partfile:
        data = json.loads(partfile.read())

    data_size = len(data['abstracts'])

    print('Loaded Data\nData Size: %d' % data_size)

    print('Started processing')

    # List of all the entities transformations for setting b
    entities_trans = []
    
    i = 0
    for abstract in data['abstracts']:
        if setting_entities == 'B':
            # For setting b
            sorted_entities = sorted({e for e in abstract if e.startswith('entity') and len(e)!=6}, key=natural_keys)
            sorted_entities = {ent:'entity%d'%idx for (idx, ent) in enumerate(sorted_entities)}
            entities_trans.append(sorted_entities)
            abstract = [sorted_entities[e] if e.startswith('entity') and len(e)!=6 else e for e in abstract]
            #
        abstract = [vocab[word] if word in vocab else 1 for word in abstract]
        abstract = np.array(abstract, dtype='int64')
        abstract = np.pad(abstract, (0, 500-len(abstract)), mode='constant')
        data['abstracts'][i] = abstract
        i += 1
    data['abstracts'] = np.array(data['abstracts'])

    print('Finished processing abstracts')

    i = 0
    for title in data['titles']:
        if setting_entities == 'B':
            # for setting b
            title = [entities_trans[i][e] if e.startswith('entity') and len(e)!=6 else e for e in title]
            #
        title = [vocab[word] if word in vocab else 1 for word in title]
        title = np.array(title, dtype='int64')
        title = np.pad(title, (0, 50 - len(title)), mode='constant')
        data['titles'][i] = title
        i += 1
    data['titles'] = np.array(data['titles'])

    print('Finished processing titles')

    i = 0
    for entities in data['entities_list']:
        if setting_entities == 'B':
            # For setting b
            entities = [entities_trans[i][e] for e in entities]
        j = 0
        for e in entities:
            e = vocab[e]
            entities[j] = e
            j += 1
        entities = np.array(entities, dtype='int64')
        entities = np.pad(entities, (0, 20 - len(entities)), mode='constant')
        data['entities_list'][i] = entities
        i += 1
    data['entities_list'] = np.array(data['entities_list'])

    print('Finished processing entities_list')

    data['answers'] = np.array(data['answers'])

    print('Finished preprocessing')

    return data



def print_params():
    print(40 * '=')
    print(model)
    print(40 * '=')
    total_params = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')


def data_yielder(split_type):
    global halfway_train
    split_arr = []
    split_idx = 0
    if split_type == 0:
        # Train split
        split_arr = train_arr
        # if already trained halfway go to index 4 (5th json file)
        if halfway_train != -1:
            split_idx = halfway_train
        print('Train Split')
    elif split_type == 1:
        # Valid split
        split_arr = valid_arr
        print('Valid Split')
    elif split_type == 2:
        # Test split
        split_arr = test_arr
        print('Test Split')
    batch_size = b_size
    print('Batch Size: %d' % batch_size)
    while split_idx != len(split_arr):
        if split_type == 0:
            # save halfway progress
            state = {
                'epoch': epoch,
                'finished_valtest': True,
                'halfway_train' : split_idx,
                'state_dict': model.state_dict(),
                'best_cost': min_mean_valid_c,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state, filename=odir + 'checkpoint_small.pth.tar')
            halfway_train = split_idx
            print('Saved halfway progress for train.')
        batch_size = b_size
        data = load_preprocess_data(split_arr[split_idx])

        contexts = data['abstracts']
        quests = data['titles']
        candidates = data['entities_list']
        targets = data['answers']
        if shuffle_train:
            if split_type == 0:
                print('Shuffling data...')
                test = [(x, y, z, w) for x, y, z, w in zip(contexts, quests, candidates, targets)]
                np.random.shuffle(test)
                contexts = [x for (x, y, z, w) in test]
                quests = [y for (x, y, z, w) in test]
                candidates = [z for (x, y, z, w) in test]
                targets = [w for (x, y, z, w) in test]

                contexts = np.asarray(contexts)
                quests = np.asarray(quests)
                candidates = np.asarray(candidates)
                targets = np.asarray(targets)

        for i in range(0, data_size, batch_size):
            if i + batch_size > data_size:
                batch_size = data_size - i
            b_context = np.array(contexts[i:i + batch_size])
            b_quest = np.array(quests[i:i + batch_size])
            b_candidates = np.array(candidates[i:i + batch_size])
            b_target = np.array(targets[i:i + batch_size])
            yield b_context.reshape((batch_size, -1)), b_quest.reshape((batch_size, -1)), b_candidates.reshape((batch_size, -1)), b_target
        split_idx += 1

def train_one_epoch(epoch):
    global sum_cost, sum_acc, m_batches
    gb = model.train()
    sum_cost, sum_acc, m_batches = 0.0, 0.0, 0
    for batch_idx, (b_context, b_quest, b_candidates, b_target) in enumerate(data_yielder(0)):
        m_batches += 1
        optimizer.zero_grad()
        cost_, acc_, log_soft_res, soft_res = model(b_context, b_quest, b_candidates, b_target)
        cost_.backward()
        optimizer.step()
        sum_cost += cost_.data.item()
        sum_acc += acc_
        mean_cost = sum_cost / (m_batches * 1.0)
        mean_acc = sum_acc / (m_batches * 1.0)
        if m_batches % 10 == 0:
            print(
                'train b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                    m_batches, epoch, cost_.data.item(), acc_, mean_cost, mean_acc
                )
            )
        logger.info(
            'train b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data.item(), acc_, mean_cost, mean_acc
            )
        )
    # Print and log final mean cost and accuracy of epoch
    print('Train Epoch %d @@ Mean Cost: %f @@ Mean Acc: %f' % (epoch, mean_cost, mean_acc))
    with open(odir + 'epochs_log.log', 'a+') as logfile:
        logfile.write('Train Epoch %d @@ Mean Cost: %f @@ Mean Acc: %f\n\n' % (epoch, mean_cost, mean_acc))


def valid_one_epoch(epoch):
    gb = model.eval()
    sum_cost, sum_acc, m_batches = 0.0, 0.0, 0
    for b_context, b_quest, b_candidates, b_target in data_yielder(1):
        m_batches += 1
        cost_, acc_, log_soft_res, soft_res = model(b_context, b_quest, b_candidates, b_target)
        sum_cost += cost_.data.item()
        sum_acc += acc_
        mean_cost = sum_cost / (m_batches * 1.0)
        mean_acc = sum_acc / (m_batches * 1.0)
        if m_batches % 10 == 0:
            print(
                'valid b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                    m_batches, epoch, cost_.data.item(), acc_, mean_cost, mean_acc
                )
            )
        logger.info(
            'valid b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data.item(), acc_, mean_cost, mean_acc
            )
        )
    # Print and log final mean cost and accuracy of epoch
    print('Valid Epoch %d @@ Mean Cost: %f @@ Mean Acc: %f' % (epoch, mean_cost, mean_acc))
    with open(odir + 'epochs_log.log', 'a+') as logfile:
        logfile.write('Valid Epoch %d @@ Mean Cost: %f @@ Mean Acc: %f\n\n' % (epoch, mean_cost, mean_acc))
    # Return mean_cost or mean_acc depending on which you want to count for early stopping
    return mean_acc


def test_one_epoch(epoch):
    gb = model.eval()
    sum_cost, sum_acc, m_batches = 0.0, 0.0, 0
    for b_context, b_quest, b_candidates, b_target in data_yielder(2):
        m_batches += 1
        cost_, acc_, log_soft_res, soft_res = model(b_context, b_quest, b_candidates, b_target)
        sum_cost += cost_.data.item()
        sum_acc += acc_
        mean_cost = sum_cost / (m_batches * 1.0)
        mean_acc = sum_acc / (m_batches * 1.0)
        if m_batches % 10 == 0:
            print(
                'test b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                    m_batches, epoch, cost_.data.item(), acc_, mean_cost, mean_acc
                )
            )
        logger.info(
            'test b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data.item(), acc_, mean_cost, mean_acc
            )
        )
    # Print and log final mean cost and accuracy of epoch
    print('Test Epoch %d @@ Mean Cost: %f @@ Mean Acc: %f' % (epoch, mean_cost, mean_acc))
    with open(odir + 'epochs_log.log', 'a+') as logfile:
        logfile.write('Test Epoch %d @@ Mean Cost: %f @@ Mean Acc: %f\n\n' % (epoch, mean_cost, mean_acc))


def dummy_test():
    b_context = np.random.randint(low=1, high=vocab_size - 1, size=(20, 5))
    b_context = np.concatenate([b_context, np.zeros(b_context.shape, dtype=np.int32)], axis=1)
    b_quest = np.random.randint(low=1, high=vocab_size - 1, size=(20, 4))
    b_quest = np.concatenate([b_quest, np.zeros(b_quest.shape, dtype=np.int32)], axis=1)
    b_candidates = np.unique(b_context[:, :4], axis=1)
    b_candidates = np.concatenate([b_candidates, np.zeros(b_candidates.shape, dtype=np.int32)], axis=1)
    b_target = np.array(b_candidates.shape[0] * [1])
    print(b_context.shape)
    print(b_quest.shape)
    print(b_candidates.shape)
    print(b_target.shape)
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        cost_, acc_, log_soft_res, soft_res = model(b_context, b_quest, b_candidates, b_target)
        print(cost_.data.item(), acc_)
        cost_.backward()
        optimizer.step()


def save_checkpoint(state, filename='checkpoint_small.pth.tar'):
    torch.save(state, filename)


def load_model_from_checkpoint():
    global start_epoch, optimizer, finished_valtest, halfway_train, min_mean_valid_c
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from)
        start_epoch = checkpoint['epoch']
        finished_valtest = checkpoint['finished_valtest']
        halfway_train = checkpoint['halfway_train']
        min_mean_valid_c = checkpoint['best_cost']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    else:
        print('%s is not a file.' % resume_from)


class AOAReader_Modeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2):
        super(AOAReader_Modeler, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_h = torch.nn.Parameter(torch.randn(2, 1, self.hidden_dim))
        torch.nn.init.xavier_normal_(self.context_h)
        self.context_bigru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            bias=True,
            dropout=0,
            batch_first=True
        )
        self.question_h = torch.nn.Parameter(torch.randn(2, 1, self.hidden_dim))
        torch.nn.init.xavier_normal_(self.question_h)
        self.question_bigru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            bias=True,
            dropout=0,
            batch_first=True
        )
        self.softmax = torch.nn.Softmax()
        self.dropout_f = nn.Dropout(p=dropout_prob)
        if (use_cuda):
            self.dropout_f = self.dropout_f.cuda(gpu_device)
            self.word_embeddings = self.word_embeddings.cpu()
            self.question_bigru = self.question_bigru.cuda(gpu_device)
            self.context_bigru = self.context_bigru.cuda(gpu_device)

    def get_candidates_for_inst(self, input, context, candidates):
        ret = None
        for cand in candidates:
            if (cand.data.item() == 0):
                pass
            else:
                mask = torch.eq(context, cand)
                if (use_cuda):
                    mask = mask.type(torch.cuda.FloatTensor)
                    masked_soft = torch.mul(input.type(torch.cuda.FloatTensor), mask)
                else:
                    mask = mask.type(torch.FloatTensor)
                    masked_soft = torch.mul(input.type(torch.FloatTensor), mask)
                mask = torch.eq(candidates, cand)
                if (use_cuda):
                    mask = mask.type(torch.cuda.FloatTensor)
                else:
                    mask = mask.type(torch.FloatTensor)
                masked_cand = torch.mul(mask, torch.sum(masked_soft))
                if (ret is None):
                    ret = masked_cand
                else:
                    ret = ret + masked_cand
        return ret

    def get_candidates(self, input, context, candidates):
        ret = []
        for i in range(input.size(0)):
            res_for_one_inst = self.get_candidates_for_inst(input[i], context[i], candidates[i])
            ret.append(res_for_one_inst)
        ret = torch.stack(ret, dim=0)
        return ret

    def calculate_accuracy(self, soft_res, target):
        total = (soft_res.size(0) * 1.0)
        soft_res = np.argmax(soft_res.data.cpu().numpy(), axis=1)
        target = target.data.cpu().numpy()
        wright_ones = len(np.where(soft_res == target)[0])
        acc = wright_ones / total
        return acc

    def softmax_across_rows(self, M):
        ret = []
        for row in M:
            ret.append(self.softmax(torch.unsqueeze(row, 0)))
        ret = torch.stack(ret, dim=0).squeeze(1)
        return ret

    def get_pairwise_score(self, con_gru_out, quest_gru_out):
        rows_att = []
        cols_att = []
        for inst_counter in range(con_gru_out.size(0)):
            M = torch.mm(con_gru_out[inst_counter], quest_gru_out[inst_counter].transpose(1, 0))
            rows_att.append(self.softmax_across_rows(M))
            cols_att.append(self.softmax_across_rows(M.transpose(1, 0)).transpose(0, 1))
        rows_att = torch.stack(rows_att)
        av = rows_att.sum(1) / (rows_att.size(1) * 1.0)
        cols_att = torch.stack(cols_att)
        o = []
        for inst_counter in range(cols_att.size(0)):
            o.append(torch.mm(cols_att[inst_counter], torch.unsqueeze(av[inst_counter], 1)))
        o = torch.stack(o)
        o = o.squeeze(-1)
        return o

    def fix_input(self, context, question, candidates, target):
        context = autograd.Variable(torch.LongTensor(context), requires_grad=False)
        question = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        candidates = autograd.Variable(torch.LongTensor(candidates), requires_grad=False)
        target = autograd.Variable(torch.LongTensor(target), requires_grad=False)
        context_len = [torch.nonzero(item).size(0) for item in context.data]
        question_len = [torch.nonzero(item).size(0) for item in question.data]
        max_cands = torch.max(
            autograd.Variable(torch.LongTensor([torch.nonzero(item).size(0) for item in candidates.data])))
        max_c_len = torch.max(autograd.Variable(torch.LongTensor(context_len)))
        max_q_len = torch.max(autograd.Variable(torch.LongTensor(question_len)))
        context = context[:, :max_c_len.data.item()]
        question = question[:, :max_q_len.data.item()]
        candidates = candidates[:, :max_cands.data.item()]
        if (use_cuda):
            context = context.cuda(gpu_device)
            question = question.cuda(gpu_device)
            candidates = candidates.cuda(gpu_device)
            target = target.cuda(gpu_device)
        return context, question, candidates, target, context_len, question_len

    def forward(self, context, question, candidates, target):
        context, question, candidates, target, context_len, question_len = self.fix_input(context, question, candidates,
                                                                                          target)
        cont_embeds = self.word_embeddings(context)
        quest_embeds = self.word_embeddings(question)
        cont_embeds = self.dropout_f(cont_embeds)
        quest_embeds = self.dropout_f(quest_embeds)
        context_h = torch.cat(cont_embeds.size(0) * [self.context_h], dim=1)
        question_h = torch.cat(quest_embeds.size(0) * [self.question_h], dim=1)
        context_out, context_hn = self.context_bigru(cont_embeds, context_h)
        question_out, question_hn = self.question_bigru(quest_embeds, question_h)
        pws = self.get_pairwise_score(context_out, question_out)
        pws_cands = self.get_candidates(pws, context, candidates)
        log_soft_res = F.log_softmax(pws_cands)
        soft_res = F.softmax(pws_cands)
        acc = self.calculate_accuracy(log_soft_res, target)
        losss = F.nll_loss(log_soft_res, target, weight=None, size_average=True)
        return losss, acc, log_soft_res, soft_res


model = AOAReader_Modeler(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

finished_valtest = True
halfway_train = -1

# Initialize to something big if using mean_cost
# or something small when using mean_acc
min_mean_valid_c = -9000000

print_params()

if (use_cuda):
    model.cuda(gpu_device)

if resume_from is not None:
    load_model_from_checkpoint()
    if not finished_valtest:
        print('Epoch is already trained, going straight to valid')
    if halfway_train != -1:
        print('Epoch has pretrained')
else:
    print("=> no checkpoint found at '{}'".format(resume_from))

sum_cost, sum_acc, m_batches = 0., 0., 0
patience = 3
best_epoch = start_epoch
for epoch in range(start_epoch, 40):
    if finished_valtest:
        train_one_epoch(epoch)
        state = {
                'epoch': epoch,
                'finished_valtest': False,
                'halfway_train': -1,
                'state_dict': model.state_dict(),
                'best_cost': min_mean_valid_c,
                'optimizer': optimizer.state_dict(),
            }
        save_checkpoint(state, filename=odir + 'checkpoint_small.pth.tar')
        halfway_train = -1
        print('Saved checkpoint for training')
    mean_valid_c = valid_one_epoch(epoch)
    # Check for < when using mean_cost or > if using mean_acc
    if (mean_valid_c > min_mean_valid_c):
        # Reset patience variable to 3
        patience = 3
        best_epoch = epoch

        min_mean_valid_c = mean_valid_c
        test_one_epoch(epoch)
        state = {
            'epoch': epoch + 1,
            'finished_valtest': True,
            'halfway_train': -1,
            'state_dict': model.state_dict(),
            'best_cost': min_mean_valid_c,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, filename=odir + 'best_checkpoint_small.pth.tar')
        print('Saved best checkpoint')
    else:
        # Decrease patience variable
        patience -= 1
        # Stop training if patience is zero
        if patience == 0:
            print('Early stopping at epoch %d\n\nBest Epoch: %d' % (epoch, best_epoch))
            with open(odir + 'epochs_log.log', 'a+') as logfile:
                logfile.write('Early stopping at epoch %d\n\nBest Epoch: %d' % (epoch, best_epoch))
            break
    finished_valtest = True

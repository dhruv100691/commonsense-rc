import torch
import numpy as np

from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class Example:

    def __init__(self, input_dict):
        self.id = input_dict['id']
        self.passage = input_dict['d_words']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.d_pos = input_dict['d_pos']
        self.d_ner = input_dict['d_ner']
        self.q_pos = input_dict['q_pos']
        assert len(self.q_pos) == len(self.question.split()), (self.q_pos, self.question)
        assert len(self.d_pos) == len(self.passage.split())
        self.features = np.stack([input_dict['in_q'], input_dict['in_c'],
                                    input_dict['lemma_in_q'], input_dict['lemma_in_c'],
                                    input_dict['tf']], 1)
        assert len(self.features) == len(self.passage.split())
        self.label = input_dict['label']

        self.d_tensor = np.array([vocab[w] for w in self.passage.split()],dtype='int64')
        self.q_tensor = np.array([vocab[w] for w in self.question.split()],dtype='int64')
        self.c_tensor = np.array([vocab[w] for w in self.choice.split()],dtype='int64')
        self.d_pos_tensor = np.array([pos_vocab[w] for w in self.d_pos],dtype='int64')
        self.q_pos_tensor = np.array([pos_vocab[w] for w in self.q_pos],dtype='int64')
        self.d_ner_tensor = np.array([ner_vocab[w] for w in self.d_ner],dtype='int64')
        #self.features = torch.from_numpy(self.features).type(torch.FloatTensor)
        self.p_q_relation = np.array([rel_vocab[r] for r in input_dict['p_q_relation']],dtype='int64')
        self.p_c_relation = np.array([rel_vocab[r] for r in input_dict['p_c_relation']],dtype='int64')

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer: %s, Label: %d' % (self.passage, self.question, self.choice, self.label)

def _to_indices_and_mask(batch_tensor, need_mask=True):
    mx_len = max([t.shape[0] for t in batch_tensor])
    batch_size = len(batch_tensor)
    #indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    indices = np.zeros((batch_size,mx_len),dtype='int64')
    if need_mask:
        #mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
        mask = np.zeros((batch_size, mx_len))
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)] = t
        if need_mask:
            mask[i, :len(t)] = 1
    if need_mask:
        return indices, mask
    else:
        return indices

def _to_feature_tensor(features):
    mx_len = max([f.shape[0] for f in features])
    batch_size = len(features)
    f_dim = features[0].shape[1]
    f_tensor = np.zeros((batch_size, mx_len, f_dim))
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :] = f
    return f_tensor

def batchify(batch_data):
    p, p_mask = _to_indices_and_mask([ex.d_tensor for ex in batch_data])
    p_pos = _to_indices_and_mask([ex.d_pos_tensor for ex in batch_data], need_mask=False)
    p_ner = _to_indices_and_mask([ex.d_ner_tensor for ex in batch_data], need_mask=False)
    p_q_relation = _to_indices_and_mask([ex.p_q_relation for ex in batch_data], need_mask=False)
    p_c_relation = _to_indices_and_mask([ex.p_c_relation for ex in batch_data], need_mask=False)
    q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
    q_pos = _to_indices_and_mask([ex.q_pos_tensor for ex in batch_data], need_mask=False)
    choices = [ex.choice.split() for ex in batch_data]
    c, c_mask = _to_indices_and_mask([ex.c_tensor for ex in batch_data])
    f_tensor = _to_feature_tensor([ex.features for ex in batch_data])
    y = np.array([ex.label for ex in batch_data],dtype='float32')
    return p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, p_q_relation, p_c_relation, y

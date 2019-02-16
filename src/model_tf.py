import logging
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import tensorflow as tf

from utils import vocab
from doc import batchify
from trian_tf import TriAN
from functools import reduce
from operator import mul
from operator import add

logger = logging.getLogger()

class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.finetune_topk = args.finetune_topk
        self.lr = args.lr
        #self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        #print('Use cuda:', self.use_cuda)
        #if self.use_cuda:
        #    torch.cuda.set_device(int(args.gpu))
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.network = TriAN(args)
        self.init_optimizer()
        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        else:
            self.word_emb_mat = self.load_embeddings(vocab.tokens(), args.embedding_file)
        self.sess.run(tf.global_variables_initializer())
        self._report_num_trainable_parameters()

    def _report_num_trainable_parameters(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)

        print('Number of parameters: ', num_params)

    def train(self, train_data):
        #self.network.train()
        print("Training")
        self.updates = 0
        feed_input = {}
        iter_cnt, num_iter = 0, (len(train_data) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(train_data):
            feed_input[self.network.p],feed_input[self.network.p_pos],feed_input[self.network.p_ner],feed_input[self.network.p_mask],feed_input[self.network.q],\
            feed_input[self.network.q_pos],feed_input[self.network.q_mask],feed_input[self.network.c],feed_input[self.network.c_mask],feed_input[self.network.f_tensor],\
            feed_input[self.network.p_q_relation],feed_input[self.network.p_c_relation],feed_input[self.network.y] = [x for x in batch_input]
            feed_input[self.network.word_emb_mat] = self.word_emb_mat
            feed_input[self.network.keep_prob_input] =1 - self.args.dropout_emb
            feed_input[self.network.keep_prob_output] =1 - self.args.dropout_rnn_output

            _, loss = self.sess.run([self.network.train_op, self.network.ce_loss], feed_dict=feed_input)

            #torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            #self.optimizer.step()
            #self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 20 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter-1, loss))

    def evaluate(self, dev_data, debug=False, eval_train=False):
        if len(dev_data) == 0:
            return -1.0
        #self.network.eval()
        feed_input={}
        correct, total, prediction, gold = 0, 0, [], []
        dev_data = sorted(dev_data, key=lambda ex: ex.id)
        iter_cnt, num_iter = 0, (len(dev_data) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(dev_data):
            feed_input[self.network.p], feed_input[self.network.p_pos], feed_input[self.network.p_ner], feed_input[self.network.p_mask], feed_input[self.network.q], \
            feed_input[self.network.q_pos], feed_input[self.network.q_mask], feed_input[self.network.c], feed_input[self.network.c_mask], feed_input[self.network.f_tensor], \
            feed_input[self.network.p_q_relation], feed_input[self.network.p_c_relation], feed_input[self.network.y] = [x for x in batch_input]
            feed_input[self.network.word_emb_mat] = self.word_emb_mat
            feed_input[self.network.keep_prob_input] = 1
            feed_input[self.network.keep_prob_output] = 1

            pred_proba = self.sess.run([self.network.pred_proba], feed_dict=feed_input)
            prediction += [v[0] for v in pred_proba[0]]
            gold += [int(label) for label in feed_input[self.network.y]]
            assert(len(prediction) == len(gold))
            iter_cnt += 1
            if iter_cnt %10 ==0:
               print('Iter: %d/%d' % (iter_cnt, num_iter-1))
           
        if eval_train:
            prediction = [1 if p > 0.5 else 0 for p in prediction]
            acc = sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, gold)]) / len(gold)
            return acc

        cur_pred, cur_gold, cur_choices = [], [], []
        if debug:
            writer = open('./data/output.log', 'w', encoding='utf-8')
        for i, ex in enumerate(dev_data):
            if i + 1 == len(dev_data):
                cur_pred.append(prediction[i])
                cur_gold.append(gold[i])
                cur_choices.append(ex.choice)
            if (i > 0 and ex.id[:-1] != dev_data[i - 1].id[:-1]) or (i + 1 == len(dev_data)):
                py, gy = np.argmax(cur_pred), np.argmax(cur_gold)
                if debug:
                    writer.write('Passage: %s\n' % dev_data[i - 1].passage)
                    writer.write('Question: %s\n' % dev_data[i - 1].question)
                    for idx, choice in enumerate(cur_choices):
                        writer.write('*' if idx == gy else ' ')
                        writer.write('%s  %f\n' % (choice, cur_pred[idx]))
                    writer.write('\n')
                if py == gy:
                    correct += 1
                total += 1
                cur_pred, cur_gold, cur_choices = [], [], []
            if i >= len(prediction):
               break
            cur_pred.append(prediction[i])
            cur_gold.append(gold[i])
            cur_choices.append(ex.choice)

        acc = 1.0 * correct / total
        if debug:
            writer.write('Accuracy: %f\n' % acc)
            writer.close()
        return acc

    def predict(self, test_data):
        # DO NOT SHUFFLE test_data
        #self.network.eval()
        prediction = []
        for batch_input in self._iter_data(test_data):
            feed_input = [x for x in batch_input[:-1]]
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
        return prediction

    def _iter_data(self, data):
        num_iter = (len(data) + self.batch_size - 1) // self.batch_size

        for i in range(num_iter-1): #hack remove -1
            start_idx = i * self.batch_size
            batch_data = data[start_idx:(start_idx + self.batch_size)]
            batch_input = batchify(batch_data)
            # Transfer to GPU
            #if self.use_cuda:
            #    batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
            #else:
            #print ("BATCH",batch_input)
            #batch_input = [Variable(x) for x in batch_input]
            yield batch_input

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        #embedding = self.network.embedding.weight.data
        embedding={}
        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                #assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = list(map(float, parsed[1:]))
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]] = np.array(vec,dtype='float32')
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]]+=np.array(vec,dtype='float32')

        for w, c in vec_counts.items():
            embedding[vocab[w]]  /= c
        emb_mat = np.asarray([v for v in embedding.values()], dtype='float32')

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

        return  emb_mat

    def init_optimizer(self):
        #parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            #self.optimizer = optim.SGD(parameters, self.lr,
            #                           momentum=0.4,
            #                           weight_decay=0)
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.args.optimizer == 'adamax':
            #self.optimizer = optim.Adamax(parameters,
            #                            lr=self.lr,
            #                            weight_decay=0)
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()

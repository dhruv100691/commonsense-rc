import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from utils import vocab, pos_vocab, ner_vocab, rel_vocab
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn


def SeqAttnMatch(input_size,x,y,y_mask,scope_attn=None):
    """Given sequences X and Y, match sequence Y to each element in X.
        * o_i = sum(alpha_j * y_j) for i in X
        * alpha_j = softmax(y_j * x_i)
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
    """
    batch_size=tf.shape(x)[0]
    with tf.variable_scope(scope_attn,reuse=tf.AUTO_REUSE):
        W1= tf.get_variable(name='W1', shape=[input_size, input_size], dtype=tf.float32,initializer=tf.random_normal_initializer(-0.5, 0.5))

        x_proj = tf.nn.relu(tf.reshape(tf.matmul(tf.reshape(x,[-1,tf.shape(x)[2]]),W1),[batch_size,tf.shape(x)[1],-1]))
        y_proj = tf.nn.relu(tf.reshape(tf.matmul(tf.reshape(y,[-1,tf.shape(y)[2]]),W1),[batch_size,tf.shape(y)[1],-1]))
        alpha = tf.nn.softmax(tf.matmul(x_proj,tf.transpose(y_proj,perm=[0,2,1]))) #[batch_size , len1,len2]
        attn_i = tf.matmul(alpha,y) # [batch_size , len1, hdim]

    return attn_i

def BilinearSeqAttn(x,y,normalize=True,scope_attn=None):
    """A bilinear attention layer over a sequence X w.r.t y:

        * o_i = softmax(x_i'Wy) for x_i in X.

        Optionally don't normalize output weights.
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
            attni=batch * hdim1
    """
    batch_size = tf.shape(x)[0]
    with tf.variable_scope(scope_attn,reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable(name='W3', shape=[y.get_shape()[1], x.get_shape()[2]], dtype=tf.float32,initializer=tf.random_normal_initializer(-0.5, 0.5))
        y_proj = tf.expand_dims(tf.matmul(y, W1),1) #batch,1,hdim1
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(x,tf.transpose(y_proj,perm=[0,2,1])),[batch_size,1,-1])) #batch,1,len
        attn_i = tf.reshape(tf.matmul(alpha, x), [batch_size, -1])

    return attn_i

def SelfAttention(x,scope_attn):
    """
     * o_i = softmax(Wx_i) for x_i in X.
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
            attni=batch_len*hdim1
    """
    input_size = x.get_shape()[2]
    batch_size = x.get_shape()[0]

    with tf.variable_scope(scope_attn,reuse=tf.AUTO_REUSE):
        W1= tf.get_variable(name='W3', shape=[input_size, 1], dtype=tf.float32,initializer=tf.random_normal_initializer(-0.5, 0.5))
        x_proj = tf.reshape(tf.matmul(tf.reshape(x,[-1,input_size]),W1),[batch_size,tf.shape(x)[1]])
        alpha = tf.expand_dims(tf.nn.softmax(x_proj),1) #batch ,1, len
        attn_i = tf.reshape(tf.matmul(alpha,x),[batch_size,-1])

    return attn_i

def StackedBRNN(input_rnn,input_size, hidden_size, num_layers,dropout_input=0, dropout_output=0,scope_stack=None):
    cells_fw=[]
    cells_bw=[]
    for i in range(num_layers):
        cell_fw = BasicLSTMCell(hidden_size, state_is_tuple=True)
        cell_bw = BasicLSTMCell(hidden_size, state_is_tuple=True)
        d_cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_output, input_keep_prob=dropout_input)
        d_cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_output, input_keep_prob=dropout_input)
        cells_fw.append(d_cell_fw)
        cells_bw.append(d_cell_bw)
    outputs,_,_=stack_bidirectional_dynamic_rnn(cells_fw, cells_bw,input_rnn,dtype=tf.float32,sequence_length=input_size,scope=scope_stack)

    return outputs

class TriAN(object):
    def __init__(self,args):
        self.args = args
        self.embedding_dim = 300
        self.word_emb_mat = tf.placeholder('float', [None, self.embedding_dim], name='new_emb_mat')
        self.p = tf.placeholder('int32', [args.batch_size, None], name='p')
        self.p_pos = tf.placeholder('int32', [args.batch_size, None], name='p_pos')
        self.p_ner = tf.placeholder('int32', [args.batch_size, None], name='p_ner')
        self.p_mask = tf.placeholder('bool', [args.batch_size, None], name='p_mask')
        self.q = tf.placeholder('int32', [args.batch_size, None], name='q')
        self.q_pos = tf.placeholder('int32', [args.batch_size, None], name='q_pos')
        self.q_mask = tf.placeholder('bool', [args.batch_size, None], name='q_mask')
        self.c = tf.placeholder('int32', [args.batch_size, None], name='c')
        self.c_mask = tf.placeholder('bool', [args.batch_size, None], name='c_mask')
        self.f_tensor = tf.placeholder('float32', [args.batch_size, None,5], name='f_tensor')
        self.p_q_relation = tf.placeholder('int32', [args.batch_size, None], name='p_q_relation')
        self.p_c_relation = tf.placeholder('int32', [args.batch_size, None], name='p_c_relation')
        self.y = tf.placeholder('float32', [args.batch_size], name='y')
        self.keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
        self.keep_prob_output = tf.placeholder(tf.float32, name='keep_prob_output')

        init_weights = tf.random_normal_initializer(0, 0.1)
        with tf.variable_scope("embedding_layer"):
            # embedding matrices
            self.embedding = tf.get_variable("word_emb_mat", dtype='float', shape=[2, self.embedding_dim], initializer=init_weights)
            self.pos_embedding = tf.get_variable("pos_emb_mat", dtype='float', shape=[len(pos_vocab), self.args.pos_emb_dim],
                                                 initializer=init_weights)
            self.ner_embedding = tf.get_variable("ner_emb_mat", dtype='float', shape=[len(ner_vocab), self.args.ner_emb_dim],
                                                 initializer=init_weights)
            self.rel_embedding = tf.get_variable("rel_emb_mat", dtype='float', shape=[len(rel_vocab), self.args.rel_emb_dim],
                                                 initializer=init_weights)

            self.embedding = tf.concat(axis=0, values=[self.embedding, self.word_emb_mat])
            with tf.name_scope("embedding_dropout"):
                self.embedding = tf.nn.dropout(self.embedding, keep_prob=self.keep_prob_input,noise_shape=[12626, 1])

            p_emb = tf.nn.embedding_lookup(self.embedding,self.p)
            q_emb = tf.nn.embedding_lookup(self.embedding,self.q)
            c_emb = tf.nn.embedding_lookup(self.embedding,self.c)
            p_pos_emb = tf.nn.embedding_lookup(self.pos_embedding,self.p_pos)
            p_ner_emb = tf.nn.embedding_lookup(self.ner_embedding,self.p_ner)
            q_pos_emb = tf.nn.embedding_lookup(self.pos_embedding,self.q_pos)
            p_q_rel_emb = tf.nn.embedding_lookup(self.rel_embedding,self.p_q_relation)
            p_c_rel_emb = tf.nn.embedding_lookup(self.rel_embedding,self.p_c_relation)

        with tf.variable_scope("attention_layer",reuse=tf.AUTO_REUSE):
            #question aware passage embeddings
            p_q_weighted_emb = SeqAttnMatch(self.embedding_dim,p_emb, q_emb, self.q_mask,"p_q_weighted_att")#batch_size,len_p,emb_dim
            c_q_weighted_emb = SeqAttnMatch(self.embedding_dim,c_emb, q_emb, self.q_mask,"c_q_weighted_att")#batch_size,len_c,emb_dim
            c_p_weighted_emb = SeqAttnMatch(self.embedding_dim,c_emb, p_emb, self.p_mask,"c_p_weighted_att")#batch_size,len_c,emb_dim

        p_rnn_input = tf.concat([p_emb, p_q_weighted_emb, p_pos_emb, p_ner_emb,self.f_tensor, p_q_rel_emb, p_c_rel_emb],axis=2)
        c_rnn_input = tf.concat([c_emb, c_q_weighted_emb, c_p_weighted_emb], axis=2)
        q_rnn_input = tf.concat([q_emb, q_pos_emb], axis=2)

        p_len = tf.reduce_sum(tf.cast(self.p_mask, 'int32'), 1)  # [N]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)
        c_len = tf.reduce_sum(tf.cast(self.c_mask, 'int32'), 1)

        with tf.variable_scope("encoding_layer",reuse=tf.AUTO_REUSE):
            # batch_size,len,2*hidden_size
            p_hiddens = StackedBRNN(p_rnn_input,p_len, self.args.hidden_size, self.args.doc_layers,self.keep_prob_input, self.keep_prob_output,"doc_rnn")
            c_hiddens = StackedBRNN(c_rnn_input,c_len, self.args.hidden_size, 1,self.keep_prob_input, self.keep_prob_output,"a_rnn")
            q_hiddens = StackedBRNN(q_rnn_input,q_len, self.args.hidden_size, 1,self.keep_prob_input, self.keep_prob_output,"q_rnn")

        with tf.variable_scope("output_layer",reuse=tf.AUTO_REUSE):
            # batch,2*hidden_size
            q_hidden = SelfAttention(q_hiddens,"q_self")
            c_hidden = SelfAttention(c_hiddens,"c_self")
            p_hidden = BilinearSeqAttn(p_hiddens,q_hidden,True,"p_q_bi_lin_attn")

            W3 = tf.get_variable(name='W3', shape=[2*self.args.hidden_size, 2*self.args.hidden_size], dtype=tf.float32,initializer=init_weights)
            W4 = tf.get_variable(name='W4', shape=[2*self.args.hidden_size, 2*self.args.hidden_size], dtype=tf.float32,initializer=init_weights)

            #logits = tf.reshape(tf.matmul(tf.expand_dims(tf.matmul(c_hidden,W3),1),
            #                              tf.reshape(p_hidden,[self.args.batch_size,2*self.args.hidden_size,1])),[self.args.batch_size,-1]) #batch,1
            #logits_final = tf.add(logits,tf.reshape(tf.matmul(tf.expand_dims(tf.matmul(c_hidden,W4), 1),tf.reshape(q_hidden, [self.args.batch_size, 2*self.args.hidden_size, 1])),
            #                    [self.args.batch_size, -1]))  # batch,1
            logits = tf.reduce_sum(tf.matmul(p_hidden,W3) * c_hidden,-1)
            logits_final = tf.add(logits,tf.reduce_sum(tf.matmul(q_hidden,W3) * c_hidden,-1))

            self.pred_proba = tf.nn.sigmoid(logits_final)

        with tf.name_scope("optimization"):
            # Loss function
            self.ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_final, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.args.lr)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.ce_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.args.grad_clipping)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))







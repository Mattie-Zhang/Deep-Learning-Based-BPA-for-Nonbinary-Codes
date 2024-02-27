import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import datetime

from code_gen import CodeCls
from tool_encode import get_data
from tool_bp import gen_char_che, gen_char_var, gen_char_post, gen_perm_decode
from tool_decode import hard_decision, check_codes, com_err_rate

# --------------define code----------------------------

name = './64_32_16.txt' # GF(8): 1+x+x^3, GF(16): 1+x+x^4, GF(32): 1+x^2+x^5, GF(64): 1+x+x^6
code = CodeCls(name)

N, M, q, log_q, row_deg_max, col_deg_max, nb_edge = code.N, code.M, code.q, code.log_q, code.row_deg_max, code.col_deg_max, code.nb_edge

# char_edge, char_eqn = gen_char_che(code)
# np.save(f'./char_edge{N}_{M}_{q}.npy', char_edge)  # shape (nb_test, N, q)
# np.save(f'./char_eqn{N}_{M}_{q}.npy', char_eqn)  # shape (nb_test, N, q)
char_edge = np.load(f'./char_edge{N}_{M}_{q}.npy')
char_eqn = np.load(f'./char_eqn{N}_{M}_{q}.npy')
char_var = gen_char_var(N, col_deg_max)
char_post = gen_char_post(N, col_deg_max)

perm_decode, deperm_decode = gen_perm_decode(code)

print(perm_decode)

print(deperm_decode)
exit(0)

# --------------parameters----------------------------

nb_iter = 10
batch_size = 100
nb_epoch = 1000000
learning_rate = 0.001
dis_fac = 0.8
dis_fac_lt = [np.power(dis_fac, nb_iter-i) for i in range(nb_iter)]
rate = float(N - M) / float(N)

snr_l = 1.0 # the noise rate of training set
snr_h = 5.5
snr_step = 0.5
snr_db = np.arange(snr_l, snr_h + snr_step, snr_step, dtype=float)
snr = 10.0 ** (snr_db / 10.0)
noise_std = np.sqrt(1.0 / (2.0 * rate * snr))
snr_num = len(snr_db)

# --------------decoder----------------------------

method = 'MSA-FB'

mat = code.mat.copy().T
mat[np.nonzero(mat)] = range(1, nb_edge+1)
mat = mat.T
idx = mat[np.nonzero(mat)].reshape(M, -1) - 1  # shape (M, row_deg_max)
idx_inv = np.argsort(idx.flatten())

print(idx)

a_idx = code.min_t[np.arange(q)[:,None], np.arange(q)[None,:]]
b_idx = code.add_t[np.arange(q)[:,None], np.arange(q)[None,:]]

print(a_idx)

alpha_cv = np.random.normal(0,1, (batch_size,nb_edge,q))

# print(alpha_cv)

def for_back(alpha_cv):

    beta_tmp = tf.reshape(alpha_cv, [-1, nb_edge * q])  # can change to indexing???
    beta_tmp = tf.matmul(beta_tmp, perm_decode)
    beta_tmp = tf.reshape(beta_tmp, [-1, nb_edge, q])
    
    beta_tmp = beta_tmp[:, idx] # shape (batch_size, M, row_deg_max, q)
    a = np.ones((batch_size, M, row_deg_max, q), dtype=int) * numpy.NINF
    b = np.ones((batch_size, M, row_deg_max, q), dtype=int) * numpy.NINF
    a[:, :, 0, 0], b[:, :, -1, 0] = 0, 0
    
    for i in range(1, row_deg_max):
        a_tmp = a[:, :, i-1, a_idx]
        b_tmp = b[:, :, row_deg_max-i, b_idx]
        a[:,:,i] = np.max(beta_tmp[:, :, i]+a_tmp, axis=-1)
        b[:,:,row_deg_max-i-1] = np.max(beta_tmp[:, :, i]+b_tmp, axis=-1)

    beta_tmp = np.empty((batch_size, M, row_deg_max, q), dtype=tf.float32)
    for i in range(row_deg_max):
        b_tmp = b[:, :, row_deg_max-i, b_idx]
        beta_tmp[:,:,i] = np.max(a[:,:,i]+b_tmp, axis=-1)
        
    beta_vc = beta_tmp.reshape(batch_size, -1, q)[:, idx_inv]
    
    beta_tmp = tf.reshape(beta_tmp, [-1, nb_edge * q])
    beta_tmp = tf.matmul(beta_tmp, deperm_decode)
    beta_vc = tf.reshape(beta_tmp, [-1, nb_edge, q])
    return beta_vc     
    
for_back(alpha_cv) 
exit(0)


# opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) # 

# initializer = tf.random_normal_initializer(mean=1.0, stddev=0.05)
# w_llr_var = tf.Variable(initial_value=initializer(shape=(nb_edge,)), trainable=True)  # nb_iter, 
# w_llr_post = tf.Variable(initial_value=initializer(shape=(N,)), trainable=True)  # nb_iter, 
# w_var = tf.Variable(initial_value=initializer(shape=(col_deg_max, nb_edge)), trainable=True)  # nb_iter, 
# w_post = tf.Variable(initial_value=initializer(shape=(nb_edge,)), trainable=True)  # nb_iter, 

# v_e1 = tf.Variable(initial_value=initializer(shape=(nb_edge,1)), trainable=True)  # nb_iter, 
# v_e2 = tf.Variable(initial_value=initializer(shape=(nb_edge,1)), trainable=True)  # nb_iter, 

# initializer = tf.random_normal_initializer(mean=0.0, stddev=0.05)
# u_e1 = tf.Variable(initial_value=initializer(shape=(nb_edge,1)), trainable=True)  # nb_iter, 
# u_e2 = tf.Variable(initial_value=initializer(shape=(nb_edge,1)), trainable=True)  # nb_iter, 

w_llr_var = np.ones((nb_edge), dtype=np.float32) # nb_iter, 
w_llr_post = np.ones((N), dtype=np.float32) # nb_iter, 
w_var = np.ones((col_deg_max, nb_edge), dtype=np.float32) # nb_iter, 
w_post = np.ones((nb_edge), dtype=np.float32) # nb_iter, 

w_llr_var = tf.convert_to_tensor(w_llr_var)
w_llr_post = tf.convert_to_tensor(w_llr_post)
w_var = tf.convert_to_tensor(w_var)
w_post = tf.convert_to_tensor(w_post)

def decode(mode, codewords, llr):  # tensor shape in calculation: (batch_size, nb_edge, q)

    llr = tf.convert_to_tensor(llr, dtype=tf.float32)
    alpha_cv0 = tf.repeat(llr, col_deg_max, axis=1)
    alpha_cv = tf.identity(alpha_cv0)  # edges are stored column first with respect to the parity check matrix
    beta_vc = tf.zeros((batch_size, nb_edge, q), dtype=tf.float32)
    w_var_t = tf.tile(w_var, [N, 1])  # 1, 
    
    alpha_cv = alpha_cv - tf.reduce_min(alpha_cv, -1, keepdims=True)  # normalize by substracting the minimum value of each row

    if mode == 'training':
        loss = tf.constant(0, shape=(batch_size, N), dtype=tf.float32)
    # else:
        # flag = np.zeros(batch_size, dtype=int) + 1
        # opt_decode = np.zeros((batch_size, N), dtype=int)

    for i in range(0, nb_iter):
    
        # check node processing (SPA-foward backward algorithm)
        bata_vc = for_back(alpha_cv)

        # check node processing (SPA) (run very slow when the data size is large, and a little faster when the data size is small)
        # alpha_tmp = tf.transpose(alpha_cv, perm=(1, 2, 0))
        # alpha_tmp = tf.gather(alpha_tmp, char_edge, axis=0)
        # beta_tmp = tf.reshape(tf.constant([], dtype=tf.float32), (0, q, count, row_deg_max-1, batch_size))
        # for j in range(nb_edge):
            # tmp = tf.gather_nd(alpha_tmp[j], char_eqn[j])
            # tmp = tf.expand_dims(tmp, axis=0)
            # beta_tmp = tf.concat([beta_tmp, tmp], axis=0)
        # beta_tmp = tf.reduce_sum(beta_tmp, axis=-2)
        # beta_tmp = tf.reduce_max(beta_tmp, axis=-2)  
        # beta_vc = tf.transpose(beta_tmp, perm=(2, 0, 1))  # shape (batch_size, nb_edge, q)
        
        # check node processing (SPA) 
        # alpha_tmp = tf.transpose(alpha_cv, perm=(1, 2, 0))
        # beta_tmp = tf.reshape(tf.constant([], dtype=tf.float32), (0, q, batch_size))
        # for j in range(nb_edge):
        #     tmp = tf.gather(alpha_tmp, char_edge[j], axis=0)
        #     tmp = tf.gather_nd(tmp, char_eqn[j])
        #     tmp = tf.reduce_sum(tmp, axis=-2)
        #     tmp = tf.reduce_max(tmp, axis=-2)
        #     tmp = tf.expand_dims(tmp, axis=0)
        #     beta_tmp = tf.concat([beta_tmp, tmp], axis=0)
        # beta_vc = tf.transpose(beta_tmp, perm=(2, 0, 1))  # shape (batch_size, nb_edge, q) 
        
        # beta_vc = beta_vc - tf.reduce_min(beta_vc, -1, keepdims=True)  # normalize by substracting the minimum value of each row
        # beta_vc = tf.nn.relu(beta_vc - tf.multiply(tf.reduce_max(beta_vc, -1, keepdims=True)/q, u_e2) - tf.multiply((tf.reduce_sum(beta_vc, -1, keepdims=True) - tf.reduce_max(beta_vc, -1, keepdims=True))/q, v_e2))  # [i]
        
        # variable node processing
        w_var_tmp = tf.multiply(char_var, w_var_t)  # [i]
        alpha_cv = tf.einsum('i,aik->aik', w_llr_var, alpha_cv0) + tf.einsum('ij,ajk->aik', w_var_tmp, beta_vc)  # [i]
        
        alpha_cv = alpha_cv - tf.reduce_min(alpha_cv, -1, keepdims=True)  # normalize by substracting the minimum value of each row
        # alpha_cv = tf.nn.relu(alpha_cv - tf.multiply(tf.reduce_max(alpha_cv, -1, keepdims=True)/q, u_e1) - tf.multiply((tf.reduce_sum(alpha_cv, -1, keepdims=True) - tf.reduce_max(alpha_cv, -1, keepdims=True))/q, v_e1))  # [i]
        
        # tmp = tf.expand_dims(alpha_tmp[:, :, 0], axis=-1)  # normalize by substracting the first element of each row
        # alpha_cv = alpha_tmp - tf.repeat(tmp, repeats=[q], axis=-1)  

        # a posteriori information
        w_post_tmp = tf.multiply(char_post, w_post)  # [i]
        post = tf.einsum('i,aik->aik', w_llr_post, llr) + tf.einsum('ij,ajk->aik', w_post_tmp, beta_vc)  # [i]

        if mode == 'training':
            loss += dis_fac_lt[i] * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=codewords, logits=post)

        # else:
            # label_tmp = np.nonzero(flag)[0]
            # a_pred = hard_decision(tf.gather(post, label_tmp))
            # opt_decode[label_tmp] = a_pred
            # label_tmp_new = label_tmp[np.nonzero(check_codes(a_pred, code))]
            # flag[label_tmp_new] = 0

    if mode == 'training':
        return loss 
        
    else:
        return hard_decision(post)
        # return opt_decode


# --------------test----------------------------

nb_test = snr_num * 20000
t_words, t_llr = get_data(nb_test, code, noise_std)
np.save(f'./t_words{N}_{M}_{q}0.npy', t_words)  # shape (nb_test, N)
np.save(f'./t_llr{N}_{M}_{q}0.npy', t_llr)  # shape (nb_test, N, q)
# t_words = np.load(f'./t_words{N}_{M}_{q}0.npy')
# t_llr = np.load(f'./t_llr{N}_{M}_{q}0.npy')

def decode_test(t_words, t_llr):
    mode = 'eval'
    t_words_pred = np.zeros((0, N), dtype=int)
    for i in range(int(nb_test / batch_size)):
        tmp_pred = decode(mode, None, t_llr[i*batch_size:(i+1)*batch_size])
        t_words_pred = np.concatenate((t_words_pred, tmp_pred), axis=0)
    err_rate = com_err_rate(t_words, t_words_pred, code, snr_num)
    return err_rate

print(datetime.datetime.now(), 'start testing')
err_rate = decode_test(t_words, t_llr)
print(datetime.datetime.now(), err_rate)
exit(0)

# --------------train----------------------------

def save_params():
    w_llr_var_tmp, w_var_tmp, w_llr_post_tmp, w_post_tmp = w_llr_var.numpy(), w_var.numpy(), w_llr_post.numpy(), w_post.numpy()  #, u_e1_tmp, v_e1_tmp, u_e2_tmp, v_e2_tmp, u_e1.numpy(), v_e1.numpy(), u_e2.numpy(), v_e2.numpy()
    tmp = 1 - np.tile(np.eye(col_deg_max), (1, N))  # only save the values that have been updated 
    # w_var_tmp_red = [w_var_tmp[i][np.nonzero(tmp)] for i in range(nb_iter)] 
    w_var_tmp_red = w_var_tmp[np.nonzero(tmp)]
      
    params = {'w_llr_var': w_llr_var_tmp, 'w_var': w_var_tmp_red, 'w_llr_post': w_llr_post_tmp, 'w_post': w_post_tmp}  # , 'u_e1': u_e1_tmp, 'v_e1': v_e1_tmp, 'u_e2': u_e2_tmp, 'v_e2': v_e2_tmp
    np.save(f'./params{N}_{M}_{q}_0.npy', params)

for n in range(nb_epoch):
    mode = 'training'

    codewords, llr = get_data(batch_size, code, noise_std)

    # loss = lambda: decode(mode, llr, codewords)
    # tf.keras.optimizers.Adam(learning_rate=learning_rate).minimize(loss, [w_var, w_post])
        
    with tf.GradientTape() as tape:
        loss = decode(mode, codewords, llr)
    grads = tape.gradient(loss, [w_llr_var, w_var, w_llr_post, w_post])  #, u_e1, v_e1, u_e2, v_e2
    
    grads = [tf.clip_by_value(g, clip_value_min=-1000, clip_value_max=1000) for g in grads]
    
    opt.apply_gradients(zip(grads, [w_llr_var, w_var, w_llr_post, w_post]))  #, u_e1, v_e1, u_e2, v_e2
     
    if (n+1) % 3000 == 0:
        learning_rate = learning_rate * 0.8  # 0.001 * (1.0 / (1.0 + 0.2 * int((n+1)/5000)))
        opt.learning_rate = learning_rate
        save_params()

    if (n+1) % 1000 == 0:
        err_rate = decode_test(t_words, t_llr)
        print(n+1, datetime.datetime.now(), err_rate)
            
 






from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from ctc_beam_search_decoder import *
import time

sess = tf.Session()
input_pt = tf.placeholder(tf.float32, shape=(100, 1, 29))
decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(input_pt,
                                                           [100],
                                                           beam_width=5,
                                                           top_paths=1,
                                                           merge_repeated=False)

vocab_list = ['\'', ' '] + [chr(i) for i in range(97, 123)]


# vocab_list = ['A']

# vocab_list = ['\'', ' ']+[chr(i) for i in range(97, 123)]

def generate_probs(num_time_steps, probs_dim):
    probs_mat = np.random.random(size=(num_time_steps, probs_dim))
    probs_mat = [probs_mat[index] / sum(probs_mat[index]) for index in range(num_time_steps)]
    # probs_mat = np.exp(probs_mat) / np.exp(probs_mat).sum(axis=1)[:, None]
    return probs_mat


def generate_probs1():
    return np.array([[0.3, 0.7], [0.4, 0.6]])


def t_beam_search_decoder():
    max_time_steps = 100
    probs_dim = len(vocab_list) + 1
    beam_size = 5
    num_results_per_sample = 3

    input_prob_matrix_0 = np.asarray(generate_probs(max_time_steps, probs_dim), dtype=np.float32)
    # input_prob_matrix_0 = np.asarray(generate_probs1(), dtype=np.float32)
    # print(input_prob_matrix_0)
    # Add arbitrary offset - this is fine
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)  # + 2.0

    # len max_time_steps array of batch_size x depth matrices
    inputs = ([
        input_log_prob_matrix_0[t, :][np.newaxis, :] for t in range(max_time_steps)]
    )
    tf.reset_default_graph()
    inputs_t = [ops.convert_to_tensor(x) for x in inputs]
    inputs_t = array_ops.stack(inputs_t)
    st = time.time()
    # run CTC beam search decoder in tensorflow
    # with tf.Session() as sess:
    # decoded_g, log_probabilities_g = tf.nn.ctc_greedy_decoder(inputs_t, [max_time_steps], merge_repeated=False)
    # with tf.Graph().as_default():
    # a = tf.constant(5)
    # b = tf.constant(5)
    # c = a + b
    # print(sess.run(c))
    # result = ops.convert_to_tensor(inputs_t, name="input_pt")
    # print(result.get_shape())
    tf_decoded, tf_log_probs = sess.run([decoded, log_probabilities], feed_dict={input_pt: inputs})

    # tf_decoded_g, tf_log_probs_g = sess.run([decoded_g, log_probabilities_g])
    st1 = time.time() - st
    print("time spent is %.4f" % st1)
    # run original CTC beam search decoder
    import kenlm
    model = kenlm.Model('/Users/lonica/Downloads/sougou_2.binary')
    beam_result = ctc_beam_search_decoder(
        probs_seq=input_prob_matrix_0,
        beam_size=beam_size,
        vocabulary=vocab_list,
        blank_id=len(vocab_list),
        cutoff_prob=1.0,
        ext_scoring_func=model.score
    )
    #
    # # run log- CTC beam search decoder
    beam_result_log = ctc_beam_search_decoder_log(
        probs_seq=input_prob_matrix_0,
        beam_size=beam_size,
        vocabulary=vocab_list,
        blank_id=len(vocab_list),
        cutoff_prob=1.0,
        ext_scoring_func=model.score
    )
    # compare decoding result
    # print(
    #   "{tf-decoder log probs} \t {org-decoder log probs} \t{log-decoder log probs}:  {tf_decoder result}  {org_decoder result} {log-decoder result}")
    for index in range(len(beam_result)):
        print("bm", beam_result[index][1])
    for index in range(len(beam_result_log)):
        print(beam_result_log[index][1])
        #   tf_result = ''.join([vocab_list[i] for i in tf_decoded[index].values])

        # print(('%6f\t%f\t%f:"%s","%s","%s"') % (tf_log_probs[0][index], beam_result[index][0], beam_result_log[index][0],
        #       tf_result, beam_result[index][1], beam_result_log[index][1]))

        # tf_result_g = ''.join([vocab_list[i] for i in tf_decoded_g[0].values])
        # print("greedy search %.2f '%s'" % (tf_log_probs_g[0][0], tf_result_g))
        # import torch
        # import pytorch_ctc
        #
        # scorer = pytorch_ctc.Scorer()
        # decoder = pytorch_ctc.CTCBeamDecoder(scorer, vocab_list, top_paths=3, beam_width=20,
        #                          blank_index=0, space_index=28, merge_repeated=False)
        #
        # output, score, out_seq_len = decoder.decode(input_prob_matrix_0, sizes=None)
        # print(output, score, out_seq_len)


if __name__ == '__main__':
    # for i in range(1000):

    t_beam_search_decoder()

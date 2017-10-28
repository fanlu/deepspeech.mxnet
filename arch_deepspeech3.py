# pylint: disable=C0111, too-many-statements, too-many-locals
# pylint: too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
"""
architecture file for deep speech 2 model
"""
import json
import math
import argparse
import mxnet as mx

from stt_layer_batchnorm import batchnorm
from stt_layer_conv import conv
from stt_layer_fc import sequence_fc
from stt_layer_gru import bi_gru_unroll, gru_unroll
from stt_layer_lstm import bi_lstm_unroll
from stt_layer_slice import slice_symbol_to_seq_symobls
from stt_layer_warpctc import warpctc_layer


def prepare_data(args, is_val=False):
    """
    set atual shape of data
    """
    rnn_type = args.config.get("arch", "rnn_type")
    num_rnn_layer = args.config.getint("arch", "num_rnn_layer")
    num_hidden_rnn_list = json.loads(args.config.get("arch", "num_hidden_rnn_list"))
    num_hidden_proj = args.config.getint("arch", "num_hidden_proj")
    batch_size = args.config.getint("common", "batch_size")
    if is_val:
        batch_size = args.config.getint("common", "val_batch_size")

    if rnn_type == 'lstm':
        init_c = [('l%d_init_c' % l, (batch_size, num_hidden_rnn_list[l]))
                  for l in range(num_rnn_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l]))
                  for l in range(num_rnn_layer)]
    elif rnn_type == 'bilstm':
        forward_init_c = [('forward_l%d_init_c' % l, (batch_size, num_hidden_rnn_list[l]))
                          for l in range(num_rnn_layer)]
        backward_init_c = [('backward_l%d_init_c' % l, (batch_size, num_hidden_rnn_list[l]))
                           for l in range(num_rnn_layer)]
        init_c = forward_init_c + backward_init_c
        if num_hidden_proj > 0:
            forward_init_h = [('forward_l%d_init_h' % l, (batch_size, num_hidden_proj))
                              for l in range(num_rnn_layer)]
            backward_init_h = [('backward_l%d_init_h' % l, (batch_size, num_hidden_proj))
                               for l in range(num_rnn_layer)]
        else:
            forward_init_h = [('forward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l]))
                              for l in range(num_rnn_layer)]
            backward_init_h = [('backward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l]))
                               for l in range(num_rnn_layer)]
        init_h = forward_init_h + backward_init_h
    elif rnn_type == 'gru':
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l]))
                  for l in range(num_rnn_layer)]
    elif rnn_type == 'bigru':
        forward_init_h = [('forward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l]))
                          for l in range(num_rnn_layer)]
        backward_init_h = [('backward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l]))
                           for l in range(num_rnn_layer)]
        init_h = forward_init_h + backward_init_h
    else:
        raise Exception('network type should be one of the lstm,bilstm,gru,bigru')

    if rnn_type == 'lstm' or rnn_type == 'bilstm':
        init_states = init_c + init_h
    elif rnn_type == 'gru' or rnn_type == 'bigru':
        init_states = init_h
    return init_states


def arch(args, seq_len=None):
    """
    define deep speech 2 network
    """
    if isinstance(args, argparse.Namespace):
        mode = args.config.get("common", "mode")
        is_bucketing = args.config.getboolean("arch", "is_bucketing")
        if mode == "train" or is_bucketing:
            channel_num = args.config.getint("arch", "channel_num")
            conv_layer1_filter_dim = \
                tuple(json.loads(args.config.get("arch", "conv_layer1_filter_dim")))
            conv_layer1_stride = tuple(json.loads(args.config.get("arch", "conv_layer1_stride")))

            conv_layer2_filter_dim = \
                tuple(json.loads(args.config.get("arch", "conv_layer2_filter_dim")))
            conv_layer2_stride = tuple(json.loads(args.config.get("arch", "conv_layer2_stride")))

            conv_layer3_filter_dim = \
                tuple(json.loads(args.config.get("arch", "conv_layer3_filter_dim")))
            conv_layer3_stride = tuple(json.loads(args.config.get("arch", "conv_layer3_stride")))

            rnn_type = args.config.get("arch", "rnn_type")
            num_rnn_layer = args.config.getint("arch", "num_rnn_layer")

            num_hidden_proj = args.config.getint("arch", "num_hidden_proj")

            num_hidden_rnn_list = json.loads(args.config.get("arch", "num_hidden_rnn_list"))

            is_batchnorm = args.config.getboolean("arch", "is_batchnorm")

            if seq_len is None:
                seq_len = args.config.getint('arch', 'max_t_count')

            num_label = args.config.getint('arch', 'max_label_length')

            num_rear_fc_layers = args.config.getint("arch", "num_rear_fc_layers")
            num_hidden_rear_fc_list = json.loads(args.config.get("arch", "num_hidden_rear_fc_list"))
            act_type_rear_fc_list = json.loads(args.config.get("arch", "act_type_rear_fc_list"))
            # model symbol generation
            # input preparation
            data = mx.sym.Variable('data')
            label = mx.sym.Variable('label')

            net = mx.sym.Reshape(data=data, shape=(-4, -1, 1, 0, 0))
            net = conv(net=net,
                       channels=channel_num,
                       filter_dimension=conv_layer1_filter_dim,
                       stride=conv_layer1_stride,
                       no_bias=is_batchnorm,
                       name='conv1')
            if is_batchnorm:
               # batch norm normalizes axis 1
               net = batchnorm(net, name="conv1_batchnorm")

            net = conv(net=net,
                       channels=channel_num,
                       filter_dimension=conv_layer2_filter_dim,
                       stride=conv_layer2_stride,
                       no_bias=is_batchnorm,
                       name='conv2')
            if is_batchnorm:
                # batch norm normalizes axis 1
                net = batchnorm(net, name="conv2_batchnorm")

            net = conv(net=net,
                       channels=96,
                       filter_dimension=conv_layer3_filter_dim,
                       stride=conv_layer3_stride,
                       no_bias=is_batchnorm,
                       name='conv3')
            if is_batchnorm:
                # batch norm normalizes axis 1
                net = batchnorm(net, name="conv3_batchnorm")

            net = mx.sym.transpose(data=net, axes=(0, 2, 1, 3))
            net = mx.sym.Reshape(data=net, shape=(0, 0, -3))
            seq_len_after_conv_layer1 = int(
                math.floor((seq_len - conv_layer1_filter_dim[0]) / conv_layer1_stride[0])) + 1
            seq_len_after_conv_layer2 = int(
                math.floor((seq_len_after_conv_layer1 - conv_layer2_filter_dim[0])
                           / conv_layer2_stride[0])) + 1
            seq_len_after_conv_layer3 = int(
                math.floor((seq_len_after_conv_layer2 - conv_layer3_filter_dim[0])
                           / conv_layer3_stride[0])) + 1
            net = slice_symbol_to_seq_symobls(net=net, seq_len=seq_len_after_conv_layer3, axis=1)
            if rnn_type == "bilstm":
                net, f_states, b_states = bi_lstm_unroll(net=net,
                                     seq_len=seq_len_after_conv_layer3,
                                     num_hidden_lstm_list=num_hidden_rnn_list,
                                     num_lstm_layer=num_rnn_layer,
                                     dropout=0.,
                                     num_hidden_proj=num_hidden_proj,
                                     is_batchnorm=is_batchnorm,
                                     is_bucketing=is_bucketing)
            elif rnn_type == "gru":
                net = gru_unroll(net=net,
                                 seq_len=seq_len_after_conv_layer3,
                                 num_hidden_gru_list=num_hidden_rnn_list,
                                 num_gru_layer=num_rnn_layer,
                                 dropout=0.,
                                 is_batchnorm=is_batchnorm,
                                 is_bucketing=is_bucketing)
            elif rnn_type == "bigru":
                net = bi_gru_unroll(net=net,
                                    seq_len=seq_len_after_conv_layer3,
                                    num_hidden_gru_list=num_hidden_rnn_list,
                                    num_gru_layer=num_rnn_layer,
                                    dropout=0.,
                                    is_batchnorm=is_batchnorm,
                                    is_bucketing=is_bucketing)
            else:
                raise Exception('rnn_type should be one of the followings, bilstm,gru,bigru')

            # rear fc layers
            net = sequence_fc(net=net, seq_len=seq_len_after_conv_layer3,
                              num_layer=num_rear_fc_layers, prefix="rear",
                              num_hidden_list=num_hidden_rear_fc_list,
                              act_type_list=act_type_rear_fc_list,
                              is_batchnorm=is_batchnorm)

            cls_weight = mx.sym.Variable("cls_weight")
            cls_bias = mx.sym.Variable("cls_bias")
            fc_seq = []
            character_classes_count = args.config.getint('arch', 'n_classes') + 1
            for seqidx in range(seq_len_after_conv_layer3):
                hidden = net[seqidx]
                hidden = mx.sym.FullyConnected(data=hidden,
                                               num_hidden=character_classes_count,
                                               weight=cls_weight,
                                               bias=cls_bias)
                fc_seq.append(hidden)
            net = mx.sym.Concat(*fc_seq, dim=0, name="warpctc_layer_concat")

            # warpctc layer
            net = warpctc_layer(net=net,
                                seq_len=seq_len_after_conv_layer3,
                                label=label,
                                num_label=num_label,
                                character_classes_count=
                                (args.config.getint('arch', 'n_classes') + 1))
            args.config.set('arch', 'max_t_count', str(seq_len_after_conv_layer3))
            return net
        elif mode == 'load' or mode == 'predict':
            conv_layer1_filter_dim = \
                tuple(json.loads(args.config.get("arch", "conv_layer1_filter_dim")))
            conv_layer1_stride = tuple(json.loads(args.config.get("arch", "conv_layer1_stride")))
            conv_layer2_filter_dim = \
                tuple(json.loads(args.config.get("arch", "conv_layer2_filter_dim")))
            conv_layer2_stride = tuple(json.loads(args.config.get("arch", "conv_layer2_stride")))
            conv_layer3_filter_dim = \
                tuple(json.loads(args.config.get("arch", "conv_layer3_filter_dim")))
            conv_layer3_stride = tuple(json.loads(args.config.get("arch", "conv_layer3_stride")))
            if seq_len is None:
                seq_len = args.config.getint('arch', 'max_t_count')
            seq_len_after_conv_layer1 = int(
                math.floor((seq_len - conv_layer1_filter_dim[0]) / conv_layer1_stride[0])) + 1
            seq_len_after_conv_layer2 = int(
                math.floor((seq_len_after_conv_layer1 - conv_layer2_filter_dim[0])
                           / conv_layer2_stride[0])) + 1
            seq_len_after_conv_layer3 = int(
                math.floor((seq_len_after_conv_layer2 - conv_layer3_filter_dim[0])
                           / conv_layer3_stride[0])) + 1

            args.config.set('arch', 'max_t_count', str(seq_len_after_conv_layer3))
        else:
            raise Exception('mode must be the one of the followings - train,predict,load')


class BucketingArch(object):
    def __init__(self, args):
        self.args = args

    def sym_gen(self, seq_len):
        args = self.args
        net = arch(args, seq_len)
        init_states = prepare_data(args)
        init_state_names = [x[0] for x in init_states]
        init_state_names.insert(0, 'data')
        return net, init_state_names, ('label',)

    def get_sym_gen(self):
        return self.sym_gen

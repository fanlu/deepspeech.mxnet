import mxnet as mx


def warpctc_layer(net, label, num_label, seq_len, character_classes_count):
    # cls_weight = mx.sym.Variable("cls_weight")
    # cls_bias = mx.sym.Variable("cls_bias")
    # fc_seq = []
    # for seqidx in range(seq_len):
    #     hidden = net[seqidx]
    #     hidden = mx.sym.FullyConnected(data=hidden,
    #                                    num_hidden=character_classes_count,
    #                                    weight=cls_weight,
    #                                    bias=cls_bias)
    #     fc_seq.append(hidden)
    # net = mx.sym.Concat(*fc_seq, dim=0)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')

    net = mx.sym.WarpCTC(data=net, label=label, label_length=num_label, input_length=seq_len,
                         name="warpctc_layer_warpctc")

    return net

    # # pred_fc = mx.sym.FullyConnected(data=net, num_hidden=11)
    # pred_ctc = mx.sym.Reshape(data=net, shape=(-4, seq_len, -1, 0))
    #
    # # _contrib_CTCLoss
    # loss = mx.sym.contrib.ctc_loss(data=pred_ctc, label=label, name='ctc_loss')
    # ctc_loss = mx.sym.MakeLoss(loss)
    #
    # softmax_class = mx.symbol.SoftmaxActivation(data=net)
    # softmax_loss = mx.sym.MakeLoss(softmax_class)
    # softmax_loss = mx.sym.BlockGrad(softmax_loss)
    #
    # return mx.sym.Group([softmax_loss, ctc_loss])

    # return ctc_loss

import mxnet as mx


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
  """
  symbol: the pretrained network symbol
  arg_params: the argument parameters of the pretrained model
  num_classes: the number of classes for the fine-tune datasets
  layer_name: the layer name before the last fully-connected layer
  """
  all_layers = symbol.get_internals()
  net = all_layers[layer_name + '_output']
  net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
  net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
  new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
  return (net, new_args)

if __name__ == "__main__":
  sym, arg_params, aux_params = mx.model.load_checkpoint('checkpoints/deep_bucket', 54)

  (new_sym, new_args) = get_fine_tune_model(sym, arg_params, 2)
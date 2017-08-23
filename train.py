import sys

sys.path.insert(0, "../../python")

from log_util import LogUtil
import os.path
import mxnet as mx
import config_util
# from config_util import get_checkpoint_path, parse_contexts
from stt_metric import STTMetric
#tensorboard setting
from tensorboard import SummaryWriter
import socket
import json
#import stt_bucketing_module
# from stt_bucketing_module import STTBucketingModule


def save_checkpoint(module, prefix, epoch, save_optimizer_states=False):
    symbol, data_names, label_names = module._sym_gen(module._default_bucket_key)
    symbol.save('%s-symbol.json' % prefix)
    param_name = '%s-%04d.params' % (prefix, epoch)
    module.save_params(param_name)
    if save_optimizer_states:
        state_name = '%s-%04d.states' % (prefix, epoch)
        module._curr_module.save_optimizer_states(state_name)


def get_initializer(args):
    init_type = getattr(mx.initializer, args.config.get('train', 'initializer'))
    init_scale = args.config.getfloat('train', 'init_scale')
    if init_type is mx.initializer.Xavier:
        return mx.initializer.Xavier(magnitude=init_scale, factor_type=args.config.get('train', 'factor_type'))
    return init_type(init_scale)

class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """
    def __init__(self, learning_rate=0.001):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        return self.learning_rate


def _get_lr_scheduler(args, kv):
    learning_rate = args.config.getfloat('train', 'learning_rate')
    lr_factor = args.config.getfloat('train', 'lr_factor')
    if lr_factor >= 1:
        return (learning_rate, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    mode = args.config.get('common', 'mode')
    begin_epoch = 0
    if mode == "load":
        model_file = args.config.get('common', 'model_file')
        begin_epoch = int(model_file.split("-")[1]) if len(model_file) == 16 else int(model_file.split("n_epoch")[1].split("n_batch")[0])
    step_epochs = [int(l) for l in args.config.get('train', 'lr_step_epochs').split(',')]
    for s in step_epochs:
        if begin_epoch >= s:
            learning_rate *= lr_factor
    if learning_rate != args.config.getfloat('train', 'learning_rate'):
        log = LogUtil().getlogger()
        log.info('Adjust learning rate to %e for epoch %d' % (learning_rate, begin_epoch))

    steps = [epoch_size * (x - begin_epoch) for x in step_epochs if x - begin_epoch > 0]
    return (learning_rate, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))


def do_training(args, module, data_train, data_val, begin_epoch=0):
    from distutils.dir_util import mkpath

    host_name = socket.gethostname()
    log = LogUtil().getlogger()
    mkpath(os.path.dirname(config_util.get_checkpoint_path(args)))

    #seq_len = args.config.get('arch', 'max_t_count')
    batch_size = args.config.getint('common', 'batch_size')
    save_checkpoint_every_n_epoch = args.config.getint('common', 'save_checkpoint_every_n_epoch')
    save_checkpoint_every_n_batch = args.config.getint('common', 'save_checkpoint_every_n_batch')
    enable_logging_train_metric = args.config.getboolean('train', 'enable_logging_train_metric')
    enable_logging_validation_metric = args.config.getboolean('train', 'enable_logging_validation_metric')

    contexts = config_util.parse_contexts(args)
    num_gpu = len(contexts)
    eval_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu, is_logging=enable_logging_validation_metric,is_epoch_end=True)
    # tensorboard setting
    loss_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu, is_logging=enable_logging_train_metric,is_epoch_end=False)

    optimizer = args.config.get('optimizer', 'optimizer')
    learning_rate = args.config.getfloat('train', 'learning_rate')
    learning_rate_start = args.config.getfloat('train', 'learning_rate_start')
    learning_rate_annealing = args.config.getfloat('train', 'learning_rate_annealing')
    lr_factor = args.config.getfloat('train', 'lr_factor')

    mode = args.config.get('common', 'mode')
    num_epoch = args.config.getint('train', 'num_epoch')
    clip_gradient = args.config.getfloat('optimizer', 'clip_gradient')
    weight_decay = args.config.getfloat('optimizer', 'weight_decay')
    save_optimizer_states = args.config.getboolean('train', 'save_optimizer_states')
    show_every = args.config.getint('train', 'show_every')
    optimizer_params_dictionary = json.loads(args.config.get('optimizer', 'optimizer_params_dictionary'))
    kvstore_option = args.config.get('common', 'kvstore_option')
    n_epoch=begin_epoch
    is_bucketing = args.config.getboolean('arch', 'is_bucketing')

    # kv = mx.kv.create(kvstore_option)
    # data = mx.io.ImageRecordIter(num_parts=kv.num_workers, part_index=kv.rank)
    # # a.set_optimizer(optimizer)
    # updater = mx.optimizer.get_updater(optimizer)
    # a._set_updater(updater=updater)

    if clip_gradient == 0:
        clip_gradient = None
    if is_bucketing and mode == 'load':
        model_file = args.config.get('common', 'model_file')
        model_name = os.path.splitext(model_file)[0]
        model_num_epoch = int(model_name[-4:])

        model_path = 'checkpoints/' + str(model_name[:-5])
        symbol, data_names, label_names = module(1600)
        model = mx.mod.BucketingModule(
            sym_gen=module,
            default_bucket_key=data_train.default_bucket_key,
            context=contexts)
        data_train.reset()

        model.bind(data_shapes=data_train.provide_data,
                   label_shapes=data_train.provide_label,
                   for_training=True)
        _, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_num_epoch)
        model.set_params(arg_params, aux_params)
        module = model
    else:
        module.bind(data_shapes=data_train.provide_data,
                label_shapes=data_train.provide_label,
                for_training=True)

    if begin_epoch == 0 and mode == 'train':
        module.init_params(initializer=get_initializer(args))

    kv = mx.kv.create(kvstore_option)
    lr_scheduler = SimpleLRScheduler(learning_rate=learning_rate)
    # lr, lr_scheduler = _get_lr_scheduler(args, kv)

    def reset_optimizer(force_init=False):
        optimizer_params = {'lr_scheduler': lr_scheduler,
                            'clip_gradient': clip_gradient,
                            'wd': weight_decay}
        optimizer_params.update(optimizer_params_dictionary)
        module.init_optimizer(kvstore=kv,
                              optimizer=optimizer,
                              optimizer_params=optimizer_params,
                              force_init=force_init)
    if mode == "train":
        reset_optimizer(force_init=True)
    else:
        reset_optimizer(force_init=False)
        data_train.reset()
        data_train.is_first_epoch = True

    #tensorboard setting
    tblog_dir = args.config.get('common', 'tensorboard_log_dir')
    summary_writer = SummaryWriter(tblog_dir)

    while True:

        if n_epoch >= num_epoch:
            break
        loss_metric.reset()
        log.info(host_name + '---------train---------')

        step_epochs = [int(l) for l in args.config.get('train', 'lr_step_epochs').split(',')]
        if n_epoch < step_epochs[0]:
            learning_rate_cur = learning_rate_start + n_epoch * (learning_rate - learning_rate_start) / step_epochs[0]
        else:
            learning_rate_cur = learning_rate
            for s in step_epochs:
                if n_epoch >= s:
                    learning_rate_cur *= lr_factor

        lr_scheduler.learning_rate = learning_rate_cur
        log.info("n_epoch %d's lr is %.7f" % (n_epoch, lr_scheduler.learning_rate))
        summary_writer.add_scalar('lr', lr_scheduler.learning_rate, n_epoch)
        for nbatch, data_batch in enumerate(data_train):
            module.forward_backward(data_batch)
            module.update()
            # tensorboard setting
            if (nbatch + 1) % show_every == 0:
                module.update_metric(loss_metric, data_batch.label)
                # print("loss=========== %.2f" % loss_metric.get_batch_loss())
            #summary_writer.add_scalar('loss batch', loss_metric.get_batch_loss(), nbatch)
            if (nbatch+1) % save_checkpoint_every_n_batch == 0:
                log.info('Epoch[%d] Batch[%d] SAVE CHECKPOINT', n_epoch, nbatch)
                save_checkpoint(module, prefix=config_util.get_checkpoint_path(args)+"n_epoch"+str(n_epoch)+"n_batch", epoch=(int((nbatch+1)/save_checkpoint_every_n_batch)-1), save_optimizer_states=save_optimizer_states)
        # commented for Libri_sample data set to see only train cer
        log.info(host_name + '---------validation---------')
        data_val.reset()
        eval_metric.reset()
        for nbatch, data_batch in enumerate(data_val):
            # when is_train = False it leads to high cer when batch_norm
            module.forward(data_batch, is_train=True)
            module.update_metric(eval_metric, data_batch.label)

        # tensorboard setting
        val_cer, val_n_label, val_l_dist, val_ctc_loss = eval_metric.get_name_value()
        log.info("Epoch[%d] val cer=%f (%d / %d), ctc_loss=%f", n_epoch, val_cer, int(val_n_label - val_l_dist), val_n_label, val_ctc_loss)
        curr_acc = val_cer
        summary_writer.add_scalar('CER validation', val_cer, n_epoch)
        summary_writer.add_scalar('loss validation', val_ctc_loss, n_epoch)
        assert curr_acc is not None, 'cannot find Acc_exclude_padding in eval metric'

        data_train.reset()
        data_train.is_first_epoch = False

        # tensorboard setting
        train_cer, train_n_label, train_l_dist, train_ctc_loss = loss_metric.get_name_value()
        summary_writer.add_scalar('loss epoch', train_ctc_loss, n_epoch)
        summary_writer.add_scalar('CER train', train_cer, n_epoch)

        # save checkpoints
        if n_epoch % save_checkpoint_every_n_epoch == 0:
            log.info('Epoch[%d] SAVE CHECKPOINT', n_epoch)
            save_checkpoint(module, prefix=config_util.get_checkpoint_path(args), epoch=n_epoch, save_optimizer_states=save_optimizer_states)

        n_epoch += 1

    log.info('FINISH')

import json
import os
import sys
from collections import namedtuple
from datetime import datetime
from config_util import parse_args, parse_contexts, generate_file_path
from train import do_training
import mxnet as mx
from stt_io_iter import STTIter
from label_util import LabelUtil
from log_util import LogUtil
import numpy as np
from stt_datagenerator import DataGenerator
from stt_metric import STTMetric, EvalSTTMetric
from stt_bi_graphemes_util import generate_bi_graphemes_dictionary
from stt_bucketing_module import STTBucketingModule
from stt_io_bucketingiter import BucketSTTIter
import posixpath
import BaseHTTPServer
import urllib
import cgi
import shutil
import mimetypes
import re
import array
import wave
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from create_desc_json import get_duration_wave

from io import BytesIO

# os.environ['MXNET_ENGINE_TYPE'] = "NaiveEngine"
os.environ['MXNET_ENGINE_TYPE'] = "ThreadedEnginePerDevice"
os.environ['MXNET_ENABLE_GPU_P2P'] = "0"


class WHCS:
    width = 0
    height = 0
    channel = 0
    stride = 0


class ConfigLogger(object):
    def __init__(self, log):
        self.__log = log

    def __call__(self, config):
        self.__log.info("Config:")
        config.write(self)

    def write(self, data):
        # stripping the data makes the output nicer and avoids empty lines
        line = data.strip()
        self.__log.info(line)


from main import load_labelutil


def load_data(args, wav_file):
    mode = args.config.get('common', 'mode')
    if mode not in ['train', 'predict', 'load']:
        raise Exception('mode must be the one of the followings - train,predict,load')
    batch_size = args.config.getint('common', 'batch_size')

    whcs = WHCS()
    whcs.width = args.config.getint('data', 'width')
    whcs.height = args.config.getint('data', 'height')
    whcs.channel = args.config.getint('data', 'channel')
    whcs.stride = args.config.getint('data', 'stride')
    save_dir = 'checkpoints'
    model_name = args.config.get('common', 'prefix')
    is_bi_graphemes = args.config.getboolean('common', 'is_bi_graphemes')
    overwrite_meta_files = args.config.getboolean('train', 'overwrite_meta_files')
    overwrite_bi_graphemes_dictionary = args.config.getboolean('train', 'overwrite_bi_graphemes_dictionary')
    max_duration = args.config.getfloat('data', 'max_duration')
    max_freq = args.config.getint('data', 'max_freq')
    language = args.config.get('data', 'language')

    log = LogUtil().getlogger()
    labelUtil = LabelUtil()

    # test_json = "resources/d.json"
    datagen = DataGenerator(save_dir=save_dir, model_name=model_name, max_freq=max_freq)
    datagen.train_audio_paths = [wav_file]
    datagen.train_durations = [get_duration_wave(wav_file)]
    datagen.train_texts = [""]
    # datagen.load_train_data(test_json, max_duration=max_duration)
    labelutil = load_labelutil(labelUtil, is_bi_graphemes, language="zh")
    args.config.set('arch', 'n_classes', str(labelUtil.get_count()))
    datagen.get_meta_from_file(
        np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
        np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))

    is_batchnorm = args.config.getboolean('arch', 'is_batchnorm')
    if batch_size == 1 and is_batchnorm and (mode == 'train' or mode == 'load'):
        raise Warning('batch size 1 is too small for is_batchnorm')

    max_t_count = datagen.get_max_seq_length(partition="test")
    max_label_length = \
        datagen.get_max_label_length(partition="test", is_bi_graphemes=is_bi_graphemes)

    args.config.set('arch', 'max_t_count', str(max_t_count))
    args.config.set('arch', 'max_label_length', str(max_label_length))
    from importlib import import_module
    prepare_data_template = import_module(args.config.get('arch', 'arch_file'))
    init_states = prepare_data_template.prepare_data(args)
    sort_by_duration = (mode == "train")
    is_bucketing = args.config.getboolean('arch', 'is_bucketing')
    save_feature_as_csvfile = args.config.getboolean('train', 'save_feature_as_csvfile')
    if is_bucketing:
        buckets = json.loads(args.config.get('arch', 'buckets'))
        data_loaded = BucketSTTIter(partition="train",
                                    count=datagen.count,
                                    datagen=datagen,
                                    batch_size=batch_size,
                                    num_label=max_label_length,
                                    init_states=init_states,
                                    seq_length=max_t_count,
                                    width=whcs.width,
                                    height=whcs.height,
                                    sort_by_duration=sort_by_duration,
                                    is_bi_graphemes=is_bi_graphemes,
                                    buckets=buckets,
                                    save_feature_as_csvfile=save_feature_as_csvfile)
    else:
        data_loaded = STTIter(partition="train",
                              count=datagen.count,
                              datagen=datagen,
                              batch_size=batch_size,
                              num_label=max_label_length,
                              init_states=init_states,
                              seq_length=max_t_count,
                              width=whcs.width,
                              height=whcs.height,
                              sort_by_duration=sort_by_duration,
                              is_bi_graphemes=is_bi_graphemes,
                              save_feature_as_csvfile=save_feature_as_csvfile)

    return data_loaded, args


class SimpleHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    # Simple HTTP request handler with POST commands.

    def do_POST(self):
        # print self.headers['Content-Type']
        # print self.rfile
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })
        filename = form['file'].filename
        print("filename is: " + str(filename))
        output_file_pre = "/export/aiplatform/fanlu/yuyin_test/"
        part1, part2 = filename.split(".")
        if filename.endswith(".speex"):
            data = form['file'].file.read()
            open("./" + filename, "wb").write(data)
            command = "./SpeexDecode " + filename + " " + part1 + ".wav"
            os.system(command)
            data = open(part1 + ".wav", 'rb').read()
            open("./lolol.wav", "wb").write(data)

        elif filename.endswith(".amr"):
            data = form['file'].file.read()
            open(output_file_pre + filename, "wb").write(data)
            command = "ffmpeg -y -i " + output_file_pre + part1 + ".amr -acodec pcm_s16le -ar 16000 -ac 1 -b 256 " + output_file_pre + part1 + ".wav"
            os.system(command)

        elif filename.endswith(".wav"):
            data = form['file'].file.read()
            open(output_file_pre + part1 + ".wav", "wb").write(data)

        # create_desc_json.ai_2_word_single(output_file_pre + part1 + ".wav")
        trans_res = otherNet.getTrans(output_file_pre + part1 + ".wav")
        content = bytes(trans_res.encode("utf-8"))
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)


def load_model(args, contexts, data_train):
    # load model from model_name prefix and epoch of model_num_epoch with gpu contexts of contexts
    mode = args.config.get('common', 'mode')
    load_optimizer_states = args.config.getboolean('load', 'load_optimizer_states')
    is_start_from_batch = args.config.getboolean('load', 'is_start_from_batch')

    from importlib import import_module
    symbol_template = import_module(args.config.get('arch', 'arch_file'))
    is_bucketing = args.config.getboolean('arch', 'is_bucketing')

    if mode == 'train':
        if is_bucketing:
            bucketing_arch = symbol_template.BucketingArch(args)
            model_loaded = bucketing_arch.get_sym_gen()
        else:
            model_loaded = symbol_template.arch(args)
        model_num_epoch = None
    elif mode == 'load' or mode == 'predict':
        model_file = args.config.get('common', 'model_file')
        model_name = os.path.splitext(model_file)[0]
        model_num_epoch = int(model_name[-4:])
        if is_bucketing:
            bucketing_arch = symbol_template.BucketingArch(args)
            model_loaded = bucketing_arch.get_sym_gen()
        else:
            model_path = 'checkpoints/' + str(model_name[:-5])

            data_names = [x[0] for x in data_train.provide_data]
            label_names = [x[0] for x in data_train.provide_label]

            model_loaded = mx.module.Module.load(
                prefix=model_path, epoch=model_num_epoch, context=contexts,
                data_names=data_names, label_names=label_names,
                load_optimizer_states=load_optimizer_states)
        if is_start_from_batch:
            import re
            model_num_epoch = int(re.findall('\d+', model_file)[0])

    return model_loaded, model_num_epoch


class Net(object):
    def __init__(self):
        if len(sys.argv) <= 1:
            raise Exception('cfg file path must be provided. ' +
                            'ex)python main.py --configfile examplecfg.cfg')
        self.args = parse_args(sys.argv[1])
        # set parameters from cfg file
        # give random seed
        self.random_seed = self.args.config.getint('common', 'random_seed')
        self.mx_random_seed = self.args.config.getint('common', 'mx_random_seed')
        # random seed for shuffling data list
        if self.random_seed != -1:
            np.random.seed(self.random_seed)
        # set mx.random.seed to give seed for parameter initialization
        if self.mx_random_seed != -1:
            mx.random.seed(self.mx_random_seed)
        else:
            mx.random.seed(hash(datetime.now()))
        # set log file name
        self.log_filename = self.args.config.get('common', 'log_filename')
        self.log = LogUtil(filename=self.log_filename).getlogger()

        # set parameters from data section(common)
        self.mode = self.args.config.get('common', 'mode')

        # get meta file where character to number conversions are defined

        self.contexts = parse_contexts(self.args)
        self.num_gpu = len(self.contexts)
        self.batch_size = self.args.config.getint('common', 'batch_size')
        # check the number of gpus is positive divisor of the batch size for data parallel
        self.data_train, self.args = load_data(self.args)
        self.is_batchnorm = self.args.config.getboolean('arch', 'is_batchnorm')
        self.is_bucketing = self.args.config.getboolean('arch', 'is_bucketing')

        # log current config
        self.config_logger = ConfigLogger(self.log)
        self.config_logger(self.args.config)

        # load model
        self.model_loaded, self.model_num_epoch = load_model(self.args, self.contexts, self.data_train)

        self.max_t_count = self.args.config.getint('arch', 'max_t_count')
        self.load_optimizer_states = self.args.config.getboolean('load', 'load_optimizer_states')
        self.model_file = self.args.config.get('common', 'model_file')
        self.model_name = os.path.splitext(self.model_file)[0]
        self.model_num_epoch = int(self.model_name[-4:])

        self.model_path = 'checkpoints/' + str(self.model_name[:-5])

        self.model = STTBucketingModule(
            sym_gen=self.model_loaded,
            default_bucket_key=1600,
            context=self.contexts
        )
        _, self.arg_params, self.aux_params = mx.model.load_checkpoint(self.model_path, self.model_num_epoch)

    def getTrans(self, wav_file):
        self.data_train, self.args = load_data(self.args, wav_file)

        self.model.bind(data_shapes=self.data_train.provide_data,
                        label_shapes=self.data_train.provide_label,
                        for_training=True)
        self.model.set_params(self.arg_params, self.aux_params, allow_extra=True, allow_missing=True)

        # self.model.set_params(self.arg_params, self.aux_params)

        # backward_t358_l1_batchnorm_moving_var

        model_loaded = self.model
        max_t_count = self.args.config.getint('arch', 'max_t_count')
        eval_metric = EvalSTTMetric(batch_size=self.batch_size, num_gpu=self.num_gpu, is_logging=True)
        for nbatch, data_batch in enumerate(self.data_train):
            model_loaded.forward(data_batch, is_train=False)
            model_loaded.update_metric(eval_metric, data_batch.label)
            print("my res is:")
            print(eval_metric.placeholder)
            return eval_metric.placeholder

    def printStuff(self):
        print("adfe")


otherNet = Net()

if __name__ == '__main__':
    server = HTTPServer(('', 8088), SimpleHTTPRequestHandler)
    print('Started httpserver on port')

    # Wait forever for incoming htto requests
    server.serve_forever()

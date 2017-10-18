# -*- coding: utf-8 -*-
import BaseHTTPServer
import cgi
import json
import os
import sys
from BaseHTTPServer import HTTPServer
from datetime import datetime

import mxnet as mx
import numpy as np
import time
from config_util import parse_args, parse_contexts, generate_file_path
from create_desc_json import get_duration_wave
from label_util import LabelUtil
from log_util import LogUtil
from stt_bucketing_module import STTBucketingModule
from stt_datagenerator import DataGenerator
from stt_io_bucketingiter import BucketSTTIter
from stt_io_iter import STTIter
from stt_metric import EvalSTTMetric
from ctc_beam_search_decoder import ctc_beam_search_decoder_log

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
    datagen.train_texts = ["1 1"]
    datagen.count = 1
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
        otherNet.log.info("filename is: " + str(filename))
        output_file_pre = "/Users/lonica/Downloads/wav/"
        part1, part2 = filename.rsplit(".", 1)
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

        elif filename.lower().endswith(".wav"):
            data = form['file'].file
            # import soundfile as sf
            # audio, sr1 = sf.read(data, dtype='float32')
            open(output_file_pre + part1 + ".wav", "wb").write(data.read())

        # create_desc_json.ai_2_word_single(output_file_pre + part1 + ".wav")
        otherNet.log.info("start")
        trans_res = otherNet.getTrans(output_file_pre + part1 + ".wav")
        otherNet.log.info("end")
        content = bytes(u"没有检测到语音，请重新录制".encode("utf-8"))
        if trans_res:
            content = bytes(trans_res.encode("utf-8"))
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)


def load_model(args):
    # load model from model_name prefix and epoch of model_num_epoch with gpu contexts of contexts
    is_start_from_batch = args.config.getboolean('load', 'is_start_from_batch')

    from importlib import import_module
    symbol_template = import_module(args.config.get('arch', 'arch_file'))

    model_file = args.config.get('common', 'model_file')
    model_name = os.path.splitext(model_file)[0]
    model_num_epoch = int(model_name[-4:])

    model_path = 'checkpoints/' + str(model_name[:-5])

    bucketing_arch = symbol_template.BucketingArch(args)
    model_loaded = bucketing_arch.get_sym_gen()

    return model_loaded, model_num_epoch, model_path


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
        self.is_batchnorm = self.args.config.getboolean('arch', 'is_batchnorm')
        self.is_bucketing = self.args.config.getboolean('arch', 'is_bucketing')

        # log current config
        self.config_logger = ConfigLogger(self.log)
        self.config_logger(self.args.config)


        default_bucket_key = 1600
        self.args.config.set('arch', 'max_t_count', str(default_bucket_key))
        self.args.config.set('arch', 'max_label_length', str(100))
        self.labelUtil = LabelUtil()
        is_bi_graphemes = self.args.config.getboolean('common', 'is_bi_graphemes')
        load_labelutil(self.labelUtil, is_bi_graphemes, language="zh")
        self.args.config.set('arch', 'n_classes', str(self.labelUtil.get_count()))
        self.max_t_count = self.args.config.getint('arch', 'max_t_count')
        # self.load_optimizer_states = self.args.config.getboolean('load', 'load_optimizer_states')

        # load model
        self.model_loaded, self.model_num_epoch, self.model_path = load_model(self.args)

        self.model = STTBucketingModule(
            sym_gen=self.model_loaded,
            default_bucket_key=default_bucket_key,
            context=self.contexts
        )

        from importlib import import_module
        prepare_data_template = import_module(self.args.config.get('arch', 'arch_file'))
        init_states = prepare_data_template.prepare_data(self.args)
        width = self.args.config.getint('data', 'width')
        height = self.args.config.getint('data', 'height')
        self.model.bind(data_shapes=[('data', (self.batch_size, default_bucket_key, width * height))] + init_states,
                        label_shapes=[
                            ('label', (self.batch_size, self.args.config.getint('arch', 'max_label_length')))],
                        for_training=True)

        _, self.arg_params, self.aux_params = mx.model.load_checkpoint(self.model_path, self.model_num_epoch)
        self.model.set_params(self.arg_params, self.aux_params, allow_extra=True, allow_missing=True)

        try:
            from swig_wrapper import Scorer

            vocab_list = [chars.encode("utf-8") for chars in self.labelUtil.byList]
            self.log.info("vacab_list len is %d" % len(vocab_list))
            _ext_scorer = Scorer(0.26, 0.1, self.args.config.get('common', 'kenlm'), vocab_list)
            lm_char_based = _ext_scorer.is_character_based()
            lm_max_order = _ext_scorer.get_max_order()
            lm_dict_size = _ext_scorer.get_dict_size()
            self.log.info("language model: "
                          "is_character_based = %d," % lm_char_based +
                          " max_order = %d," % lm_max_order +
                          " dict_size = %d" % lm_dict_size)
            self.scorer = _ext_scorer
            # self.eval_metric = EvalSTTMetric(batch_size=self.batch_size, num_gpu=self.num_gpu, is_logging=True,
            #                                  scorer=_ext_scorer)
        except ImportError:
            import kenlm
            km = kenlm.Model(self.args.config.get('common', 'kenlm'))
            self.scorer = km.score
            # self.eval_metric = EvalSTTMetric(batch_size=self.batch_size, num_gpu=self.num_gpu, is_logging=True,
            #                                  scorer=km.score)

    def getTrans(self, wav_file):
        self.data_train, self.args = load_data(self.args, wav_file)

        # self.model.set_params(self.arg_params, self.aux_params)

        # backward_t358_l1_batchnorm_moving_var

        model_loaded = self.model
        max_t_count = self.args.config.getint('arch', 'max_t_count')

        for nbatch, data_batch in enumerate(self.data_train):
            st = time.time()
            model_loaded.forward(data_batch, is_train=False)
            probs = model_loaded.get_outputs()[0].asnumpy()
            self.log.info("forward cost %.2f" % (time.time() - st))
            beam_size = 5
            try:
                from swig_wrapper import ctc_beam_search_decoder

                st2 = time.time()
                vocab_list = [chars.encode("utf-8") for chars in self.labelUtil.byList]
                beam_search_results = ctc_beam_search_decoder(
                    probs_seq=np.array(probs),
                    vocabulary=vocab_list,
                    beam_size=beam_size,
                    blank_id=0,
                    ext_scoring_func=self.scorer,
                    cutoff_prob=0.99,
                    cutoff_top_n=40
                )
                results = [result[1] for result in beam_search_results]
                self.log.info("decode by cpp cost %.2fs:\n%s" % (time.time() - st2, "\n".join(results)))
                res_str = "\n".join(results)
            except ImportError:
                st = time.time()

                beam_result = ctc_beam_search_decoder_log(
                    probs_seq=probs,
                    beam_size=beam_size,
                    vocabulary=self.labelUtil.byIndex,
                    blank_id=0,
                    cutoff_prob=0.99,
                    ext_scoring_func=self.scorer
                )
                st1 = time.time() - st
                results = [result[1] for result in beam_result]
                res_str = "\n".join(results)
                self.log.info("decode by py cost %.2fs:\n%s" % (st1, res_str))
        return res_str


otherNet = Net()

if __name__ == '__main__':
    server = HTTPServer(('', 8089), SimpleHTTPRequestHandler)
    otherNet.log.info('Started httpserver on port')

    # Wait forever for incoming htto requests
    server.serve_forever()

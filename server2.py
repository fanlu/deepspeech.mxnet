# -*- coding: utf-8 -*-
import BaseHTTPServer
import bisect
import cgi
import json
import os
import sys
import time
from BaseHTTPServer import HTTPServer
from datetime import datetime

import mxnet as mx
import numpy as np
import subprocess
from config_util import parse_args, parse_contexts, generate_file_path
from ctc_beam_search_decoder import ctc_beam_search_decoder_log
from label_util import LabelUtil
from log_util import LogUtil
from main import load_labelutil
from stt_datagenerator import DataGenerator
from stt_metric import ctc_greedy_decode
from stt_utils import spectrogram_from_file

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
        log.info("filename is: " + str(filename))
        output_file_pre = args.config.get("common", "wav_dir")
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
            subprocess.call(
                ["ffmpeg", "-y", "-i", output_file_pre + part1 + ".amr", "-acodec", "pcm_s16le", "-ar", "16000", "-ac",
                 "1", output_file_pre + part1 + ".wav"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, shell=False)

        elif filename.lower().endswith(".wav"):
            data = form['file'].file
            # import soundfile as sf
            # audio, sr1 = sf.read(data, dtype='float32')
            open(output_file_pre + part1 + ".wav", "wb").write(data.read())

        # create_desc_json.ai_2_word_single(output_file_pre + part1 + ".wav")
        trans_res = otherNet.getTrans(output_file_pre + part1 + ".wav")
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
    def __init__(self, args):
        self.args = args
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
        self.config_logger = ConfigLogger(log)
        self.config_logger(args.config)

        save_dir = 'checkpoints'
        model_name = self.args.config.get('common', 'prefix')
        max_freq = self.args.config.getint('data', 'max_freq')
        self.datagen = DataGenerator(save_dir=save_dir, model_name=model_name, max_freq=max_freq)
        self.datagen.get_meta_from_file(
            np.loadtxt(generate_file_path(save_dir, model_name, 'feats_mean')),
            np.loadtxt(generate_file_path(save_dir, model_name, 'feats_std')))

        self.buckets = json.loads(self.args.config.get('arch', 'buckets'))

        default_bucket_key = self.buckets[-1]
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
        symbol, self.arg_params, self.aux_params = mx.model.load_checkpoint(self.model_path, self.model_num_epoch)
        # all_layers = symbol.get_internals()
        # s_sym = all_layers['concat36457_output']
        # sm = mx.sym.SoftmaxOutput(data=s_sym, name='softmax')

        # self.model = STTBucketingModule(
        #     sym_gen=self.model_loaded,
        #     default_bucket_key=default_bucket_key,
        #     context=self.contexts
        # )
        s_mod = mx.mod.BucketingModule(sym_gen=self.model_loaded, context=self.contexts,
                                       default_bucket_key=default_bucket_key)

        from importlib import import_module
        prepare_data_template = import_module(self.args.config.get('arch', 'arch_file'))
        self.init_states = prepare_data_template.prepare_data(self.args)
        self.width = self.args.config.getint('data', 'width')
        self.height = self.args.config.getint('data', 'height')
        s_mod.bind(
            data_shapes=[('data', (self.batch_size, default_bucket_key, self.width * self.height))] + self.init_states,
            for_training=False)

        s_mod.set_params(self.arg_params, self.aux_params, allow_extra=True, allow_missing=True)
        for bucket in self.buckets:
            provide_data = [('data', (self.batch_size, bucket, self.width * self.height))] + self.init_states
            s_mod.switch_bucket(bucket_key=bucket, data_shapes=provide_data)

        self.model = s_mod

        try:
            from swig_wrapper import Scorer

            vocab_list = [chars.encode("utf-8") for chars in self.labelUtil.byList]
            log.info("vacab_list len is %d" % len(vocab_list))
            _ext_scorer = Scorer(0.26, 0.1, self.args.config.get('common', 'kenlm'), vocab_list)
            lm_char_based = _ext_scorer.is_character_based()
            lm_max_order = _ext_scorer.get_max_order()
            lm_dict_size = _ext_scorer.get_dict_size()
            log.info("language model: "
                     "is_character_based = %d," % lm_char_based +
                     " max_order = %d," % lm_max_order +
                     " dict_size = %d" % lm_dict_size)
            self.scorer = _ext_scorer
            # self.eval_metric = EvalSTTMetric(batch_size=self.batch_size, num_gpu=self.num_gpu, is_logging=True,
            #                                  scorer=_ext_scorer)
        except ImportError:
            import kenlm
            km = kenlm.Model(self.args.config.get('common', 'kenlm'))
            # self.eval_metric = EvalSTTMetric(batch_size=self.batch_size, num_gpu=self.num_gpu, is_logging=True,
            #                                  scorer=km.score)
            self.scorer = km.score

    def getTrans(self, wav_file):
        res = spectrogram_from_file(wav_file, noise_percent=0)
        buck = bisect.bisect_left(self.buckets, len(res))
        bucket_key = self.buckets[buck]
        res = self.datagen.normalize(res)
        d = np.zeros((self.batch_size, bucket_key, res.shape[1]))
        d[0, :res.shape[0], :] = res
        init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]

        model_loaded = self.model

        provide_data = [('data', (self.batch_size, bucket_key, self.width * self.height))] + self.init_states
        data_batch = mx.io.DataBatch([mx.nd.array(d)] + init_state_arrays, label=None, bucket_key=bucket_key,
                                     provide_data=provide_data, provide_label=None)
        st = time.time()
        model_loaded.forward(data_batch, is_train=False)
        probs = model_loaded.get_outputs()[0].asnumpy()
        log.info("forward cost %.3f" % (time.time() - st))
        st = time.time()
        res = ctc_greedy_decode(probs, self.labelUtil.byList)
        log.info("greedy decode cost %.3f, result is:\n%s" % (time.time() - st, res))
        beam_size = 5
        from stt_metric import ctc_beam_decode
        st = time.time()
        results = ctc_beam_decode(scorer=self.scorer, beam_size=beam_size, vocab=self.labelUtil.byList, probs=probs)
        log.info("beam decode cost %.3f, result is:\n%s" % (time.time() - st, "\n".join(results)))
        return "greedy:\n" + res + "\nbeam:\n" + "\n".join(results)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception('cfg file path must be provided. ' +
                        'ex)python main.py --configfile examplecfg.cfg')
    args = parse_args(sys.argv[1])
    # set log file name
    log_filename = args.config.get('common', 'log_filename')
    log = LogUtil(filename=log_filename).getlogger()
    otherNet = Net(args)

    server = HTTPServer(('', args.config.getint('common', 'port')), SimpleHTTPRequestHandler)
    log.info('Started httpserver on port')
    # Wait forever for incoming htto requests
    server.serve_forever()

# coding=utf-8

from __future__ import absolute_import, division, print_function

import json
import random
import socket
from io import open
import numpy as np
import concurrent.futures
import time

from stt_utils import calc_feat_dim, spectrogram_from_file

from config_util import generate_file_path
from log_util import LogUtil
from label_util import LabelUtil
from stt_bi_graphemes_util import generate_bi_graphemes_label
from stt_phone_util import generate_phone_label, generate_zi_label, generate_py_label
from multiprocessing import cpu_count, Process, Manager, Pool


class DataGenerator(object):
    def __init__(self, save_dir, model_name, step=10, window=20, max_freq=4000, desc_file=None):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        # calc_feat_dim returns int(0.001*window*max_freq)+1
        super(DataGenerator, self).__init__()
        # feat_dim=0.001*20*8000+1=161
        self.feat_dim = calc_feat_dim(window, max_freq)
        # 1d 161 length of array filled with zeros
        self.feats_mean = np.zeros((self.feat_dim,))
        # 1d 161 length of array filled with 1s
        self.feats_std = np.ones((self.feat_dim,))
        self.max_input_length = 0
        self.max_length_list_in_batch = []
        # 1d 161 length of array filled with random value
        # [0.0, 1.0)
        host_name = socket.gethostname()
        self.rng = random.Random(hash(host_name))
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.save_dir = save_dir
        self.model_name = model_name

    def get_meta_from_file(self, feats_mean, feats_std):
        self.feats_mean = feats_mean
        self.feats_std = feats_std

    def featurize(self, audio_clip, overwrite=False, save_feature_as_csvfile=False, noise_percent=0.4, seq_length=-1):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq, overwrite=overwrite,
            save_feature_as_csvfile=save_feature_as_csvfile, noise_percent=noise_percent, seq_length=seq_length)

    def load_metadata_from_desc_file(self, desc_file, partition='train',
                                     max_duration=16.0, ):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        logger = LogUtil().getlogger()
        logger.info('Reading description file: {} for partition: {}'
                    .format(desc_file, partition))
        audio_paths, durations, texts = [], [], []
        with open(desc_file, 'rt', encoding='UTF-8') as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    logger.warn(str(e))
                    logger.warn('Error reading line num #{}'.format(line_num))
                    logger.warn('line {}'.format(json_line.decode("utf-8")))

        if partition == 'train':
            self.count = len(audio_paths)
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
            self.val_count = len(audio_paths)
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")

    def load_train_data(self, desc_file, max_duration):
        self.load_metadata_from_desc_file(desc_file, 'train', max_duration=max_duration)

    def load_validation_data(self, desc_file, max_duration):
        self.load_metadata_from_desc_file(desc_file, 'validation', max_duration=max_duration)

    @staticmethod
    def sort_by_duration(durations, audio_paths, texts):
        return zip(*sorted(zip(durations, audio_paths, texts)))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def normalize_self(self, feature, eps=1e-14):
        return (feature - np.mean(feature, axis=1)[:, np.newaxis]) / (np.sqrt(
            np.sum(np.square(feature), axis=1)) + eps)[:, np.newaxis]

    def get_max_label_length(self, partition, is_bi_graphemes=False, language="zh", zh_type="zi"):
        if partition == 'train':
            texts = self.train_texts + self.val_texts
        elif partition == 'test':
            texts = self.train_texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")
        if language == "en" and is_bi_graphemes:
            self.max_label_length = max([len(generate_bi_graphemes_label(text)) for text in texts])
        elif language == "zh" and zh_type == "phone":
            self.max_label_length = max([len(generate_phone_label(text)) for text in texts])
        elif language == "zh" and zh_type == "py":
            self.max_label_length = max([len(generate_py_label(text)) for text in texts])
        else:
            self.max_label_length = max([len(text) for text in texts])
        return self.max_label_length

    def get_max_seq_length(self, partition):
        if partition == 'train':
            audio_paths = self.train_audio_paths + self.val_audio_paths
            durations = self.train_durations + self.val_durations
        elif partition == 'test':
            audio_paths = self.train_audio_paths
            durations = self.train_durations
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")
        max_duration_indexes = durations.index(max(durations))
        max_seq_length = self.featurize(audio_paths[max_duration_indexes]).shape[0]
        self.max_seq_length = max_seq_length
        return max_seq_length

    def prepare_minibatch(self, audio_paths, texts, overwrite=False,
                          is_bi_graphemes=False, seq_length=-1, save_feature_as_csvfile=False, language="en",
                          zh_type="zi", noise_percent=0.4):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts), \
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [
            self.featurize(a, overwrite=overwrite, save_feature_as_csvfile=save_feature_as_csvfile,
                           noise_percent=noise_percent, seq_length=seq_length) for a in
            audio_paths]
        input_lengths = [f.shape[0] for f in features]
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        if seq_length == -1:
            x = np.zeros((mb_size, self.max_seq_length, feature_dim))
        else:
            x = np.zeros((mb_size, seq_length, feature_dim))
        y = np.zeros((mb_size, self.max_label_length))
        labelUtil = LabelUtil()
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat  # padding with 0 padding with noise？
            if language == "en" and is_bi_graphemes:
                label = generate_bi_graphemes_label(texts[i])
                label = labelUtil.convert_bi_graphemes_to_num(label)
                y[i, :len(label)] = label
            elif language == "en" and not is_bi_graphemes:
                label = labelUtil.convert_word_to_num(texts[i])
                y[i, :len(texts[i])] = label
            elif language == "zh" and zh_type == "phone":
                label = generate_phone_label(texts[i])
                label = labelUtil.convert_bi_graphemes_to_num(label)
                y[i, :len(label)] = label
            elif language == "zh" and zh_type == "py":
                label = generate_py_label(texts[i])
                label = labelUtil.convert_bi_graphemes_to_num(label)
                y[i, :len(label)] = label
            elif language == "zh" and zh_type == "zi":
                label = generate_zi_label(texts[i])
                label = labelUtil.convert_bi_graphemes_to_num(label)
                y[i, :len(label)] = label
            label_lengths.append(len(label))
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths,  # list(int) Length of each label
        }

    def iterate_test(self, minibatch_size=16):
        return self.iterate(self.test_audio_paths, self.test_texts,
                            minibatch_size)

    def iterate_validation(self, minibatch_size=16):
        return self.iterate(self.val_audio_paths, self.val_texts,
                            minibatch_size)

    def preprocess_sample_normalize(self, threadIndex, audio_paths, overwrite, noise_percent, return_dict):
        if len(audio_paths) > 0:
            audio_clip = audio_paths[0]
            feat = self.featurize(audio_clip=audio_clip, overwrite=overwrite)
            feat_squared = np.square(feat)
            count = float(feat.shape[0])
            dim = feat.shape[1]
            if len(audio_paths) > 1:
                for audio_path in audio_paths[1:]:
                    next_feat = self.featurize(audio_clip=audio_path, overwrite=overwrite, noise_percent=noise_percent)
                    next_feat_squared = np.square(next_feat)
                    feat_vertically_stacked = np.concatenate((feat, next_feat)).reshape(-1, dim)
                    feat = np.sum(feat_vertically_stacked, axis=0, keepdims=True)
                    feat_squared_vertically_stacked = np.concatenate(
                        (feat_squared, next_feat_squared)).reshape(-1, dim)
                    feat_squared = np.sum(feat_squared_vertically_stacked, axis=0, keepdims=True)
                    count += float(next_feat.shape[0])
            return_dict[threadIndex] = {'feat': feat, 'feat_squared': feat_squared, 'count': count}

    def sample_normalize(self, k_samples=1000, overwrite=False, noise_percent=0.4):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        log = LogUtil().getlogger()
        log.info("Calculating mean and std from samples")
        # if k_samples is negative then it goes through total dataset
        if k_samples < 0:
            audio_paths = self.train_audio_paths * 10

        # using sample
        else:
            k_samples = min(k_samples, len(self.train_audio_paths))
            samples = self.rng.sample(self.train_audio_paths, k_samples)
            audio_paths = samples
        # manager = Manager()
        # return_dict = manager.dict()
        # jobs = []
        # for threadIndex in range(cpu_count()):
        #     proc = Process(target=self.preprocess_sample_normalize,
        #                    args=(threadIndex, audio_paths, overwrite, noise_percent, return_dict))
        #     jobs.append(proc)
        #     proc.start()
        # for proc in jobs:
        #     proc.join()

        # return_dict = {}
        # self.preprocess_sample_normalize(1, audio_paths, overwrite, noise_percent, return_dict)

        pool = Pool(processes=cpu_count())
        results = []
        for i, f in enumerate(audio_paths):
            result = pool.apply_async(spectrogram_from_file, args=(f,), kwds={"overwrite":overwrite, "noise_percent":noise_percent})
            results.append(result)
        pool.close()
        pool.join()
        feat_dim = self.feat_dim
        feat = np.zeros((1, feat_dim))
        feat_squared = np.zeros((1, feat_dim))
        count = 0
        return_dict = {}
        for data in results:
            next_feat = data.get()
            next_feat_squared = np.square(next_feat)
            feat_vertically_stacked = np.concatenate((feat, next_feat)).reshape(-1, feat_dim)
            feat = np.sum(feat_vertically_stacked, axis=0, keepdims=True)
            feat_squared_vertically_stacked = np.concatenate(
                (feat_squared, next_feat_squared)).reshape(-1, feat_dim)
            feat_squared = np.sum(feat_squared_vertically_stacked, axis=0, keepdims=True)
            count += float(next_feat.shape[0])
        return_dict[1] = {'feat': feat, 'feat_squared': feat_squared, 'count': count}

        # return_dict = {}
        # with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        #     feat_dim = self.feat_dim
        #     feat = np.zeros((1, feat_dim))
        #     feat_squared = np.zeros((1, feat_dim))
        #     count = 0
        #     for f, data in zip(audio_paths, executor.map(spectrogram_from_file, audio_paths, overwrite=overwrite, noise_percent=noise_percent)):
        #         try:
        #             next_feat = data
        #             next_feat_squared = np.square(next_feat)
        #             feat_vertically_stacked = np.concatenate((feat, next_feat)).reshape(-1, feat_dim)
        #             feat = np.sum(feat_vertically_stacked, axis=0, keepdims=True)
        #             feat_squared_vertically_stacked = np.concatenate(
        #                 (feat_squared, next_feat_squared)).reshape(-1, feat_dim)
        #             feat_squared = np.sum(feat_squared_vertically_stacked, axis=0, keepdims=True)
        #             count += float(next_feat.shape[0])
        #         except Exception as exc:
        #             log.info('%r generated an exception: %s' % (f, exc))
        #     return_dict[1] = {'feat': feat, 'feat_squared': feat_squared, 'count': count}

        feat = np.sum(np.vstack([item['feat'] for item in return_dict.values()]), axis=0)
        count = sum([item['count'] for item in return_dict.values()])
        feat_squared = np.sum(np.vstack([item['feat_squared'] for item in return_dict.values()]), axis=0)

        self.feats_mean = feat / float(count)
        self.feats_std = np.sqrt(feat_squared / float(count) - np.square(self.feats_mean))
        np.savetxt(
            generate_file_path(self.save_dir, self.model_name, 'feats_mean'), self.feats_mean)
        np.savetxt(
            generate_file_path(self.save_dir, self.model_name, 'feats_std'), self.feats_std)
        log.info("End calculating mean and std from samples")


def featurize1(datagen, file=None, overwrite=False, noise_percent=0.4):
    return datagen.featurize(file, overwrite=overwrite, noise_percent=noise_percent)


if __name__ == "__main__":
    log = LogUtil().getlogger()
    # with open("/Users/lonica/Downloads/resulttxt_1.json", 'rt', encoding='UTF-8') as json_line_file:
    #     for line_num, json_line in enumerate(json_line_file):
    #         try:
    #             spec = json.loads(json_line)
    #         except Exception as e:
    #             print(json_line)
    #             log.warn('Error reading line #{}: {}'.format(line_num, json_line))
    #             log.warn(str(e))

    # def deal(threadIndex, path, return_dict):
    #     return_dict[threadIndex] = {"path": path}
    # def deal_file(f):
    #     if random.random() < 0.5:
    #         time.sleep(0.01)
    #     return {"f": f}
    #
    # audio_paths = range(100)
    #
    # manager = Manager()
    # return_dict = manager.dict()
    # jobs = []
    # for threadIndex in range(cpu_count()):
    #     proc = Process(target=deal,
    #                    args=(threadIndex, audio_paths, return_dict))
    #     jobs.append(proc)
    #     proc.start()
    # for proc in jobs:
    #     proc.join()
    # for v in return_dict.values():
    #     print(v)
    # result = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count() - 2) as executor:
    #     future_to_f = {executor.submit(deal_file, f): f for f in audio_paths}
    #     for future in concurrent.futures.as_completed(future_to_f):
    #         f = future_to_f[future]
    #         try:
    #             data = future.result()
    #             result.append(data)
    #         except Exception as exc:
    #             print('%r generated an exception: %s' % (f, exc))
    #         else:
    #             print('%r file ' % f)
    # # for r in result:
    # #     print(r)
    # print(result)
    datagen = DataGenerator(save_dir="checkpoints", model_name="deep_bucket_4", max_freq=8000)
    datagen.load_train_data("./resources/train.json", max_duration=16)
    st1 = time.time()
    datagen.sample_normalize(k_samples=-1, noise_percent=0)
    log.info("time %s", time.time() - st1)
    # datagen.featurize("/Users/lonica/Downloads/output_1.wav", overwrite=True, save_feature_as_csvfile=True)
    # datagen.featurize("/Users/lonica/Downloads/103-1240-0000.wav", overwrite=True, save_feature_as_csvfile=True)
    # datagen.featurize("/Users/lonica/Downloads/5390-30102-0021.wav", overwrite=True, save_feature_as_csvfile=True)
    # datagen.featurize("/Users/lonica/Downloads/AISHELL-ASR0009-OS1_sample/SPEECH_DATA/S0150/S0150_mic/BAC009S0150W0498.wav", overwrite=True, save_feature_as_csvfile=True)

from __future__ import print_function
import mxnet as mx
import sys
sys.path.insert(0, "../../python")

import bisect
import random
import numpy as np

BATCH_SIZE = 1
SEQ_LENGTH = 0
NUM_GPU = 1


def get_label(buf, num_lable):
    ret = np.zeros(num_lable)
    for i in range(len(buf)):
        ret[i] = int(buf[i])
    return ret


class BucketSTTIter(mx.io.DataIter):
    def __init__(self, count, datagen, batch_size, num_label, init_states, seq_length, width, height,
                 sort_by_duration=True,
                 is_bi_graphemes=False,
                 partition="train",
                 buckets=[]
                 ):
        super(BucketSTTIter, self).__init__()

        self.maxLabelLength = num_label
        # global param
        self.batch_size = batch_size
        self.count = count
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (self.batch_size, seq_length, width * height))] + init_states
        self.provide_label = [('label', (self.batch_size, self.maxLabelLength))]
        self.datagen = datagen
        self.label = None
        self.is_bi_graphemes = is_bi_graphemes
        # self.partition = datagen.partition
        if partition == 'train':
            durations = datagen.train_durations
            audio_paths = datagen.train_audio_paths
            texts = datagen.train_texts
        elif partition == 'validation':
            durations = datagen.val_durations
            audio_paths = datagen.val_audio_paths
            texts = datagen.val_texts
        elif partition == 'test':
            durations = datagen.test_durations
            audio_paths = datagen.test_audio_paths
            texts = datagen.test_texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")
        # if sortagrad
        if sort_by_duration:
            durations, audio_paths, texts = datagen.sort_by_duration(durations,
                                                                     audio_paths,
                                                                     texts)
        else:
            durations = durations
            audio_paths = audio_paths
            texts = texts
        self.trainDataList = zip(durations, audio_paths, texts)
        # to shuffle data
        if not sort_by_duration:
            random.shuffle(self.trainDataList)

        self.trainDataIter = iter(self.trainDataList)
        self.is_first_epoch = True

        data_lengths = [datagen.featurize(a).shape[0] for a in audio_paths]

        if len(buckets) == 0:
            buckets = [i for i, j in enumerate(np.bincount(data_lengths))
                       if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i, sent in enumerate(data_lengths):
            buck = bisect.bisect_left(buckets, sent)
            if buck == len(buckets):
                ndiscard += 1
                continue
            self.data[buck].append(self.trainDataList[i])

        print("WARNING: discarded %d sentences longer than the largest bucket."% ndiscard)

        self.buckets = buckets
        self.nddata = []
        self.ndlabel = []
        self.default_bucket_key = max(buckets)

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        #self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)
        self.is_first_epoch = False

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        audio_paths = []
        texts = []
        for duration, audio_path, text in self.data[i][j:j+self.batch_size]:
            audio_paths.append(audio_path)
            texts.append(text)

        if self.is_first_epoch:
            data_set = self.datagen.prepare_minibatch(audio_paths, texts, overwrite=True,
                                                      is_bi_graphemes=self.is_bi_graphemes)
        else:
            data_set = self.datagen.prepare_minibatch(audio_paths, texts, overwrite=False,
                                                      is_bi_graphemes=self.is_bi_graphemes)

        data_all = [mx.nd.array(data_set['x'])] + self.init_state_arrays
        label_all = [mx.nd.array(data_set['y'])]

        self.label = label_all
        provide_data = [('data', (self.batch_size, self.buckets[i], self.param.width * self.param.height))] + self.init_states

        return mx.io.DataBatch(data_all, label_all, pad=0,
                               bucket_key=self.buckets[i],
                               provide_data=provide_data,
                               provide_label=self.provide_label)
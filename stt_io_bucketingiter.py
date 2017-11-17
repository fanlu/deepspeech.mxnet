from __future__ import print_function
import mxnet as mx
import sys

from log_util import LogUtil

sys.path.insert(0, "../../python")

import bisect
import socket
import numpy as np
import threading

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
                 language="zh",
                 zh_type="zi",
                 partition="train",
                 buckets=[],
                 save_feature_as_csvfile=False,
                 num_parts=1,
                 part_index=0,
                 noise_percent=0.4,
                 fbank=False
                 ):
        super(BucketSTTIter, self).__init__()

        self.maxLabelLength = num_label
        # global param
        self.batch_size = batch_size
        self.count = count
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.width = width
        self.height = height
        self.datagen = datagen
        self.label = None
        self.is_bi_graphemes = is_bi_graphemes
        self.language = language
        self.zh_type = zh_type
        self.num_parts = num_parts
        self.part_index = part_index
        self.noise_percent = noise_percent
        self.fbank = fbank
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
        log = LogUtil().getlogger()
        # if sortagrad
        if sort_by_duration:
            durations, audio_paths, texts = datagen.sort_by_duration(durations,
                                                                     audio_paths,
                                                                     texts)
        else:
            durations = durations
            audio_paths = audio_paths
            texts = texts
        self.trainDataList = list(zip(durations, audio_paths, texts))

        # self.trainDataList = [d for index, d in enumerate(zip(durations, audio_paths, texts)) if index % self.num_parts == self.part_index]
        # log.info("partition: %s, num_works: %d, part_index: %d 's data size is %d of all size is %d" %
        #          (partition, self.num_parts, self.part_index, len(self.trainDataList), len(durations)))
        self.trainDataIter = iter(self.trainDataList)
        self.is_first_epoch = True

        data_lengths = [int(d * 100) for d in durations]
        if len(buckets) == 0:
            buckets = [i for i, j in enumerate(np.bincount(data_lengths))
                       if j >= batch_size]
        if len(buckets) == 0:
            raise Exception(
                'There is no valid buckets. It may occured by large batch_size for each buckets. max bincount:%d batch_size:%d' % (
                    max(np.bincount(data_lengths)), batch_size))
        buckets.sort()
        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i, sent in enumerate(data_lengths):
            buck = bisect.bisect_left(buckets, sent)
            if buck == len(buckets):
                ndiscard += 1
                continue
            self.data[buck].append(self.trainDataList[i])
        if ndiscard != 0:
            print("WARNING: discarded %d sentences longer than the largest bucket." % ndiscard)
        # self.num_parts = 3 debug
        # self.part_index = 2
        for index_buck, buck in enumerate(self.data):
            self.data[index_buck] = [d for index_d, d in enumerate(
                self.data[index_buck][:len(self.data[index_buck]) // self.num_parts * self.num_parts]) if
                                     index_d % self.num_parts == self.part_index]
            log.info("partition: %s, num_works: %d, part_index: %d %d's data size is %d " %
                     (partition, self.num_parts, self.part_index, index_buck, len(self.data[index_buck])))
        self.buckets = buckets
        self.nddata = []
        self.ndlabel = []
        self.default_bucket_key = max(buckets)

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        if not self.fbank:
            self.provide_data = [('data', (self.batch_size, self.default_bucket_key, width * height))] + init_states
        else:
            self.provide_data = [('data', (self.batch_size, 3, self.default_bucket_key, 41))] + init_states
        self.provide_label = [('label', (self.batch_size, self.maxLabelLength))]
        self.save_feature_as_csvfile = save_feature_as_csvfile

        # self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        # np.random.seed(self.n_epochs)
        np.random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        audio_paths = []
        texts = []
        durations = []
        for duration, audio_path, text in self.data[i][j:j + self.batch_size]:
            audio_paths.append(audio_path)
            durations.append(duration)
            texts.append(text)
        log = LogUtil().getlogger()
        # log.info("%s, %s, %s" % (socket.gethostname(), audio_paths, durations))

        if self.is_first_epoch:
            if not self.fbank:
                data_set = self.datagen.prepare_minibatch(audio_paths, texts, overwrite=True,
                                                          is_bi_graphemes=self.is_bi_graphemes,
                                                          seq_length=self.buckets[i],
                                                          save_feature_as_csvfile=self.save_feature_as_csvfile,
                                                          language=self.language,
                                                          zh_type=self.zh_type,
                                                          noise_percent=self.noise_percent)
            else:
                data_set = self.datagen.prepare_minibatch_fbank(audio_paths, texts, overwrite=True,
                                                                is_bi_graphemes=self.is_bi_graphemes,
                                                                seq_length=self.buckets[i],
                                                                save_feature_as_csvfile=self.save_feature_as_csvfile,
                                                                language=self.language,
                                                                zh_type=self.zh_type,
                                                                noise_percent=self.noise_percent)
        else:
            if not self.fbank:
                data_set = self.datagen.prepare_minibatch(audio_paths, texts, overwrite=False,
                                                          is_bi_graphemes=self.is_bi_graphemes,
                                                          seq_length=self.buckets[i],
                                                          save_feature_as_csvfile=self.save_feature_as_csvfile,
                                                          language=self.language,
                                                          zh_type=self.zh_type,
                                                          noise_percent=self.noise_percent)
            else:
                data_set = self.datagen.prepare_minibatch_fbank(audio_paths, texts, overwrite=False,
                                                                is_bi_graphemes=self.is_bi_graphemes,
                                                                seq_length=self.buckets[i],
                                                                save_feature_as_csvfile=self.save_feature_as_csvfile,
                                                                language=self.language,
                                                                zh_type=self.zh_type,
                                                                noise_percent=self.noise_percent)

        data_all = [mx.nd.array(data_set['x'])] + self.init_state_arrays
        label_all = [mx.nd.array(data_set['y'])]

        self.label = label_all
        if not self.fbank:
            provide_data = [('data', (self.batch_size, self.buckets[i], self.width * self.height))] + self.init_states
        else:
            provide_data = [('data', (self.batch_size, 3, self.buckets[i], 41))] + self.init_states

        return mx.io.DataBatch(data_all, label_all, pad=0,
                               index=audio_paths,
                               bucket_key=self.buckets[i],
                               provide_data=provide_data,
                               provide_label=self.provide_label)


class BucketPrefetchingIter(mx.io.DataIter):
    """Performs pre-fetch for other data iterators.

    This iterator will create another thread to perform ``iter_next`` and then
    store the data in memory. It potentially accelerates the data read, at the
    cost of more memory usage.

    Parameters
    ----------
    iters : DataIter or list of DataIter
        The data iterators to be pre-fetched.
    rename_data : None or list of dict
        The *i*-th element is a renaming map for the *i*-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data.
    rename_label : None or list of dict
        Similar to ``rename_data``.

    Examples
    --------
    >>> iter1 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> iter2 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> piter = mx.io.PrefetchingIter([iter1, iter2],
    ...                               rename_data=[{'data': 'data_1'}, {'data': 'data_2'}])
    >>> print(piter.provide_data)
    [DataDesc[data_1,(25, 10L),<type 'numpy.float32'>,NCHW],
     DataDesc[data_2,(25, 10L),<type 'numpy.float32'>,NCHW]]
    """

    def __init__(self, iters, rename_data=None, rename_label=None):
        super(BucketPrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter > 0
        self.iters = iters
        self.rename_data = rename_data
        self.rename_label = rename_label
        self.batch_size = self.provide_data[0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for i in self.data_taken:
            i.set()
        self.started = True
        self.current_batch = [None for i in range(self.n_iter)]
        self.next_batch = [None for i in range(self.n_iter)]

        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()

        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for i in self.data_taken:
            i.set()
        for thread in self.prefetch_threads:
            thread.join()

    @property
    def provide_data(self):
        if self.rename_data is None:
            return sum([i.provide_data for i in self.iters], [])
        else:
            return sum([[
                mx.io.DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, mx.io.DataDesc) else mx.io.DataDesc(*x)
                for x in i.provide_data
            ] for r, i in zip(self.rename_data, self.iters)], [])

    @property
    def provide_label(self):
        if self.rename_label is None:
            return sum([i.provide_label for i in self.iters], [])
        else:
            return sum([[
                mx.io.DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, mx.io.DataDesc) else mx.io.DataDesc(*x)
                for x in i.provide_label
            ] for r, i in zip(self.rename_label, self.iters)], [])

    def reset(self):
        for i in self.data_ready:
            i.wait()
        for i in self.iters:
            i.reset()
        for i in self.data_ready:
            i.clear()
        for i in self.data_taken:
            i.set()

    def iter_next(self):
        for i in self.data_ready:
            i.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            for batch in self.next_batch:
                assert batch.pad == self.next_batch[0].pad, \
                    "Number of entry mismatches between iterators"
            self.current_batch = mx.io.DataBatch(sum([batch.data for batch in self.next_batch], []),
                                                 sum([batch.label for batch in self.next_batch], []),
                                                 self.next_batch[0].pad,
                                                 self.next_batch[0].index,
                                                 bucket_key=self.next_batch[0].bucket_key,
                                                 provide_data=self.next_batch[0].provide_data,
                                                 provide_label=self.next_batch[0].provide_label)
            for i in self.data_ready:
                i.clear()
            for i in self.data_taken:
                i.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

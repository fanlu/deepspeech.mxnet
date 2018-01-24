from __future__ import print_function

import mxnet as mx
import glob
import argparse
import os
import time

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    args = parser.parse_args()
    return args


def audio_encode(args, i, item, q_out):
    pass


def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        audio_encode(args, i, item, q_out)


def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1


if __name__ == "__main__":
    args = parse_args()
    files = glob.glob("/Users/lonica/Downloads/")
    # q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
    # q_out = multiprocessing.Queue(1024)
    # read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
    #                 for i in range(args.num_thread)]
    # for p in read_process:
    #     p.start()
    # write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
    # write_process.start()
    #
    # for i, item in enumerate(files):
    #     q_in[i % len(q_in)].put((i, item))
    # for q in q_in:
    #     q.put(None)
    # for p in read_process:
    #     p.join()
    #
    # q_out.put(None)
    # write_process.join()

# coding=utf-8
import glob

from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import sys
from multiprocessing import cpu_count, Process, Manager
import time
import os

max_duration = 16


def deal_file(file_name, min_silence=500, silence_thresh=-50, result=[]):
    sound = AudioSegment.from_wav(file_name)
    info = "%s start pid: %s deal: %s" % (str(time.time()), str(os.getpid()), file_name)
    print(info)
    deal_chunk(sound, min_silence, silence_thresh, result)
    return result


def deal_chunk(sound, min_silence, silence_thresh, result):
    # AudioSegment.from_wav
    chunks = split_on_silence(sound, min_silence_len=min_silence,
                              silence_thresh=silence_thresh)  # silence time:700ms and silence_dBFS<-70dBFS
    for chunk in chunks:
        if chunk.duration_seconds > 10:
            if min_silence > 100:
                print(min_silence)
                result = deal_chunk(chunk, min_silence - 100, silence_thresh, result)
            else:
                print(silence_thresh)
                result = deal_chunk(chunk, min_silence, silence_thresh + 10, result)
        else:
            result.append(chunk.set_channels(1).set_frame_rate(16000))
    return result


if __name__ == "__main__":

    # file_name = "/Users/lonica/Downloads/7ebec23e-0d20-4e3d-afca-de325f7c2239.wav"

    import concurrent.futures

    files = glob.glob("/Users/lonica/Downloads/7*be*.wav")

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count() - 2) as executor:
        for f, data in zip(files, executor.map(deal_file, files, min_silence=500, silence_thresh=-50, result=[])):
            print('%r file return %d chunks' % (f, len(data)))
            result = []
            for d in data:
                # 如果result最后一个chunk时间小于阈值
                if len(result) > 0 and result[len(result) - 1].duration_seconds < 2:
                    # 合并到result最后一条
                    if d.duration_seconds + result[len(result) - 1].duration_seconds < max_duration:
                        result[len(result) - 1] += d
                    else:
                        result.append(d)
                else:
                    result.append(d)
            path, name = f.rsplit("/", 1)
            for i, d in enumerate(result):
                output_file = path + "/zt/500/" + name[:-4] + "_%03d.wav" % i
                print(output_file, d.duration_seconds)
                d.export(output_file, format="wav")

                # words = chunks[2:] #first and second are not words.

                # len1 = len(words)
                #
                # new = AudioSegment.empty()
                # silence = AudioSegment.silent(duration=1000)#1000ms silence
                #
                #
                # order = range(len1)
                # random.shuffle(order)
                # print order
                # comments = ""
                #
                # for i in order:
                #     new += words[i]+silence
                #     comments += str(i)+","
                #
                # save_name = file_name.split(".")[0]+"-random{0}.".format(random.randrange(0,9))+file_name.split(".")[1]
                # new.export(save_name, format="mp3",tags={'artist': 'AppLeU0', 'album': file_name, 'comments': comments[:-1]})

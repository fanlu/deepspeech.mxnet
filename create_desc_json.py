# encoding=utf-8
"""
Use this script to create JSON-Line description files that can be used to
train deep-speech models through this library.
This works with data directories that are organized like LibriSpeech:
data_directory/group/speaker/[file_id1.wav, file_id2.wav, ...,
                              speaker.trans.txt]

Where speaker.trans.txt has in each line, file_id transcription
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import wave
import glob
from collections import defaultdict
import random
import pypinyin
from stt_phone_util import generate_zi_label, strQ2B
from sentence2phoneme import sentence2phoneme, loadmap
import thulac

thu1 = thulac.thulac(seg_only=True)
_data_path = "/Users/lonica/Downloads/"
# _data_path = "/export/fanlu/aishell/"

word_2_lexicon = defaultdict(list)


def read_lexicon():
    lines = open(_data_path + "resource_aishell/lexicon.txt").readlines()
    for line in lines:
        r = line.strip().split(" ")
        word_2_lexicon[r[0]].append(" ".join(r[1:]))


def tvt():
    "S0002-S0723"
    "S0724-S0763"
    "S0764-S0916"


def ai_2_phone():
    lines = open(_data_path + "data_aishell/transcript/aishell_transcript_v0.8.txt").readlines()
    out_file = open("resources/aishell_train.json", 'w')
    out_file1 = open("resources/aishell_validation.json", 'w')
    out_file2 = open("resources/aishell_test.json", 'w')
    for line in lines:
        rs = line.strip().split(" ")
        ps = []
        for r in rs[1:]:
            if r:
                phone = word_2_lexicon.get(r)
                if phone:
                    ps.append(random.choice(phone))
                else:
                    print("error word:%s" % r)
        if rs[0][6:11] <= "S0723":
            wav = _data_path + "data_aishell/wav/train/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            audio = wave.open(wav)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file.write(line + "\n")
        elif rs[0][6:11] <= "S0763":
            wav = _data_path + "data_aishell/wav/dev/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            audio = wave.open(wav)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file1.write(line + "\n")
        else:
            wav = _data_path + "data_aishell/wav/test/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            audio = wave.open(wav)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file2.write(line + "\n")
    out_file.close()
    out_file1.close()
    out_file2.close()


def ai_thchs30_2_word():
    ori_wavs = glob.glob("/export/fanlu/thchs30/data_thchs30/data/*.wav")
    out_file = open("resources/thchs30_data.json", 'w')
    for w in ori_wavs:
        path, name = w.rsplit("/", 1)
        rs = open(w + ".trn").readlines()[0].strip()
        ps = generate_zi_label(rs)
        audio = wave.open(w)
        duration = float(audio.getnframes()) / audio.getframerate()
        audio.close()
        line = "{\"key\":\"" + w + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
        out_file.write(line + "\n")
        # for w2 in glob.glob(path.replace("data","data_aug")+"/" + name.split(".")[0] + "*.wav"):
        #  audio = wave.open(w2)
        #  duration = float(audio.getnframes()) / audio.getframerate()
        #  audio.close()
        #  line = "{\"key\":\"" + w2 + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
        #  out_file.write(line + "\n")
        # print(ps)
    out_file.close()


def search_2_word():
    tran = open("/export/aiplatform/search/transcript").readlines()
    out_file = open('resources/search.json', 'w')
    for t in tran:
        path, d, txt = t.split(" ", 2)
        ps = generate_zi_label(txt.strip())
        audio_path = "/export/aiplatform/search/" + "wav/" + path
        audio = wave.open(audio_path)
        duration = float(audio.getnframes()) / audio.getframerate()
        audio.close()
        line = "{\"key\":\"" + audio_path + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(
            ps) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


def ai_2_word():
    lines = open(_data_path + "data_aishell/transcript/aishell_transcript_v0.8.txt").readlines()
    out_file = open("resources/aishell_train_noise.json", 'w')
    out_file1 = open("resources/aishell_validation_noise.json", 'w')
    out_file2 = open("resources/aishell_test_noise.json", 'w')
    for line in lines:
        rs = line.strip().split(" ")
        ps = generate_zi_label("".join(rs[1:]))
        if rs[0][6:11] <= "S0723":
            wav = _data_path + "data_aishell/wav/train/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            dir = _data_path + "data_aishell/wav/train_aug/" + rs[0][6:11] + "/" + rs[0]
            for w in glob.glob(dir + "*.wav"):
                audio = wave.open(w)
                duration = float(audio.getnframes()) / audio.getframerate()
                audio.close()
                line = "{\"key\":\"" + w + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
                out_file.write(line + "\n")
        elif rs[0][6:11] <= "S0763":
            wav = _data_path + "data_aishell/wav/dev/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            audio = wave.open(wav)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file1.write(line + "\n")
        else:
            wav = _data_path + "data_aishell/wav/test/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            audio = wave.open(wav)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file2.write(line + "\n")
    out_file.close()
    out_file1.close()
    out_file2.close()


def ai_2_word_single(wav):
    out_file = open("resources/d.json", 'w')
    audio = wave.open(wav)
    duration = float(audio.getnframes()) / audio.getframerate()
    audio.close()
    line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"1 1\"}"
    out_file.write(line + "\n")


pinyin_2_phone_map = {}
phone_2_pinyin_map = {}


def pinyin_2_phone():
    a = open(_data_path + "table.txt")
    b = a.readlines()
    for line in b:
        line = line.replace("\n", "")
        part1, part2 = line.split("->")
        part3, part4 = part2.split("+")
        phone_2_pinyin_map[(part3, part4)] = part1
        pinyin_2_phone_map[part1] = (part3, part4)
    lines = open(_data_path + "special.txt")
    for line in lines:
        line = line.replace("\n", "")
        part1, part2 = line.split("->")
        part3, part4 = part2.split("+")
        phone_2_pinyin_map[(part3, part4)] = part1
        pinyin_2_phone_map[part1] = (part3, part4)


def word_2_pinyin(file_path):
    lines = open(file_path).readlines()
    with open("resources/" + file_path.rsplit("/")[1].split(".")[0] + "_py.json", 'w') as out_file:
        for line in lines:
            d = json.loads(line)
            word = d.get("text")
            text1 = "".join([i for i in word if i != " "])
            text2 = thu1.cut(text1.encode('utf-8'), text=True)
            # pypinyin.pinyin(word, style=pypinyin.STYLE_INITIALS)
            # pypinyin.pinyin(word, style=pypinyin.FINALS_TONE3)
            py = pypinyin.pinyin(text2.decode('utf-8'), style=pypinyin.TONE3)
            pys = " ".join([i[0] for i in py if i[0] != " "])
            # print(pys)
            out = "{\"key\":\"" + d.get("key") + "\", \"duration\": " + str(
                d.get("duration")) + ", \"text\":\"" + pys + "\"}"
            try:
                out_file.write(out + "\n")
            except:
                print(out)
                continue


def main(data_directory, output_file):
    labels = []
    durations = []
    keys = []
    for group in os.listdir(data_directory):
        speaker_path = os.path.join(data_directory, group)
        for speaker in os.listdir(speaker_path):
            labels_file = os.path.join(speaker_path, speaker,
                                       '{}-{}.trans.txt'
                                       .format(group, speaker))
            for line in open(labels_file):
                split = line.strip().split()
                file_id = split[0]
                label = ' '.join(split[1:]).lower()
                audio_file = os.path.join(speaker_path, speaker,
                                          file_id) + '.wav'
                audio = wave.open(audio_file)
                duration = float(audio.getnframes()) / audio.getframerate()
                audio.close()
                keys.append(audio_file)
                durations.append(duration)
                labels.append(label)
    with open(output_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                               'text': labels[i]})
            out_file.write(line + '\n')


def aishell(data_directory, output_file):
    with open(output_file, 'w') as out_file:
        for group in os.listdir(data_directory):
            speaker_path = os.path.join(data_directory, group)
            if os.path.isdir(speaker_path):
                for speaker in os.listdir(speaker_path):
                    mic_path = os.path.join(speaker_path, speaker)
                    if os.path.isdir(mic_path):
                        for wav in glob.glob(mic_path + "/*.wav"):
                            audio = wave.open(wav)
                            duration = float(audio.getnframes()) / audio.getframerate()
                            audio.close()

                            text = open(wav.replace("wav", "txt")).readlines()[0].strip()

                            line = json.dumps({'key': wav, 'duration': duration,
                                               'text': text})
                            # line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + text + "\"}"
                            out_file.write(line + '\n')


py_2_phone_map = {}


def py_2_phone():
    ts = open("resources/table.txt").readlines()
    ss = open("resources/special.txt").readlines()
    for t in ts:
        py, smym = t.split('->')
        py_2_phone_map[py] = smym.strip().split('+')
    for s in ss:
        py, smym = s.split('->')
        py_2_phone_map[py] = smym.strip().split('+')


def zi_2_phone():
    pinyin2phoneme = loadmap("resources/table.txt")
    ls = open("resources/aishell_thchs30_noise.json").readlines()
    out_file = open("resources/aishell_thchs30_noise_phone.json", "w")
    for l in ls:
        d = json.loads(l)
        text = d.get("text").encode('utf-8')
        phone = sentence2phoneme(text, pinyin2phoneme)
        line = "{\"key\":\"" + d.get("key") + "\", \"duration\": " + str(
            d.get("duration")) + ", \"text\":\"" + phone + "\"}"
        out_file.write(line + "\n")
    out_file.close()


py_2_phone_map = {}


def py_2_phone():
    ts = open("/Users/lonica/Downloads/resource_aishell/table.txt").readlines()
    ss = open("/Users/lonica/Downloads/resource_aishell/special.txt").readlines()
    for t in ts:
        py, smym = t.split('->')
        py_2_phone_map[py] = smym.strip().split('+')
    for s in ss:
        py, smym = s.split('->')
        py_2_phone_map[py] = smym.strip().split('+')


def zi_2_phone():
    ls = open("train.json").readlines()
    out_file = open("resources/aishell_thchs30_noise_phone.json", "w")
    for l in ls:
        d = json.loads(l)
        text = d.get("text")
        str_ = text.strip().replace(" ", "")
        zis = []
        for ch in str_:
            if ch != u' ':
                phones = word_2_lexicon.get(ch.encode('utf-8'))
                if len(phones) > 1:
                    zis += phones
        pyl = pypinyin.pinyin(str_, style=pypinyin.TONE3)
        phone = []
        for p in pyl:
            # print(p[0])
            py = p[0][:-1]
            sd = p[0][-1]
            try:
                sd = int(sd)
            except:
                py = p[0]
                sd = "5"
            print(py)
            sm, ym = py_2_phone_map.get(py)
            phone.append(sm)
            phone.append(ym + str(sd))
        line = "{\"key\":\"" + d.get("key") + "\", \"duration\": " + str(
            d.get("duration")) + ", \"text\":\"" + " ".join(
            phone) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


def split_file_2_multi(input, num):
    path, name = input.rsplit("/")
    name_pre, name_post = name.split(".")
    for i, line in enumerate(open(input).readlines()):
        with open('%s/%s_%d.json' % (path, name_pre, (i % num)), 'a+') as tmp:
            tmp.write(line)


def client_2_word():
    DIR = "/export/aiplatform/client_files4/"

    def compare(x, y):
        stat_x = os.stat(DIR + "/" + x)
        stat_y = os.stat(DIR + "/" + y)
        if stat_x.st_ctime < stat_y.st_ctime:
            return -1
        elif stat_x.st_ctime > stat_y.st_ctime:
            return 1
        else:
            return 0

    # iterms = os.listdir(DIR)

    # iterms.sort(compare)

    # for iterm in iterms:
    #    print(iterm)
    wavs = open(DIR + 'wav.txt').readlines()
    labels = open(DIR + 'label.txt').readlines()
    out_file = open(DIR + 'client4.json', 'w')
    for i, (path, txt) in enumerate(zip(wavs, labels)):
        ps = generate_zi_label(txt.replace(",", "").replace("。", "").replace("，", "").strip())
        audio_path = DIR + path.strip()
        audio = wave.open(audio_path)
        duration = float(audio.getnframes()) / audio.getframerate()
        audio.close()
        line = "{\"key\":\"" + audio_path + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(
            ps) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


def xiaoshuo_2_word():
    d = set()
    for i, line in enumerate(open("resources/unicodemap_zi.csv").readlines()):
        d.add(line.rsplit(",", 1)[0])
    special_2_normal = {'\xe3\x80\x80': "", '\xef\xb9\x92': "", '\xe3\x80\x8d': "", '\xe3\x80\x8c': "",
                        '\xee\x80\x84': "", "'": "", '\xc2\xa0': "", '\xe3\x80\x8e': "",
                        "\"": "", '\xef\xbb\xbf': "", ",": "", "。": "", "，": "", "\\": "", ")": "", '\xe3\x80\x8f': "", '\xe2\x80\x95': '',
                        '\xee\x97\xa5': '', '\xef\xbf\xbd': '', '\xef\xbc\x8e': '',
                        '|': '', '\xe2\x94\x80': '', "s\xc3\xa8": "色", "r\xc3\xac": "日", 'r\xc7\x94': "乳",
                        '\xe5\xa6\xb3': "你", 'x\xc3\xacng': "性", 'j\xc4\xabng': "精", 'ch\xc5\xabn': "春",
                        'sh\xc3\xa8': "射", 'y\xc3\xb9': "欲", 'y\xc4\xabn': "阴", 'm\xc3\xa9n': "门",
                        '\xe3\x80\x87': '零', '\xe9\x99\xbd': '阳', '\xe6\xa7\x8d': '枪', '\xe9\x99\xb0': '阴',
                        '\xe9\xa8\xb7': '骚', '\xe4\xba\xa3': "", '\xe4\xb8\xb5': "", '\xe5\xa9\xac': '淫', '\xe4\xbe\x86': '来',
			'\xe6\xb2\x92': '没', '\xe2\x80\xa2': "", '\xe2\x95\x94': "", '\xe2\x95\x95': "", '\xe2\x95\xa0': "",
			'\xe4\xba\x8a': '事', '\xe6\x95\x8e': '教', '\xe5\xb2\x80': '出', '\xe2\x95\x97': '',
                        }
    DIR = "/export/aiplatform/"
    out_file = open(DIR + 'resulttxtnew26.json', 'w')
    for i in glob.glob(DIR + "resulttxtnew26/*/*.wav"):
        txt = "".join([line.strip() for line in open(i[:-3] + "txt").readlines()])
        txt = strQ2B(txt.strip().decode("utf8")).encode("utf8")
        for k, v in special_2_normal.items():
            txt = txt.replace(k, v)
        ps = generate_zi_label(txt)
        if len(ps) == 0:
            continue
        flag = False
        for p in ps:
            if not p in d:
                print("not in d is %s %s. %s" % (p, [p], "".join(ps)))
                flag = True
                break
        if flag:
            continue
        audio = wave.open(i)
        duration = float(audio.getnframes()) / audio.getframerate()
        audio.close()
        if duration > 16:
            continue
        line = "{\"key\":\"" + i + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_directory', type=str,
    #                     help='Path to data directory')
    # parser.add_argument('output_file', type=str,
    #                     help='Path to output file')
    # args = parser.parse_args()
    # # aishell(args.data_directory, args.output_file)
    #
    # aishell("/Users/lonica/Downloads/AISHELL-ASR0009-OS1_sample/SPEECH_DATA/", "train1.json")
    # split_file_2_multi("resources/aishell_train.json", 3)
    # read_lexicon()
    # print(len(word_2_lexicon))
    # ai_2_word()

    # ai_thchs30_2_word()

    # search_2_word()

    #client_2_word()

    xiaoshuo_2_word()

    # py_2_phone()
    # zi_2_phone()
    # word_2_pinyin('resources/aishell_validation.json')

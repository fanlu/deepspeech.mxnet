# -- coding: utf-8 --
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
import string
from collections import defaultdict
import soundfile as sf
import random
import pypinyin
import time
from pydub import AudioSegment
import concurrent.futures
from multiprocessing import cpu_count
from stt_phone_util import generate_zi_label, strQ2B
from sentence2phoneme import sentence2phoneme, loadmap
from stt_metric import levenshtein_distance
import thulac

_data_path = "/Users/lonica/Downloads/"
# _data_path = "/export/fanlu/aishell/"
_data_path = "/export/aiplatform/"

word_2_lexicon = defaultdict(list)

# special_2_normal = {'\xe3\x80\x80': "", '\xef\xb9\x92': "", '\xe3\x80\x8d': "", '\xe3\x80\x8c': "",
#                     '\xee\x80\x84': "", "'": "", '\xc2\xa0': "", '\xe3\x80\x8e': "",
#                     "\"": "", '\xef\xbb\xbf': "", ",": "", "。": "", "，": "", "\\": "", ")": "", '\xe3\x80\x8f': "",
#                     '\xe2\x80\x95': '',
#                     '\xee\x97\xa5': '', '\xef\xbf\xbd': '', '\xef\xbc\x8e': '',
#                     '|': '', '\xe2\x94\x80': '', "s\xc3\xa8": "色", "r\xc3\xac": "日", 'r\xc7\x94': "乳",
#                     '\xe5\xa6\xb3': "你", 'x\xc3\xacng': "性", 'j\xc4\xabng': "精", 'ch\xc5\xabn': "春",
#                     'sh\xc3\xa8': "射", 'y\xc3\xb9': "欲", 'y\xc4\xabn': "阴", 'm\xc3\xa9n': "门",
#                     '\xe3\x80\x87': '零', '\xe9\x99\xbd': '阳', '\xe6\xa7\x8d': '枪', '\xe9\x99\xb0': '阴',
#                     '\xe9\xa8\xb7': '骚', '\xe4\xba\xa3': "", '\xe4\xb8\xb5': "", '\xe5\xa9\xac': '淫',
#                     '\xe4\xbe\x86': '来',
#                     '\xe6\xb2\x92': '没', '\xe2\x80\xa2': "", '\xe2\x95\x94': "", '\xe2\x95\x95': "",
#                     '\xe2\x95\xa0': "",
#                     '\xe4\xba\x8a': '事', '\xe6\x95\x8e': '教', '\xe5\xb2\x80': '出', '\xe2\x95\x97': '',
#                     }

special_2_normal = {'\xe3\x80\x80': "", '\xee\x80\x84': "", '\xc2\xa0': "", '\xef\xbb\xbf': "", '\xe4\xba\xa3': "",
                    '\xe4\xb8\xb5': "", '\xee\x97\xa5': '', '\xef\xbf\xbd': '',
                    "s\xc3\xa8": "色", "r\xc3\xac": "日", 'r\xc7\x94': "乳", 'x\xc3\xacng': "性", 'j\xc4\xabng': "精",
                    'ch\xc5\xabn': "春", 'sh\xc3\xa8': "射", 'y\xc3\xb9': "欲", 'y\xc4\xabn': "阴", 'm\xc3\xa9n': "门",
                    '\xe9\xa8\xb7': '骚', '\xe5\xa9\xac': '淫', '\xe9\x99\xbd': '阳', '\xe9\x99\xb0': '阴',
                    '\xe6\xa7\x8d': '枪', '\xe4\xbe\x86': '来', '\xe5\xa6\xb3': "你", '\xe3\x80\x87': '零',
                    '\xe6\xb2\x92': '没', '\xe4\xba\x8a': '事', '\xe6\x95\x8e': '教', '\xe5\xb2\x80': '出',
                    }


def deletePunc(mystr):
    mystr = mystr.translate(None, string.punctuation)
    for k in "，。？！、【】：；‘“”’（）《》…─﹒╠―•╙╖╔╗╘╕．﹒『』「」".decode("utf-8"):
        mystr = mystr.replace(k.encode("utf-8"), "")
    for k, v in special_2_normal.items():
        mystr = mystr.replace(k, v)
    # identity = string.maketrans(' ', ' ')

    return mystr


def get_duration_wave(w):
    audio = wave.open(w)
    duration = float(audio.getnframes()) / audio.getframerate()
    audio.close()
    return duration


def get_duration_sf(w):
    so = sf.SoundFile(w)
    duration = float(len(so)) / so.samplerate
    return duration


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
    out_file = open(_data_path + "data_aishell/aishell_train_8k.json", 'w')
    out_file1 = open(_data_path + "data_aishell/aishell_validation_8k.json", 'w')
    out_file2 = open(_data_path + "data_aishell/aishell_test_8k.json", 'w')
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
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file.write(line + "\n")
        elif rs[0][6:11] <= "S0763":
            wav = _data_path + "data_aishell/wav/dev/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file1.write(line + "\n")
        else:
            wav = _data_path + "data_aishell/wav/test/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file2.write(line + "\n")
    out_file.close()
    out_file1.close()
    out_file2.close()


def ai_thchs30_2_word():
    ori_wavs = glob.glob(_data_path + "/thchs30/data_thchs30/8k/data/*.wav")
    out_file = open(_data_path + "/thchs30/data_thchs30/8k/thchs30_data.json", 'w')
    for w in ori_wavs:
        path, name = w.rsplit("/", 1)
        rs = open(w.replace("8k", "") + ".trn").readlines()[0].strip()
        ps = generate_zi_label(rs)
        duration = get_duration_wave(w)
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
        duration = get_duration_wave(audio_path)
        line = "{\"key\":\"" + audio_path + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(
            ps) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


def ai_2_word():
    lines = open(_data_path + "data_aishell/transcript/aishell_transcript_v0.8.txt").readlines()
    out_file = open(_data_path + "data_aishell/wav8000/aishell_train_8k.json", 'w')
    out_file1 = open(_data_path + "data_aishell/wav8000/aishell_validation_8k.json", 'w')
    out_file2 = open(_data_path + "data_aishell/wav8000/aishell_test_8k.json", 'w')
    for line in lines:
        rs = line.strip().split(" ")
        ps = generate_zi_label("".join(rs[1:]))
        if rs[0][6:11] <= "S0723":
            wav = _data_path + "data_aishell/wav8000/train/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            # dir = _data_path + "data_aishell/wav/train_aug/" + rs[0][6:11] + "/" + rs[0]
            # for w in glob.glob(dir + "*.wav"):
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file.write(line + "\n")
        elif rs[0][6:11] <= "S0763":
            wav = _data_path + "data_aishell/wav8000/dev/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file1.write(line + "\n")
        else:
            wav = _data_path + "data_aishell/wav/test/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file2.write(line + "\n")
    out_file.close()
    out_file1.close()
    out_file2.close()


def ai_2_word_single(wav):
    out_file = open("resources/d.json", 'w')
    duration = get_duration_wave(wav)
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
    thu1 = thulac.thulac(seg_only=True)
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
                duration = get_duration_wave(audio_file)
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
                            duration = get_duration_wave(wav)

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
        duration = get_duration_wave(audio_path)
        line = "{\"key\":\"" + audio_path + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(
            ps) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


def xiaoshuo_2_word():
    d = set()
    for i, line in enumerate(open("resources/unicodemap_zi.csv").readlines()):
        d.add(line.rsplit(",", 1)[0])

    DIR = "/export/aiplatform/8k/"
    out_file = open(DIR + 'resulttxtnew26.json', 'w')
    for i in glob.glob(DIR + "resulttxtnew26/*/*.wav"):
        txt = "".join([line.strip() for line in open(i.replace("8k/", "")[:-3] + "txt").readlines()])
        txt = strQ2B(txt.strip().decode("utf8")).encode("utf8")
        ps = generate_zi_label(deletePunc(txt))
        if len(ps) == 0:
            continue
        flag = False
        for p in ps:
            if p not in d or p.isdigit():
                print("not in d is %s %s. %s" % (p, [p], "".join(ps)))
                flag = True
                break
        if flag:
            continue
        duration = get_duration_wave(i)
        if duration > 16:
            continue
        line = "{\"key\":\"" + i + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
        out_file.write(line + "\n")
    out_file.close()


def aia_2_word(DIR):
    scp = [i for i in glob.glob(DIR + "/*/*.scp") if "noise" not in i]
    dir_name = DIR.rsplit("/", 1)[1]
    out_file = open(_data_path + 'fanlu/' + dir_name + '.json', 'w')
    d = set()
    for i, line in enumerate(open("resources/unicodemap_zi.csv").readlines()):
        d.add(line.rsplit(",", 1)[0])
    for j in scp:
        for m, line in enumerate(open(j).readlines()):
	    #print(line)
            file_name, txt = line.strip().split("\t", 1)
            path = "/export/fanlu/" + '16k/' + dir_name + "/" + j.rsplit("/", 1)[1].replace(".scp", "") + "/" + file_name + ".wav"
            if not os.path.exists(path):
                print("%s not exist" % path)
                continue
            duration = get_duration_wave(path)
            if duration > 16:
                continue
            txt = strQ2B(txt.strip().decode("utf8")).encode("utf8")
            ps = generate_zi_label(deletePunc(txt))
            if len(ps) == 0:
                continue
            flag = False
            for p in ps:
                if p not in d or p.isdigit():
                    print("not in d is %s %s. %s" % (p, [p], "".join(ps)))
                    flag = True
                    break
            if flag:
                continue
            line = "{\"key\":\"" + path.replace("fanlu", "aiplatform") + "\", \"duration\": " + str(
                duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file.write(line + "\n")
    out_file.close()


def check_biaozhu():
    f = _data_path + "bdp1.txt"
    import json
    count = 0
    all = 0
    amount = 0
    import codecs
    wfobj = codecs.open(_data_path + "bdp2.txt", 'w', encoding="utf-8")
    # f2 = open(_data_path + "bdp2.txt", "w")
    for i in open(f).readlines():
        d = json.loads(i.strip())
        manual = d.get("manual", "").encode("utf-8").replace("，", "").replace("。", "").replace(",", "").replace(".", "")
        machine = d.get("machine", "").encode("utf-8").replace("，", "").replace("。", "").replace(",", "").replace(".",
                                                                                                                  "")
        if "A" not in manual and "B" not in manual and "C" not in manual and "D" not in manual and "E" not in manual:
            manuals = generate_zi_label(manual)
            machines = generate_zi_label(machine)
            l_distance = levenshtein_distance(manuals, machines)
            count += l_distance
            all += len(manuals)
            amount += 1
            wav_file = "/export/aiplatform/data_label/task0/" + d.get("name", "")
            duration = get_duration_wave(wav_file)
            if duration > 16:
                continue
            c = {"key": wav_file, "duration": str(duration), "text": " ".join([m.decode("utf-8") for m in manuals])}
            # line = "{\"key\":\"" + wav_file + "\", \"duration\": " + str(1) + ", \"text\":\"" + " ".join([m.decode("utf-8") for m in manuals]) + "\"}"
            wfobj.write(json.dumps(c, ensure_ascii=False) + "\n")
    wfobj.close()
    print("amount: %d, error: %d, all: %d, cer: %.4f" % (amount, count, all, count / float(all)))


def auto(input_pcm):
    b = AudioSegment.from_raw(input_pcm, sample_width=2, frame_rate=16000,
                              channels=1)
    dir_path, file_name = input_pcm.rsplit("/", 1)
    if not os.path.exists(dir_path.replace("fanlu", "fanlu/8k")):
	print("mkdir %s" % dir_path.replace("fanlu", "fanlu/8k"))
        os.makedirs(dir_path.replace("fanlu", "fanlu/8k"))
    if not os.path.exists(dir_path.replace("fanlu", "fanlu/16k")):
        os.makedirs(dir_path.replace("fanlu", "fanlu/16k"))
    
    b.set_frame_rate(8000).export(dir_path.replace("fanlu", "fanlu/8k") + "/" + file_name + ".wav", format="wav")
    b.set_frame_rate(16000).export(dir_path.replace("fanlu", "fanlu/16k") + "/" + file_name + ".wav", format="wav")
    return "success"

def trans(DIR):
    audio_paths = glob.glob(DIR + "/*/*.pcm")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count() - 5) as executor:
        future_to_f = {executor.submit(auto, f): f for f in audio_paths}
        for future in concurrent.futures.as_completed(future_to_f):
            f = future_to_f[future]
            try:
                data = future.result()
                if data != "success":
                    print("%s error" % data)
            except Exception as exc:
                print('%r generated an exception: %s' % (f, exc))

def deal_1():
    f = open("/export/aiplatform/fanlu/aishell_thchs30_xs_nf.json").readlines()
    f2 = open("/export/aiplatform/fanlu/aishell_thchs30_xs_nf2.json", "w")
    for i, line in enumerate(f):
        d = json.loads(line.strip())
        du = d.get("duration", 0)
        if du < 1:
            continue
	f2.write(line)
    f2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to data directory')
    # parser.add_argument('output_file', type=str,
    #                     help='Path to output file')
    args = parser.parse_args()
    # # aishell(args.data_directory, args.output_file)
    #
    # aishell("/Users/lonica/Downloads/AISHELL-ASR0009-OS1_sample/SPEECH_DATA/", "train1.json")
    # split_file_2_multi("resources/aishell_train.json", 3)
    # read_lexicon()
    # print(len(word_2_lexicon))
    # ai_2_word()

    # ai_thchs30_2_word()
    # st = time.time()
    # for i in range(10000):
    #     get_duration_sf("/Users/lonica/Downloads/wav/7ebec23e-0d20-4e3d-afca-de325f7c2239_003.wav")
    # print(time.time()-st)
    # st1 = time.time()
    # for i in range(10000):
    #     get_duration_wave("/Users/lonica/Downloads/wav/7ebec23e-0d20-4e3d-afca-de325f7c2239_003.wav")
    # print(time.time()-st1)

    #trans(args.data_dir)

    #aia_2_word(args.data_dir)
    deal_1()

    # search_2_word()

    # client_2_word()

    # xiaoshuo_2_word()

    # check_biaozhu()
    # for k, v in special_2_normal.items():
    #    print(k)
    # py_2_phone()
    # zi_2_phone()
    # word_2_pinyin('resources/aishell_validation.json')

# coding=utf-8
import codecs
import glob
import json
import random
import string
import editdistance
import numpy as np
import os
from aip import AipSpeech
from pydub import AudioSegment
# 定义常量
from stt_phone_util import generate_zi_label
from create_desc_json import deletePunc, get_duration_wave

# APP_ID = '9775068'
# API_KEY = 'DhxrbsSxPgOgBFNSVmnTTNPX'
# SECRET_KEY = '6ae71eb009010918234e934aee561d71'
APP_ID = '9982901'
API_KEY = 'f4DvxSWlZGYE1SxtVLfhvGkp'
SECRET_KEY = 'CDyrmram5k8uEpd5eDtqDotl6zaXRQKn'
# 初始化AipSpeech对象
aipSpeech = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

_data_path = '/Users/lonica/Downloads/'
#_data_path = "/export/aiplatform/"


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def wav_2_file(path):
    files = glob.glob(_data_path + '001_*.wav')
    wfobj = codecs.open(_data_path + "001_baidu.txt", 'w', encoding="utf-8")
    for i, f in enumerate(files):
        # 识别本地文件
        result = aipSpeech.asr(get_file_content(f), 'wav', 16000, {
            'lan': 'zh',
        })
        print(i, result)
        if result.get("err_no") == 0:
            print(result.get("result")[0].encode("utf-8"))
            wfobj.write(result.get("result")[0] + "\n")
        else:
            wfobj.write(str(result.get("err_no")) + "\n")
        if i % 10 == 0:
            wfobj.flush()
    wfobj.close()


def gen_label():
    f = open(_data_path + "001_baidu.txt")
    # wfobj = codecs.open(_data_path + "001_baidu_2.txt", 'w', encoding="utf-8")
    wfobj = open(_data_path + "001_baidu_2.txt", 'w')
    for i, line in enumerate(f.readlines()):
        if not line.strip():
            continue
        newi = generate_zi_label(deletePunc(line.strip()))
        wfobj.write(" ".join(newi) + "\n")
        if i % 10 == 0:
            wfobj.flush()
    wfobj.close()


def gen_wav(txt, cfg, output):
    # cfg = {
    #     'spd': 5,  # 语速，取值0-9，默认为5中语速
    #     'pit': 5,  # 音调，取值0-9，默认为5中语调
    #     'vol': 5,  # 语速，取值0-15，默认为5中语速
    #     'per': 1   # 发音人选择, 0为女声，1为男声，3为情感合成-度逍遥，4为情感合成-度丫丫，默认为普通女
    # }
    result = aipSpeech.synthesis(txt, 'zh', 1, cfg)
    if not isinstance(result, dict):
        # audio = AudioSegment(data=result)
        o_d = output.rsplit("/", 1)[0]
        if not os.path.exists(o_d):
            os.makedirs(o_d)
        with open(output, 'wb') as f:
            f.write(result)
    else:
        print("error", result.get("err_no", ""), result.get("err_msg", ""))


def search(chunk, area):
    len_chunk = len(chunk)
    for i, a in enumerate(area):
        if editdistance.eval(area[i: i + len_chunk], chunk) == 0:
            return i


def train_2_baidu():
    out_file2 = codecs.open(_data_path + "/wav/S0002.txt", 'w', encoding="utf-8")
    for i, line in enumerate(open(_data_path + "/resource_aishell/aishell_transcript_v0.8.txt").readlines()):
        # cfg = {
        #     'spd': random.randint(4, 6),  # 语速，取值0-9，默认为5中语速
        #     'pit': random.randint(4, 7),  # 音调，取值0-9，默认为5中语调
        #     'vol': random.randint(4, 7),  # 音量，取值0-15，默认为5中音量
        #     'per': random.randint(0, 3)  # 发音人选择, 0为女声，1为男声，3为情感合成-度逍遥，4为情感合成-度丫丫(不好)，默认为普通女
        # }
        file1, d = line.strip().split(" ", 1)
        if file1 >= "BAC009S0003W0121":
            continue
        cfg = {'spd': 5, 'pit': 5, 'vol': 5, 'per': 0}
        # d = json.loads(line.strip())
        path = os.path.join(_data_path, "wav", file1[6:11], "%s_%s_%s_%s_%s.%s" % (
            file1, cfg.get("spd"), cfg.get("pit"), cfg.get("vol"),
            cfg.get("per"), "mp3"))
        gen_wav(d, cfg, path)
        AudioSegment.from_mp3(path).export(path[:-3] + "wav", format="wav")
        ps = generate_zi_label(d)
        lin = "{\"key\":\"" + path[:-3] + "wav" + "\", \"duration\": " + str(
            get_duration_wave(path[:-3] + "wav")) + ", \"text\":\"" + " ".join(
            ps).decode("utf-8") + "\"}"
        out_file2.write(lin + "\n")
        if (i + 1) % 100 == 0:
            out_file2.flush()
    out_file2.close()


if __name__ == "__main__":
    # f = open(_data_path + "001_baidu_2.txt")
    # f_ori = open(_data_path + "001_2.txt")
    # count1, count2 = 0, 0
    # str1, str2 = "", ""
    # ori_lines = " ".join([line.strip() for line in f_ori.readlines()]).split(" ")
    # pre_point = 0
    # lines = f.readlines()
    # for i, line in enumerate(lines):
    #     if i >= 10 and i <= 137:
    #         count1 = len(line.strip().split(" "))
    #         str1 = line.strip().replace(" ", "")
    #         print(count1)
    #         # 修正 count1
    #         temp = " ".join(ori_lines[pre_point: pre_point + count1])
    #         # 若cer为0 暂不去修正
    #         if editdistance.eval(str1, temp) == 0:
    #             print("xs: " + temp)
    #             print("bd: " + line.strip())
    #         else:
    #             search_area = ori_lines[pre_point: pre_point + count1 + 20]
    #             head = lines[i + 1].strip().split(" ")[:2]
    #             search_point = search(head, search_area)
    #             print("search by %s: area is %s" % ("".join(head), "".join(search_area)))
    #             if search_point:
    #                 #  and np.abs(count1 - search_point) / float(count1) < 0.1
    #                 count1 = search_point
    #             else:
    #                 # tail = line.strip().split(" ")[-3:-1]
    #                 # print("search by %s: area is %s" % ("".join(tail), "".join(search_area)))
    #                 # search_point = search(tail, search_area)
    #                 # if search_point and np.abs(count1 - search_point - len(tail)) / float(count1) < 0.1:
    #                 #     count1 = search_point + 3
    #                 # else:
    #                 # 根据行尾去 count1 附近+-5 搜索 1
    #                 tail = line.strip().split(" ")[-2:]
    #                 search_point = search(tail, search_area)
    #                 print("search by %s: area is %s" % ("".join(tail), "".join(search_area)))
    #                 if search_point:
    #                     #  and np.abs(count1 - search_point - len(tail)) / float(count1) < 0.1
    #                     count1 = search_point + len(tail)
    #
    #             print("xs: " + " ".join(ori_lines[pre_point: pre_point + count1]))
    #             print("bd: " + line.strip())
    #         # print()
    #         # print("".join(lines[i+1].strip().split(" ")[:2]))
    #
    #         pre_point = pre_point + count1
    #         print("------ %03d" % i)
    # for i, line in enumerate(f_ori.readlines()):
    #     count2 += len(line.strip().split(" "))
    #     str2 += line.strip().replace(" ", "")
    # print("baidu result %d, ori result %d" % (count1, count2))
    # print(editdistance.eval(str1, str2))

    # out_file2 = codecs.open(_data_path + "data_aishell/baidu/train.json", 'w', encoding="utf-8")
    # for i, line in enumerate(open(_data_path + "data_aishell/transcript/aishell_transcript_v0.8.txt").readlines()):
    #     cfg = {
    #         'spd': random.randint(4, 6),  # 语速，取值0-9，默认为5中语速
    #         'pit': random.randint(4, 7),  # 音调，取值0-9，默认为5中语调
    #         'vol': random.randint(4, 7),  # 音量，取值0-15，默认为5中音量
    #         'per': random.randint(0, 3)  # 发音人选择, 0为女声，1为男声，3为情感合成-度逍遥，4为情感合成-度丫丫(不好)，默认为普通女
    #     }
    #     file, d = line.strip().split(" ", 1)
    #     path = os.path.join(_data_path, "data_aishell/baidu/", file[6:11], "%s_%s_%s_%s_%s.%s" % (
    #         file, cfg.get("spd"), cfg.get("pit"), cfg.get("vol"),
    #         cfg.get("per"), "mp3"))
    #     gen_wav(d, cfg, path)
    #     AudioSegment.from_mp3(path).export(path[:-3] + "wav", format="wav")
    #     ps = generate_zi_label(d)
    #     lin = "{\"key\":\"" + path + "\", \"duration\": " + str(
    #         get_duration_wave(path[:-3] + "wav")) + ", \"text\":\"" + " ".join(
    #         ps).decode("utf-8") + "\"}"
    #     out_file2.write(lin + "\n")
    #     if (i + 1) % 100 == 0:
    #         print(i)
    #         out_file2.flush()
    # out_file2.close()

    train_2_baidu()

    # 从URL获取文件识别
    # aipSpeech.asr('', 'pcm', 16000, {
    #     'url': 'http://121.40.195.233/res/16k_test.pcm',
    #     'callback': 'http://xxx.com/receive',
    # })

# -*- coding: utf-8 -*-
import csv
from collections import Counter
from collections import OrderedDict
import re
_data_path = "./resources/"


# _data_path = "/Users/lonica/Downloads/resource_aishell/"


def split_every(n, label):
    index = 0
    if index <= len(label) - 1 <= index + n - 1:
        yield label[index:len(label)]
        index = index + n
    while index + n - 1 <= len(label) - 1:
        yield label[index:index + n]
        index = index + n
        if index <= len(label) - 1 <= index + n - 1:
            yield label[index:len(label)]
            index = index + n


def generate_phone_label(label):
    label = label.split(' ')
    return label


def generate_phone_dictionary():
    with open('resources/unicodemap_phone.csv', 'w') as bigram_label:
        bigramwriter = csv.writer(bigram_label, delimiter=',')
        for i, line in enumerate(open(_data_path + 'thchs30/data_thchs30/lm_phone/lexicon.txt').readlines()):
            bigramwriter.writerow((line.split(" ")[0], i))


def generate_py_label(label):
    label = label.split(' ')
    return label


def generate_py_dictionary(label_list):
    f = OrderedDict()
    for label in label_list:
        str_ = label.strip().split(' ')
        for ch in str_:
            if ch != u' ':
                f[ch] = 1
    with open('resources/unicodemap_py.csv', 'w') as py_label:
        pywriter = csv.writer(py_label, delimiter=',')
        baidu_labels = list('\' abcdefghijklmnopqrstuvwxyz')
        for index, key in enumerate(baidu_labels):
            pywriter.writerow((key, index + 1))
        for index, key in enumerate(f.keys()):
            pywriter.writerow((key, index + len(baidu_labels) + 1))


def generate_zi_label(label):
    # from create_desc_json import english_word
    try:
        str_ = label.strip().decode('utf-8')
    except:
        str_ = label.strip()
    l = []
    for ch in str_:
        if ch != u' ':
            l.append(ch.encode('utf-8'))
    return l
    # ret = []
    # for i in re.findall(ur'(\w+)|([\u4e00-\u9fa5])', str_):
    #     if i[0]:
    #         if i[0].upper() in english_word:
    #             ret += [j for j in i[0].upper()]
    #         else:
    #             ret.append(i[0])
    #     else:
    #         ret.append(i[1].encode('utf-8'))
    #
    # return ret


def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code<0xff01 or inside_code>0xff5e:
            rstring += uchar
            continue
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            rstring += uchar
        rstring += unichr(inside_code)
    return rstring


def generate_word_dictionary(label_list):
    f = OrderedDict()
    for line in open(_data_path + '6855map.txt').readlines():
        r = line.strip().split(" ")
        f[strQ2B(r[1].decode('utf-8'))] = int(r[0])
    for label in label_list:
        try:
            str_ = label.strip().decode('utf-8')
        except:
            str_ = label.strip()
        for ch in str_:
            if ch != u' ':
                f[strQ2B(ch)] = 1
    with open('resources/unicodemap_zi.csv', 'w') as zi_label:
        for index, key in enumerate(f.keys()):
            zi_label.write("%s,%d\n" % (key.encode('utf-8'), index + 1))

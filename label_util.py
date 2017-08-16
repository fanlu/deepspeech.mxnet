# -*- coding: utf-8 -*-
import csv
from io import open
from log_util import LogUtil
from singleton import Singleton


class LabelUtil(Singleton):
    _log = None

    # dataPath
    def __init__(self):
        self._log = LogUtil().getlogger()
        self._log.debug("LabelUtil init")

    def load_unicode_set(self, unicodeFilePath):
        self.byChar = {}
        self.byIndex = {}
        self.unicodeFilePath = unicodeFilePath

        with open(unicodeFilePath, 'rt', encoding='UTF-8') as data_file:

            self.count = 0
            for i, r in enumerate(data_file):
                ch, inx = r.strip().rsplit(",", 1)
                self.byChar[ch] = int(inx)
                self.byIndex[int(inx)] = ch
                self.count += 1


    def to_unicode(self, src, index):
        # 1 byte
        code1 = int(ord(src[index + 0]))

        index += 1

        result = code1

        return result, index

    def convert_word_to_grapheme(self, label):

        result = []

        index = 0
        while index < len(label):
            (code, nextIndex) = self.to_unicode(label, index)

            result.append(label[index])

            index = nextIndex

        return result, "".join(result)

    def convert_word_to_num(self, word):
        try:
            label_list, _ = self.convert_word_to_grapheme(word)

            label_num = []

            for char in label_list:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def convert_bi_graphemes_to_num(self, word):
            label_num = []

            for char in word:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char.decode("utf-8")]))

            # tuple typecast: read only, faster
            return tuple(label_num)


    def convert_num_to_word(self, num_list):
        try:
            label_list = []
            for num in num_list:
                label_list.append(self.byIndex[num])

            return ' '.join(label_list)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def get_count(self):
        try:
            return self.count

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def get_unicode_file_path(self):
        try:
            return self.unicodeFilePath

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def get_blank_index(self):
        return self.byChar["-"]

    def get_space_index(self):
        return self.byChar["$"]


if __name__ == "__main__":
    labelUtil = LabelUtil()
    from stt_phone_util import generate_phone_dictionary, generate_phone_label, generate_word_dictionary, generate_zi_label, generate_py_dictionary, generate_py_label
    # generate_phone_dictionary()

    # labelUtil.load_unicode_set("resources/unicodemap_phone.csv")
    # label = generate_phone_label("x ian4 ch eng2 j ing1 j i4 zh uang4 k uang4 b i3 j iao4 k un4 n an5")
    # label = labelUtil.convert_bi_graphemes_to_num(label)

    generate_word_dictionary(["玥"])

    labelUtil.load_unicode_set("resources/unicodemap_zi.csv")
    label = generate_zi_label("而 对 楼市 成交 抑制 作用 最 大 的 限 购 玥")
    label = labelUtil.convert_bi_graphemes_to_num(label)
    print(label)
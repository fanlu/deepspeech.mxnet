#coding=utf-8
from pypinyin import pinyin, lazy_pinyin
import pypinyin
import jieba


def loadmap(filename):
    pinyin2phoneme={}
    with open(filename) as co:
        lines=co.readlines()
        for line in lines:
            line=line.strip()
            mapping=line.split('->')
            res=mapping[1].split('+')
            pinyin2phoneme[mapping[0]]=res
    return pinyin2phoneme


def splitpinyin(string):
    # print(str)
    num, li=None, None
    for s in string:
        if s.isdigit() is True:
            num=s
            li=string.split(s)
    if num is None:
        return None, string
    return num, li[0]+li[1]


def sentence2phoneme(sentence, pinyin2phoneme, seg_list=None):
    """
    :先对sentence分词，然后按词语转成拼音，在从拼音转成phoneme
    :sentence应是一个完整的句子，逗号、问号、空格、感叹号将被移除
    :param sentence: str
    :param has_split_word: bool
    :param wordslist: list
    :return:
    """
    sentence = sentence.replace('，', '').replace('。', '').replace('？', '').replace(' ', '')
    #print(sentence)
    res = []
    if seg_list is None:
        seg_list = jieba.cut(sentence, cut_all=False)
    #print(seg_list)
    for word in seg_list:
        temp=pinyin(word, style=pypinyin.TONE2)
        for ele in temp:
            if ele[0].isdigit() is True:
                print('digit %s exist, this sentence is discarded'%ele[0])
                return
            num, t = splitpinyin(ele[0])
            if t == 'zhei':
                t = 'zhe'
            if t == 'shei':
                t = 'shui'
	    if t == 'yo':
		t = 'you'
            # print(t)
            # print(pinyin2phoneme[t])
            sm = pinyin2phoneme[t][0]
            ym = pinyin2phoneme[t][1]
            if num is not None:
                ym += num
            res.append(sm)
            res.append(ym)
    return ' '.join(res)

if __name__ == '__main__':
    pinyin2phoneme = loadmap('resources/table.txt')
    res = sentence2phoneme('重 庆 晨 报 记 者 张 旭 报 道 有 梦 想 谁 都 了 不 起', pinyin2phoneme)
    print(res)


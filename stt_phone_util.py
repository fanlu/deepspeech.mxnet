import csv
from collections import Counter
from collections import OrderedDict

_data_path = "/export/fanlu/speech-to-text-wavenet/asset/data/"

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
    for index, key in enumerate(f.keys()):
      pywriter.writerow((key, index))

def generate_zi_label(label):
  try:
    str_ = label.strip().decode('utf-8')
  except:
    str_ = label.strip()
  l = []
  for ch in str_:
    if ch != u' ':
      l.append(ch.encode('utf-8'))
  return l

def dedupe(items):
  seen = set()
  for item in items:
    if item not in seen:
      yield item
      seen.add(item)


def generate_word_dictionary(label_list):
  f = OrderedDict()
  for line in open(_data_path + '6855map.txt').readlines():
    r = line.strip().split(" ")
    f[r[1].decode('utf-8')] = int(r[0])
  for label in label_list:
    try:
      str_ = label.strip().decode('utf-8')
    except:
      str_ = label.strip()
    for ch in str_:
      if ch != u' ':
          f[ch] = 1
  with open('resources/unicodemap_zi.csv', 'w') as zi_label:
    ziwriter = csv.writer(zi_label, delimiter=',')
    for index, key in enumerate(f.keys()):
      ziwriter.writerow((key.encode('utf-8'), index))


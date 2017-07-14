import csv
from collections import Counter


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
    for i, line in enumerate(open('/Users/lonica/Downloads/resource_aishell/lm_phone/lexicon.txt').readlines()):
      bigramwriter.writerow((line.split(" ")[0], i))

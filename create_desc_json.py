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

_data_path = "/Users/lonica/Downloads/"
_data_path = "/export/fanlu/aishell/"

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
  lines = open(_data_path + "resource_aishell/aishell_transcript_v0.8.txt").readlines()
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
      audio = wave.open()
      duration = float(audio.getnframes()) / audio.getframerate()
      audio.close()
      line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
      out_file.write(line + "\n")
    elif rs[0][6:11] <= "S0763":
      wav = _data_path + "data_aishell/wav/dev/" + rs[0][6:11] + "/" + rs[0] + ".wav"
      audio = wave.open()
      duration = float(audio.getnframes()) / audio.getframerate()
      audio.close()
      line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
      out_file1.write(line + "\n")
    else:
      wav = _data_path + "data_aishell/wav/test/" + rs[0][6:11] + "/" + rs[0] + ".wav"
      audio = wave.open()
      duration = float(audio.getnframes()) / audio.getframerate()
      audio.close()
      line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
      out_file2.write(line + "\n")
  out_file.close()
  out_file1.close()
  out_file2.close()


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


def word_2_pinyin():
  lines = open(_data_path + "aishell_transcript_v0.8.txt").readlines()
  with open("aishell_phone2.txt", 'w') as out_file:
    for line in lines:
      r, rs = line[:16], line[17:]
      word = rs.replace(" ", "")
      # pypinyin.pinyin(word, style=pypinyin.STYLE_INITIALS)
      # pypinyin.pinyin(word, style=pypinyin.FINALS_TONE3)
      pypinyin.pinyin(word, style=pypinyin.TONE3)


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


if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('data_directory', type=str,
  #                     help='Path to data directory')
  # parser.add_argument('output_file', type=str,
  #                     help='Path to output file')
  # args = parser.parse_args()
  # main(args.data_directory, args.output_file)

  # aishell("/Users/lonica/Downloads/AISHELL-ASR0009-OS1_sample/SPEECH_DATA/", "train1.json")

  read_lexicon()
  print(len(word_2_lexicon))
  ai_2_phone()

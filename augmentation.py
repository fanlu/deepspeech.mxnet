import scipy.io.wavfile as wav
import numpy as np
import librosa
import os
import glob
import random
from scipy.stats import randint as sp_randint
from scipy.stats.distributions import uniform, norm
from sklearn.model_selection import ParameterGrid, ParameterSampler

_data_path = "/Users/lonica/Downloads/"

_, noise_cafe = wav.read(_data_path + "cafe.wav")
_, noise_car = wav.read(_data_path + "car.wav")
_, noise_white = wav.read(_data_path + "white.wav")
noise_list = [noise_cafe, noise_car, noise_white]


def random_search():
  param_grid = {
    'noise_factor_cafe': uniform(3, 1),
    'noise_factor_car': uniform(15, 2),
    'noise_factor_white': uniform(0.05, 0.02),
    'noise_file': [0, 1, 2],
    'speed_factor': uniform(0.8, 0.4),
  }
  param_list = list(ParameterSampler(param_grid, n_iter=10))
  return [dict((k, round(v, 4) if not isinstance(v, int) else v) for (k, v) in d.items()) for d in param_list]


def grid_search():
  param_grid = [{
    'noise_factor': [round(x * 0.1, 1) for x in range(1, 10)],
    'noise_file': random.choice(noise_list),
    'speed_factor': [round(x * 0.1, 1) for x in range(8, 12)],
  }, {
    'noise_factor': [round(x * 0.1, 1) for x in range(1, 10)],
    'noise_file': random.choice(noise_list)
  }, {
    'speed_factor': [round(x * 0.1, 1) for x in range(8, 12)]
  }]
  for p in ParameterGrid(param_grid):
    yield p


def augmentation(audiofile, param, outputfile):
  if param.get('noise_file') == 0:
    noise_factor = param.get("noise_factor_cafe")
  elif param.get('noise_file') == 1:
    noise_factor = param.get("noise_factor_car")
  elif param.get('noise_file') == 2:
    noise_factor = param.get("noise_factor_white")

  sr1, data_clean = wav.read(audiofile)

  noise_a = noise_list[param.get("noise_file")]
  noise_b = np.array(noise_a).astype(np.float32)

  noise_b /= np.max(noise_b)

  data_clean_a = np.array(data_clean).astype(np.float32)

  max_holder = np.max(data_clean_a)
  data_clean_a /= np.max(data_clean_a)
  result_a = data_clean_a + noise_factor * noise_b[2000:data_clean_a.shape[0] + 2000]
  result_a *= max_holder
  # result_a = result_a.astype(np.int16)

  y_stretch = librosa.effects.time_stretch(result_a, param.get("speed_factor"))
  y_stretch = y_stretch.astype(np.int16)
  print('Saving stretched audio to: ', outputfile)
  librosa.output.write_wav(outputfile, y_stretch, sr1)
  # wav.write(outputfile, sr1, result_a)


def stretch_demo(input_file, output_file, speed):
  '''Phase-vocoder time stretch demo function.
  :parameters:
    - input_file : str
        path to input audio
    - output_file : str
        path to save output (wav)
    - speed : float > 0
        speed up by this factor
  '''

  # 1. Load the wav file, resample
  print('Loading ', input_file)

  y, sr = librosa.load(input_file)
  print type(y)

  # 2. Time-stretch through effects module
  print('Playing back at {:3.0f}% speed'.format(speed * 100))

  y_stretch = librosa.effects.time_stretch(y, speed)

  print('Saving stretched audio to: ', output_file)
  librosa.output.write_wav(output_file, y_stretch, sr)


def addnoise(audiofile, noisefile, outputfile, factor):
  """
  :param audiofile: absolute path of audio file
  :param noise: absolute path of noise file
  :param outputfile: absolute path of outputfile
  :param factor: the weight of noise data in final output audio
  :return:
  """
  _, noise = wav.read(noisefile)
  noise = np.array(noise).astype(np.float32)
  noise /= np.max(noise)
  sr1, data_clean = wav.read(audiofile)
  data_clean = np.array(data_clean).astype(np.float32)
  max_holder = np.max(data_clean)
  data_clean /= np.max(data_clean)
  result = data_clean + factor * noise[0:data_clean.shape[0]]
  result *= max_holder
  wav.write(outputfile, sr1, result)


def addspeed(inputfile, factor, outputfile):
  """
  :param inputfile:
  :param factor:
  :param outputfile:
  :return:
  """
  import wave
  CHANNELS = 1
  swidth = 2
  Change_RATE = factor

  spf = wave.open(inputfile, 'rb')
  RATE = spf.getframerate()
  signal = spf.readframes(-1)

  wf = wave.open(outputfile, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(swidth)
  wf.setframerate(RATE * Change_RATE)
  wf.writeframes(signal)
  wf.close()


# addnoise('D11_815.wav', 'white.wav', 'output_noise_white.wav', 0.003)
# addnoise('D11_815.wav', 'cafe.wav', 'output_noise_cafe.wav', 0.003)
# addnoise('D11_815.wav', 'car.wav', 'output_noise_car.wav', 0.003)
# addspeed('D11_782.wav', 1.3, 'output_speed.wav')

if __name__ == "__main__":
  wavs = glob.glob("/Users/lonica/Downloads/AISHELL-ASR0009-OS1_sample/SPEECH_DATA/*/*/*.wav")
  print(len(wavs))
  path, name = wavs[0].rsplit('/', 1)
  print(path, name)
  # for p in random_search():
  #   print(p)
  for w in wavs[100:105]:
    path, name = w.rsplit('/', 1)
    for p in random_search():
      outputfile = path + '/' + name.split('.')[0] + "-" + str(p.get("speed_factor")) + "-" + str(p.get('noise_file')) + '-'
      if p.get('noise_file') == 0:
        outputfile += str(p.get("noise_factor_cafe"))
      elif p.get('noise_file') == 1:
        outputfile += str(p.get("noise_factor_car"))
      elif p.get('noise_file') == 2:
        outputfile += str(p.get("noise_factor_white"))
      outputfile += ".wav"
      outputdir, _ = outputfile.rsplit('/', 1)
      if not os.path.exists(outputdir):
        os.mkdir(outputdir)
      print(outputfile)
      augmentation(w, p, outputfile)

import scipy.io.wavfile as wav
import numpy as np
import librosa
import os
import glob
import random
import time
import soundfile as sf
import math
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

    noisefile = noise_list[param.get("noise_file")]

    aug(audiofile, noisefile, outputfile, noise_factor, param.get("speed_factor"))


def aug(audiofile, noisefile, outputfile, noise_factor, speed_factor):
    sr1, data_clean = wav.read(audiofile)

    _, noise_a = wav.read(noisefile)
    noise_b = np.array(noise_a).astype(np.float32)

    noise_b /= np.max(noise_b)

    data_clean_a = np.array(data_clean).astype(np.float32)

    max_holder = np.max(data_clean_a)
    data_clean_a /= np.max(data_clean_a)
    start = random.randint(1, noise_b.shape[0] - data_clean_a.shape[0] - 1)
    result_a = data_clean_a + noise_factor * noise_b[start: data_clean_a.shape[0] + start]
    result_a *= max_holder
    # result_a = result_a.astype(np.int16)

    y_stretch = librosa.effects.time_stretch(result_a, speed_factor)
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
    import concurrent.futures
    from multiprocessing import cpu_count

    noise_work, sr2 = sf.read('/Users/lonica/Downloads/noise_work.wav', dtype='float32')
    # noise_work, sr2 = librosa.load('/Users/lonica/Downloads/noise_work.wav')
    st = time.time()
    result = []
    from data_utils.audio import AudioSegment
    for w in wavs[600:605]:
        path, name = w.rsplit('/', 1)
        speed = random.randint(12,12)/10.
        outputfile = path + '/' + name.split('.')[0] + "-" + str(speed) + "-" + 'work.wav'
        # audio = AudioSegment.from_file(w)
        # audio.change_speed(speed)
        # audio.to_wav_file(outputfile)
        #
        # outputfile1 = path + '/' + name.split('.')[0] + "-" + str(speed) + "-" + 'work1.wav'
        # audio, sr1 = sf.read(w, dtype='float32')
        # result_a = librosa.effects.time_stretch(audio, speed)
        # librosa.output.write_wav(outputfile1, result_a, sr1)
        noise_percent = 1
        seq_length = 600
        audio, sr1 = sf.read(w, dtype='float32')
        if random.random() < noise_percent and seq_length > 0:
            temp_audio = AudioSegment(audio, sr1)
            audio_length = audio.shape[0]
            max_length_ratio = int((float(audio_length) / (seq_length - 100) / sr1) * 10000)
            min_length_ratio = int(np.math.ceil((float(audio_length) / seq_length / sr1) * 10000))
            temp_audio.change_speed(random.randint(min_length_ratio, max_length_ratio) / 100.)
            audio = temp_audio.samples
        if random.random() < noise_percent:
            if seq_length != -1:
                max_length = seq_length * sr1 / 100
                if audio.shape[0] < max_length:
                    bg = np.zeros((max_length,))
                    rand_start = random.randint(0, max_length - audio.shape[0])
                    bg[rand_start:rand_start + audio.shape[0]] = audio
                    audio = bg
            start = random.randint(1, noise_work.shape[0] - audio.shape[0] - 1)
            audio = audio + random.randint(150, 250) / float(100.) * noise_work[start: audio.shape[0] + start]
        sf.write(outputfile, audio, sr1)
    # for w in wavs[300:400]:
    #     path, name = w.rsplit('/', 1)
    #     outputfile = path + '/' + name.split('.')[0] + "-" + str(1) + "-" + 'work.wav'
    #     # aug(w, '/Users/lonica/Downloads/noise_work.wav', outputfile, 3, 1)
    #
    #     audio, sr1 = sf.read(w, dtype='float32')
    #     max_length = int(math.ceil(audio.shape[0] / float(sr1)) * sr1)+100
    #     # audio, sr1 = librosa.load(w)
    #     bg = np.zeros((max_length,))
    #     rand_start = random.randint(1, max_length - audio.shape[0] - 1)
    #     bg[rand_start:rand_start + audio.shape[0]] = audio
    #     audio = bg
    #     start = random.randint(1, noise_work.shape[0] - audio.shape[0] - 1)
    #     result_a = audio + random.randint(150, 250) / float(100.) * noise_work[start: audio.shape[0] + start]
    #     result.append(result_a)
    #     # it's very slowly 100/5.48s
    #     # result_a = librosa.effects.time_stretch(result_a, random.randint(8, 12) / float(10.))
    #
    #     # librosa.output.write_wav(outputfile, result_a, sr1)
    #     #
    #     sf.write(outputfile, result_a, sr1)
    st1 = time.time() - st
    print("time spent is %.2f" % st1)
    # st2 = time.time()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    #     future_to_f = {executor.submit(librosa.effects.time_stretch, f, random.randint(8, 12) / float(10.)): f for f in result}
    #     for future in concurrent.futures.as_completed(future_to_f):
    #         f = future_to_f[future]
    #         try:
    #             data = future.result()
    #         except Exception as exc:
    #             print('%r generated an exception: %s' % (f, exc))
    #
    # print("time spent is %.2f" % (time.time() - st2))
    #
    # _, wav_file = wav.read(w)
    # w_f = np.array(wav_file).astype(np.float32)
    #
    # print(audio)
    # for p in random_search():
    #   outputfile = path + '/' + name.split('.')[0] + "-" + str(p.get("speed_factor")) + "-" + str(p.get('noise_file')) + '-'
    #   if p.get('noise_file') == 0:
    #     outputfile += str(p.get("noise_factor_cafe"))
    #   elif p.get('noise_file') == 1:
    #     outputfile += str(p.get("noise_factor_car"))
    #   elif p.get('noise_file') == 2:
    #     outputfile += str(p.get("noise_factor_white"))
    #   outputfile += ".wav"
    #   outputdir, _ = outputfile.rsplit('/', 1)
    #   if not os.path.exists(outputdir):
    #     os.mkdir(outputdir)
    #   print(outputfile)
    #   augmentation(w, p, outputfile)

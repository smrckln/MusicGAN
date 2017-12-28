import librosa
import numpy as np

import os
import re

class FeatureExtractor(object):

    def __init__(self, hop_length=512):
        self.hop_length = hop_length

    def extract_features(self, files=[]):
        timeseries_length = 128
        data = np.zeros((len(files), timeseries_length, 33), dtype=np.float64)

        files = self.get_full_file_names(files)
        targets = self.get_target_names(files)

        for i, file in enumerate(files):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)

            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

            print("Extracted features audio track %i of %i." % (i + 1, len(files)))


        return data, targets

    def get_full_file_names(self, files):
        return [os.path.abspath(file) for file in files]

    def get_target_names(self, files):
        targets = []
        regex = re.compile(r".*/(.*)\.(.*)\.(.*)")
        for file_name in files:
            matches = regex.match(file_name)
            if matches:
                targets.append(matches[1])

        return np.expand_dims(np.asarray(targets), axis=1)

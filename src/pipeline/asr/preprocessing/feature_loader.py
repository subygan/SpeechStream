from typing import Optional

import numpy as np
import torch


class ExternalFeatureLoader(object):
    """Feature loader that load external features store in certain format.
    Currently support pickle, npy and npz format.
    """

    def __init__(
        self, augmentor: Optional["nemo.collections.asr.parts.perturb.FeatureAugmentor"] = None,
    ):
        """
        Feature loader
        """
        self.augmentor = augmentor

    def load_feature_from_file(self, file_path: str):
        """Load samples from file_path and convert it to be of type float32
        file_path (str) is the path of the file that stores feature/sample.
        """

        if file_path.endswith(".pt") or file_path.endswith(".pth"):
            samples = torch.load(file_path, map_location="cpu").float().numpy()
            return samples
        else:
            # load pickle/npy/npz file
            samples = np.load(file_path, allow_pickle=True)
            return self._convert_samples_to_float32(samples)
            # TODO load other type of files such as kaldi io ark

    @staticmethod
    def _convert_samples_to_float32(samples: np.ndarray) -> np.ndarray:
        """Convert sample type to float32.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    def process(self, file_path: str) -> torch.Tensor:
        features = self.load_feature_from_file(file_path)
        features = self.process_segment(features)
        return features

    def process_segment(self, feature_segment):
        if self.augmentor:
            # augmentor for external features. Here possible augmentor for external embedding feature is Diaconis Augmentation and might be implemented later
            self.augmentor.perturb(feature_segment)
            return torch.tensor(feature_segment, dtype=torch.float)

        return torch.tensor(feature_segment, dtype=torch.float)

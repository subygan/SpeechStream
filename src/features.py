import numpy as np

from src.perturb import AudioAugmentor
from src.segment import AudioSegment
import torch


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(
        self,
        file_path,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
    ):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            trim_ref=trim_ref,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            trim_hop_length=trim_hop_length,
            orig_sr=orig_sr,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)
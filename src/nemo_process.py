import argparse
import os

from helpers import *
import torch
from pydub import AudioSegment
from pipeline.model import NeuralDiarizer
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


class NemoDiarizer:
    def __init__(self, audio_path: str, workdir: str):
        self.audio_path = audio_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thread = None
        self.model = None
        self._workdir = workdir

    def start_diarize(self):
        # convert audio to mono for NeMo combatibility
        sound = AudioSegment.from_file(self.audio_path).set_channels(1)
        sound.export(os.path.join(self._workdir, "mono_file.wav"), format="wav")

        print("Starting diarization")
        # TODO: move this to a config file
        conf = create_config("/app/configs/diar_infer_telephonic.yaml", self._workdir)
        # Initialize NeMo MSDD diarization model

        # create_config(temp_path)
        self.model = NeuralDiarizer(cfg=conf).to(self.device)
        self.model.diarize()

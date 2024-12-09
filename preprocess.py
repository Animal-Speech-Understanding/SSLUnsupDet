import glob

import numpy as np
import soundfile as sf


class WatkinsSpermWhalePreprocess:
    def __init__(
        self, base_path: str = "data/wavs/*.wav", window_width: float = 0.5
    ) -> None:
        self.wav_files = sorted(glob.glob(base_path))
        self.window_width = window_width

    def get_info(self) -> tuple[list[str], list[float], list[int]]:
        srs = []
        durs = []
        wavs = []
        for w in self.wav_files:
            info = sf.info(w)
            sr = info.samplerate
            dur = info.duration
            if dur > self.window_width:
                srs.append(sr)
                durs.append(dur)
                wavs.append(w)
        return wavs, durs, srs

    def run(self) -> tuple[list[str], np.ndarray]:
        wavs, durs, _ = self.get_info()
        weights = self.probability_weights(durs)
        return wavs, weights

    @staticmethod
    def probability_weights(durations: list[float]) -> np.ndarray:
        durations_array = np.array(durations, dtype=np.float32)
        weights = durations_array / durations_array.sum()
        return weights

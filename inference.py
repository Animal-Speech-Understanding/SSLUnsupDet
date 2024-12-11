import argparse
import glob
import json
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import tqdm

from models import SpectralBoundaryEncoder
from utils import box_convolve, gen_spec, get_filename, spectral_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    # parser.add_argument("-ckpt", "--checkpoint", type=int, help="choose the model ckpt")
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        help="choose the search regime (e.g. coarse or fine)",
    )

    args = parser.parse_args()
    config = args.config
    # ckpt = args.checkpoint
    search = args.search
    assert search in ["coarse", "fine"]

    with open(f"configs/{args.config}") as f:
        data = f.read()
    f.close()
    config = json.loads(data)

    save_dir = config["utils"]["save_dir"]

    dataset_params = config["dataset"]
    training_params = config["training"]
    model_params = config["model"]
    inference_params = config["inference"]

    wav_files = sorted(glob.glob(dataset_params["wavs_path"]))
    print(f"Loaded {len(wav_files)} Wav Files")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpectralBoundaryEncoder(**model_params)
    model.to(device)
    if os.path.isdir(f"{save_dir}/models"):
        model.load_state_dict(
            torch.load(f"{save_dir}/models/final_model.pt", map_location=device, weights_only=True)
        )
    else:
        checkpoint = torch.load(f"{save_dir}/checkpoints/best-checkpoint.ckpt", map_location=device)

        # Handle the state_dict key
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # If it's directly the state_dict
        from collections import OrderedDict
        # Handle unexpected key prefixes
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("model.", "")  # Remove prefix if needed
            new_state_dict[name] = v

        # Load the state_dict into the model
        model.load_state_dict(new_state_dict)
    model.eval()
    print("Loaded Model")

    if not os.path.isdir(f"{save_dir}/Inference"):
        os.mkdir(f"{save_dir}/Inference")
    print("Made Inference Directory")

    alpha = np.arange(*inference_params[search]["alpha"])
    beta = np.arange(*inference_params[search]["beta"])
    smooth_duration = np.arange(*inference_params[search]["smooth_duration"])
    if smooth_duration[0] == 0:
        smooth_duration[0] = 1

    threshold = inference_params["duration_threshold"]

    print("Entering Inference Loop")
    for wav in tqdm.tqdm(wav_files):
        alphas = []
        betas = []
        smooth_durations = []
        distances = []

        wav_dur = sf.info(wav).duration
        if wav_dur > threshold:
            continue

        filename = get_filename(wav)
        if not os.path.isdir(f"{save_dir}/Inference/{filename}"):
            os.mkdir(f"{save_dir}/Inference/{filename}")

        x, _ = librosa.load(wav, sr=dataset_params["sample_rate"])
        x = torch.tensor(x).reshape(1, 1, -1)
        x = x.to(device)

        with torch.no_grad():
            z = model(x).detach().cpu()

        for a in alpha:
            for b in beta:
                distance = spectral_distances(
                    z, metric="bounded_euclidean", alpha=a, beta=b
                )
                for s in smooth_duration:
                    alphas.append(a)
                    betas.append(b)
                    smooth_durations.append(s)
                    dist = box_convolve(distance, s)
                    distances.append(dist)

        df = pd.DataFrame(
            {
                "Alphas": alphas,
                "Betas": betas,
                "SmoothDurations": smooth_durations,
                "Distances": distances,
            }
        )
        df.to_pickle(
            f"{save_dir}/Inference/{filename}/distances_{search}.pkl"
        )

        D = gen_spec(
            x.cpu().squeeze().numpy(),
            n=model_params["transform_params"]["params"]["kernel_size"],
            h=model_params["transform_params"]["params"]["stride"],
        )

        D_res = gen_spec(
            x.cpu().squeeze().numpy(),
            n=model_params["transform_params"]["params"]["kernel_size"] * 10,
            h=model_params["transform_params"]["params"]["stride"] * 10,
        )

        np.save(f"{save_dir}/Inference/{filename}/spectrogram.npy", D)
        np.save(f"{save_dir}/Inference/{filename}/spectrogram_res.npy", D_res)
        np.save(
            f"{save_dir}/Inference/{filename}/sbgram.npy",
            z.squeeze().numpy().T,
        )
        print(f"Saved for {filename}")

        if device == "cuda":
            torch.cuda.synchronize()

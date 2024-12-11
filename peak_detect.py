import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from scipy.signal import find_peaks

from logging_config import get_logger, setup_logging
from metrics import PRF1Metric
from utils import coarsen_selections, frame_to_time

# Global variable for PRF1Metrics to avoid passing it to each process
global_prf1metrics = None


def parse_arguments() -> Dict:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process distance files with peak detection."
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="JSON file for configuration"
    )
    parser.add_argument(
        "-q",
        "--use_q",
        type=int,
        choices=[0, 1],
        help="Use uncertain (q) selection tables (0 or 1)",
        default=1,
    )
    # parser.add_argument(
    #     "-ckpt",
    #     "--checkpoint",
    #     type=int,
    #     required=True,
    #     help="Choose the model checkpoint",
    # )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        choices=["coarse", "fine"],
        required=True,
        help="Choose the search regime (e.g., coarse or fine)",
    )
    args = parser.parse_args()
    return vars(args)  # Return as a dictionary for easier access


def initialize_metrics(tolerance: List[float]) -> List[PRF1Metric]:
    """Initializes PRF1Metric objects for each tolerance level."""
    return [PRF1Metric(tolerance=t) for t in tolerance]


def load_config(config_path: str) -> Dict:
    """Loads the JSON configuration file."""
    with open(config_path) as f:
        return json.load(f)


def find_selection_table(
    filename: str, use_q: bool, logger: logging.Logger
) -> Tuple[pd.DataFrame, str]:
    """
    Finds and loads the selection table for a given filename.

    Args:
        filename (str): The base filename.
        use_q (bool): Whether to use uncertain (q) selection tables.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        Tuple[pd.DataFrame, str]: The selection table and the '.q' suffix if applicable.
    """
    base_path = f"data/selections/{filename}"
    q_suffix = ""
    if use_q:
        q_path = f"{base_path}.q.selections.txt"
        selections_path = (
            q_path if os.path.exists(q_path) else f"{base_path}.selections.txt"
        )
        q_suffix = ".q" if os.path.exists(q_path) else ""
    else:
        selections_path = f"{base_path}.selections.txt"

    if os.path.exists(selections_path):
        logger.info(f"Loaded selection table: {selections_path}")
        return pd.read_csv(selections_path, sep="\t"), q_suffix
    else:
        logger.warning(f"No selection table found for {filename}")
        return None, ""


def process_selection_table(
    selections_table: pd.DataFrame, coarsen: List[float], logger: logging.Logger
) -> Dict:
    """
    Processes the selection table to extract and coarsen onsets and offsets.

    Args:
        selections_table (pd.DataFrame): The loaded selection table.
        coarsen (List[float]): Coarsening thresholds.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        Dict: Processed onsets, offsets, and midpoints.
    """
    assert (
        selections_table.View.iloc[0] == "Waveform 1"
    ), "First View should be 'Waveform 1'"
    assert (
        selections_table.View.iloc[1] == "Spectrogram 1"
    ), "Second View should be 'Spectrogram 1'"

    selections_table = selections_table[::2].reset_index(
        drop=True
    )  # Skip alternate rows
    begin_times = selections_table["Begin Time (s)"].to_numpy()
    end_times = selections_table["End Time (s)"].to_numpy()

    onsets = {0: begin_times}
    offsets = {0: end_times}

    for c in coarsen:
        onsets[c], offsets[c] = coarsen_selections(begin_times, end_times, threshold=c)

    midpoints = {k: (onsets[k] + offsets[k]) / 2 for k in onsets.keys()}
    logger.debug(f"Processed selection table with coarsen thresholds: {coarsen}")
    return {"onsets": onsets, "offsets": offsets, "midpoints": midpoints}


def detect_peaks(
    distance: np.ndarray, min_prominence: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects peaks in the distance signal based on a minimum prominence.

    Args:
        distance (np.ndarray): The distance signal.
        min_prominence (float): Minimum prominence for peaks.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Detected peak indices and their prominences.
    """
    pks, properties = find_peaks(distance, prominence=min_prominence)
    prominences = properties["prominences"]
    return pks, prominences


def process_distance_file(
    wav_distance: str,
    prominence_list: List[float],
    coarsen: List[float],
    hop_length: int,
    sample_rate: int,
    save_dir: str,
    # ckpt: int,
    search: str,
    use_q: bool,
):
    """
    Processes a single distance file to compute and save metrics.

    Args:
        wav_distance (str): Path to the distance file.
        prominence_list (List[float]): List of prominence values, sorted in ascending order.
        coarsen (List[float]): Coarsening thresholds.
        hop_length (int): Frame hop length.
        sample_rate (int): Sampling rate.
        save_dir (str): Directory to save results.
        ckpt (int): Model checkpoint number.
        search (str): Search regime ('coarse' or 'fine').
        use_q (bool): Whether to use uncertain (q) selection tables.

    Returns:
        None
    """
    logger = logging.getLogger()

    try:
        filename = re.findall(r"\d+\w+", wav_distance)[0]
    except IndexError:
        logger.error(f"Filename extraction failed for {wav_distance}")
        return

    selections_table, q_suffix = find_selection_table(filename, use_q, logger)

    if selections_table is None:
        return

    wav_save_path = (
        f"{save_dir}/Inference/{filename}/predictions_{search}{q_suffix}.pkl"
    )

    if os.path.exists(wav_save_path):
        logger.info(f"Results already exist: {wav_save_path}")
        return

    processed_data = process_selection_table(selections_table, coarsen, logger)
    onsets = processed_data["onsets"]
    midpoints = processed_data["midpoints"]

    N_signal = {k: len(v) for k, v in onsets.items()}

    try:
        distances_table = pd.read_pickle(wav_distance)
    except Exception as e:
        logger.error(f"Failed to load distances from {wav_distance}: {e}")
        return

    # Pre-round the necessary columns for efficiency
    distances_table["Alphas"] = distances_table["Alphas"].round(2)
    distances_table["Betas"] = distances_table["Betas"].round(2)
    distances_table["SmoothDurations"] = distances_table["SmoothDurations"].round(2)

    results = []

    # Determine the minimum prominence needed
    min_prominence = min(prominence_list)

    # Iterate through each row using itertuples for better performance
    for row in distances_table.itertuples(index=False):
        distance = row.Distances
        all_pks, all_prominences = detect_peaks(distance, min_prominence)

        # For each prominence threshold, filter peaks
        for p in prominence_list:
            # Get peaks that meet the current prominence threshold
            valid_indices = all_prominences >= p
            pks = all_pks[valid_indices]

            detections = frame_to_time(pks, hop_length, sample_rate)

            preds_wrt_onsets = {
                c: {
                    prf1.tolerance: prf1.classify_predictions(onset_times, detections)
                    for prf1 in global_prf1metrics
                }
                for c, onset_times in onsets.items()
            }
            preds_wrt_midpoints = {
                c: {
                    prf1.tolerance: prf1.classify_predictions(
                        midpoint_times, detections
                    )
                    for prf1 in global_prf1metrics
                }
                for c, midpoint_times in midpoints.items()
            }

            results.append(
                {
                    "Alphas": row.Alphas,
                    "Betas": row.Betas,
                    "SmoothDurations": row.SmoothDurations,
                    "Prominence": p,
                    "N_signals": N_signal,
                    "N_detections": len(detections),
                    "OnsetPreds{coarsen:{tol:(TP,FN,FP)}}": preds_wrt_onsets,
                    "MidpointPreds{coarsen:{tol:(TP,FN,FP)}}": preds_wrt_midpoints,
                }
            )

    try:
        df = pd.DataFrame(results)
        df.to_pickle(wav_save_path)
        logger.info(f"Saved results: {wav_save_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {wav_save_path}: {e}")


def process_distance_file_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    return process_distance_file(*args)


def init_worker(prf1metrics_init, log_queue: mp.Queue):
    """Initializer for each worker process to set global variables and logging."""
    global global_prf1metrics
    global_prf1metrics = prf1metrics_init
    # Setup logger for the worker
    get_logger(log_queue)


if __name__ == "__main__":
    # Setup logging in the main process
    log_queue, listener = setup_logging("logs", base_filename="peak_detect")
    logger = get_logger(log_queue)

    try:
        args = parse_arguments()

        use_q = True if args["use_q"] == 1 else False
        # ckpt = args["checkpoint"]

        config = load_config(f"configs/{args['config']}")
        logger.info(f"Loaded configuration from {args['config']}")

        save_dir = config["utils"]["save_dir"]
        hop_length = config["model"]["transform_params"]["params"]["stride"]
        sample_rate = config["dataset"]["sample_rate"]
        tolerance = config["metrics"]["tolerance"]
        coarsen = config["metrics"]["coarsen"]

        search = args["search"]
        assert search in ["coarse", "fine"], "Search regime must be 'coarse' or 'fine'"
        prominence = sorted(
            np.arange(*config["detection"][search]["prominence"])
        )  # Ensure ascending order

        prf1metrics = initialize_metrics(tolerance)
        logger.info(f"Initialized PRF1 metrics with tolerance levels: {tolerance}")

        wav_distances = sorted(
            glob.glob(f"{save_dir}/Inference/*/distances_{search}.pkl")
        )

        if not wav_distances:
            logger.error("No distance files found.")
            raise Exception("No distance files found.")
        else:
            logger.info(f"Found {len(wav_distances)} distance files to process.")

        args_list = [
            (
                wav_distance,
                prominence,
                coarsen,
                hop_length,
                sample_rate,
                save_dir,
                # ckpt,
                search,
                use_q,
            )
            for wav_distance in wav_distances
        ]

        num_cpus = mp.cpu_count()
        logger.info(f"Starting processing with {num_cpus} CPU cores.")

        with mp.Pool(
            processes=num_cpus,
            initializer=init_worker,
            initargs=(prf1metrics, log_queue),
        ) as pool:
            # Use tqdm to display a progress bar
            list(
                tqdm.tqdm(
                    pool.imap_unordered(process_distance_file_wrapper, args_list),
                    total=len(args_list),
                    desc="Processing Distance Files",
                )
            )

        logger.info("Processing completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred in the main process: {e}")
    finally:
        # Stop the logging listener
        listener.stop()

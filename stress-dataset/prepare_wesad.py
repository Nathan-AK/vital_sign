import gc
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
WESAD_DIR = DATASET_DIR / "WESAD"

OUTPUT_FILE = DATASET_DIR / "wesad_all_features.csv"


# ============================================================
# WESAD SAMPLING RATES
# ============================================================

BVP_FS = 64
ACC_FS = 32
TEMP_FS = 4
LABEL_FS = 700


# ============================================================
# WINDOW CONFIGURATION
# ============================================================

WINDOW_SECONDS = 60
STEP_SECONDS = 30

LABEL_VALID_VALUES = [1, 2, 3, 4]
MIN_VALID_LABEL_RATIO = 0.80
MIN_LABEL_PURITY = 0.80


# ============================================================
# HEART RATE EXTRACTION FROM BVP
# ============================================================

def extract_heart_rate(bvp_signal):
    """
    Estimate heart rate from wrist BVP signal.
    Returns BPM or None if the signal is invalid.
    """

    bvp_signal = np.asarray(bvp_signal, dtype=float).flatten()

    if len(bvp_signal) < BVP_FS * 10:
        return None

    if not np.all(np.isfinite(bvp_signal)):
        return None

    try:
        # BVP pulse frequency band
        sos = butter(
            3,
            [0.5, 4.0],
            btype="bandpass",
            fs=BVP_FS,
            output="sos"
        )

        filtered = sosfiltfilt(sos, bvp_signal)

        signal_std = np.std(filtered)

        if signal_std < 1e-10:
            return None

        # Detect pulse peaks
        peaks, _ = find_peaks(
            filtered,
            distance=int(0.35 * BVP_FS),
            prominence=0.25 * signal_std
        )

        if len(peaks) < 3:
            return None

        # Convert peak intervals to BPM
        intervals = np.diff(peaks) / BVP_FS

        bpm_values = 60.0 / intervals

        # Remove implausible estimates
        bpm_values = bpm_values[
            (bpm_values >= 40) &
            (bpm_values <= 200)
        ]

        if len(bpm_values) == 0:
            return None

        return float(np.median(bpm_values))

    except Exception:
        return None


# ============================================================
# MOTION FEATURE EXTRACTION FROM ACC
# ============================================================

def extract_motion_features(acc_signal):
    """
    Extract motion features from 3-axis wrist acceleration.
    """

    acc_signal = np.asarray(acc_signal, dtype=float)

    if acc_signal.ndim != 2 or acc_signal.shape[1] != 3:
        return None

    if not np.all(np.isfinite(acc_signal)):
        return None

    # Magnitude of 3-axis acceleration
    magnitude = np.sqrt(np.sum(acc_signal ** 2, axis=1))

    if len(magnitude) < 2:
        return None

    motion_level = float(np.std(magnitude))
    motion_change = float(np.mean(np.abs(np.diff(magnitude))))

    return {
        "motion_level": motion_level,
        "motion_change": motion_change,
    }


# ============================================================
# TEMPERATURE FEATURE EXTRACTION
# ============================================================

def extract_temperature_features(temp_signal):
    """
    Extract wrist skin temperature features.
    WESAD temperature is not directly equivalent to DS18B20.
    """

    temp_signal = np.asarray(temp_signal, dtype=float).flatten()

    if len(temp_signal) == 0:
        return None

    if not np.all(np.isfinite(temp_signal)):
        return None

    return {
        "skin_temp": float(np.mean(temp_signal)),
        "temp_std": float(np.std(temp_signal)),
        "temp_min": float(np.min(temp_signal)),
        "temp_max": float(np.max(temp_signal)),
    }


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject):

    pkl_file = WESAD_DIR / subject / f"{subject}.pkl"

    print(f"Reading: {pkl_file}")

    # Load one subject at a time to limit memory usage
    with open(pkl_file, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    wrist = data["signal"]["wrist"]

    bvp = np.asarray(wrist["BVP"], dtype=float).flatten()
    acc = np.asarray(wrist["ACC"], dtype=float)
    temp = np.asarray(wrist["TEMP"], dtype=float).flatten()
    labels = np.asarray(data["label"]).flatten()

    del data
    gc.collect()

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("Unexpected wrist ACC shape")

    # Determine common available duration
    total_seconds = min(
        len(bvp) / BVP_FS,
        len(acc) / ACC_FS,
        len(temp) / TEMP_FS,
        len(labels) / LABEL_FS
    )

    if total_seconds < WINDOW_SECONDS:
        raise ValueError("Subject recording is shorter than one window")

    window_bvp = WINDOW_SECONDS * BVP_FS
    step_bvp = STEP_SECONDS * BVP_FS

    features = []

    number_of_windows = int(
        (total_seconds - WINDOW_SECONDS) // STEP_SECONDS
    ) + 1

    for window_number in range(number_of_windows):

        start_seconds = window_number * STEP_SECONDS
        end_seconds = start_seconds + WINDOW_SECONDS

        # Convert time boundaries into each sensor's sample indices
        bvp_start = round(start_seconds * BVP_FS)
        bvp_end = round(end_seconds * BVP_FS)

        acc_start = round(start_seconds * ACC_FS)
        acc_end = round(end_seconds * ACC_FS)

        temp_start = round(start_seconds * TEMP_FS)
        temp_end = round(end_seconds * TEMP_FS)

        label_start = round(start_seconds * LABEL_FS)
        label_end = round(end_seconds * LABEL_FS)

        bvp_window = bvp[bvp_start:bvp_end]
        acc_window = acc[acc_start:acc_end]
        temp_window = temp[temp_start:temp_end]
        label_window = labels[label_start:label_end]

        if (
            len(bvp_window) != window_bvp
            or len(acc_window) == 0
            or len(temp_window) == 0
            or len(label_window) == 0
        ):
            continue

        # Keep only recognized WESAD labels
        valid_mask = np.isin(
            label_window,
            LABEL_VALID_VALUES
        )

        valid_label_ratio = float(np.mean(valid_mask))

        if valid_label_ratio < MIN_VALID_LABEL_RATIO:
            continue

        valid_labels = label_window[valid_mask].astype(int)

        # Map WESAD labels to binary classification:
        # 1 = stress
        # 0 = non-stress
        binary_labels = np.where(
            valid_labels == 2,
            1,
            0
        )

        counts = Counter(binary_labels.tolist())

        majority_target, majority_count = counts.most_common(1)[0]

        label_purity = majority_count / len(binary_labels)

        if label_purity < MIN_LABEL_PURITY:
            continue

        # Extract features
        heart_rate = extract_heart_rate(bvp_window)

        motion_features = extract_motion_features(acc_window)

        temperature_features = extract_temperature_features(
            temp_window
        )

        if (
            heart_rate is None
            or motion_features is None
            or temperature_features is None
        ):
            continue

        original_label_counts = Counter(valid_labels.tolist())
        original_label = original_label_counts.most_common(1)[0][0]

        row = {
            "subject": subject,
            "start_seconds": start_seconds,
            "heart_rate": heart_rate,
            **temperature_features,
            **motion_features,
            "target": int(majority_target),
            "original_label": int(original_label),
            "label_purity": float(label_purity),
            "valid_label_ratio": valid_label_ratio,
        }

        features.append(row)

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not WESAD_DIR.exists():
        print(f"WESAD directory not found: {WESAD_DIR}")
        return

    subjects = sorted(
        folder.name
        for folder in WESAD_DIR.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}.pkl").exists()
    )

    print("===================================")
    print("WESAD FEATURE PREPROCESSING")
    print("===================================")
    print("Subjects found:", subjects)
    print("Window:", WINDOW_SECONDS, "seconds")
    print("Step:", STEP_SECONDS, "seconds")
    print("===================================")

    all_features = []
    successful_subjects = []
    skipped_subjects = []

    for subject in subjects:

        print(f"\nLoading {subject}...")

        try:
            subject_features = process_subject(subject)

            if subject_features:

                all_features.extend(subject_features)
                successful_subjects.append(subject)

                print(
                    f"{subject}: "
                    f"{len(subject_features)} valid windows"
                )

                # Save progress after each successful subject
                df_checkpoint = pd.DataFrame(all_features)
                df_checkpoint.to_csv(OUTPUT_FILE, index=False)

                print("Checkpoint saved:", OUTPUT_FILE)

            else:
                print(f"{subject}: No valid windows")
                skipped_subjects.append(subject)

        except Exception as error:

            print(f"SKIPPING {subject}: {error}")
            skipped_subjects.append(subject)

        finally:
            gc.collect()

    if not all_features:
        print("\nNo features generated.")
        return

    df = pd.DataFrame(all_features)

    # Final save
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n===================================")
    print("WESAD PREPROCESSING SUMMARY")
    print("===================================")

    print("Successful subjects:", successful_subjects)
    print("Skipped subjects:", skipped_subjects)

    print("\nTotal windows:", len(df))

    print("\nSubject distribution:")
    print(df["subject"].value_counts())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nFeature preview:")
    print(df.head())

    print("\nSaved:", OUTPUT_FILE)
    print("===================================")


if __name__ == "__main__":
    main()
import numpy as np
from trial import Trial
from scipy.ndimage import median_filter

def to_windows(arr, win=200):
    """
    Splits the input array into non-overlapping windows of specified size.

    Parameters:
    arr (numpy.ndarray): The input array of shape (n_channels, n_samples).
    win (int): The size of each window.

    Returns:
    numpy.ndarray: The reshaped array of shape (n_channels, n_windows, win).
    """
    n_samples = arr.shape[1]
    n_trimmed = n_samples - (n_samples % win)
    arr_trimmed = arr[:, :n_trimmed]
    arr_reshaped = arr_trimmed.reshape(arr.shape[0], -1, win)
    return arr_reshaped

def window_variance(arr, win=200):
    variance = to_windows(arr, win=win).var(axis=-1)
    return variance

def window_mean(arr, win=200):
    mean = to_windows(arr, win=win).mean(axis=-1)
    return mean

def window_biggest_jump(arr, win=200):
    w = to_windows(arr,win=win)
    biggest_jump = np.abs(np.diff(w, axis=-1)).max(axis=-1)
    return biggest_jump


def get_diff(arr, damaged):
    clean_mean = window_mean(arr)
    dmg_mean = window_mean(damaged)
    diff = (dmg_mean - clean_mean)[[3, 4]][:, 5:8]
    normal_size = np.sqrt(np.median(window_variance(arr)[[3, 4]], axis=1))
    return np.round(diff / normal_size[:, None], 2)

def window_scores(arr, win=200):
    #var_score: each window’s variance divided by that channel’s median window variance. Flatline gives about 0, noise about 10, and normal windows are around 1.
    var = window_variance(arr, win=win)
    var_score = var / np.median(var, axis=-1, keepdims=True)

    mean = window_mean(arr, win=win)
    typical = np.sqrt(np.median(var, axis=-1, keepdims=True))
    mean_score = abs(mean - np.median(mean, axis=-1, keepdims=True)) / typical

    return var_score, mean_score

def window_labels(mask, win=200):
    mask_windowed = to_windows(mask, win=win).mean(axis=-1) > 0.5
    return mask_windowed

def detect(var_score, mean_score):
    # Returns a boolean array indicating which windows are likely to be corrupted based on variance and mean scores.
    # A window is considered corrupted if its variance score is less than 0.1 (flatline) or greater than 10 (noise), or if its mean score is greater than 3 (jump).
    return (var_score < 0.1) | (var_score > 9) | (mean_score > 2)

def window_features(arr, win=200):
    var = window_variance(arr, win=win)
    typical = np.sqrt(np.median(var,axis=-1, keepdims=True))

    var_score, _ = window_scores(arr, win=win)

    mean = window_mean(arr, win=win)
    baseline = median_filter(mean, size=(1,21), mode="nearest")
    mean_score = np.abs(mean-baseline) / typical

    biggest_jump = window_biggest_jump(arr,win=win)
    jump_score = biggest_jump / typical
    return np.stack([var_score, mean_score, jump_score],axis=-1)

if __name__ == "__main__":
    t = Trial("data/A01T")   # your path
    arr = t.arr

    



    # var_score, mean_score = window_scores(arr)
    # print(f"Var Score: {np.percentile(var_score, [1, 50, 99])}")
    # print(f"Mean Score: {np.percentile(mean_score, [1, 50, 99])}")
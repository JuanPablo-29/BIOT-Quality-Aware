import numpy as np
from trial import Trial

def corrupt_flatline(arr, channels, start, end):
    """
    Corrupts the given channels in the array by setting their values to zero (flatline) between the specified start and end indices.

    Parameters:
    arr (numpy.ndarray): The input array of shape (n_channels, n_samples).
    channels (list): A list of channel indices to be corrupted.
    start (int): The starting index for corruption.
    end (int): The ending index for corruption.

    Returns:
    numpy.ndarray: The corrupted array with specified channels flatlined between start and end indices.
    """
    corrupted_arr = arr.copy()
    corrupted_arr[channels, start:end] = 0
    mask = np.zeros(arr.shape, dtype=bool)
    mask[channels, start:end] = True
    return corrupted_arr, mask

def corrupt_noise(arr, channels, start, end, level, seed=None):
    """
    Corrupts the given channels in the array by adding Gaussian noise between the specified start and end indices.

    Parameters:
    arr (numpy.ndarray): The input array of shape (n_channels, n_samples).
    channels (list): A list of channel indices to be corrupted.
    start (int): The starting index for corruption.
    end (int): The ending index for corruption.
    level (float): The standard deviation of the Gaussian noise to be added.
    seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
    numpy.ndarray: The corrupted array with specified channels having added noise between start and end indices.
    """
    rng = np.random.default_rng(seed)
    corrupted_arr = arr.copy()
    normal_size = np.sqrt(np.median(window_variance(arr)[channels], axis=1))
    noise = rng.normal(0, level * normal_size[:,None], size=(len(channels), end - start))
    corrupted_arr[channels, start:end] += noise
    mask = np.zeros(arr.shape, dtype=bool)
    mask[channels, start:end] = True
    return corrupted_arr, mask

def corrupt_jump(arr, channels, start, end, size):
    """
    Corrupts the given channels in the array by adding a jump (step change) between the specified start and end indices.

    Parameters:
    arr (numpy.ndarray): The input array of shape (n_channels, n_samples).
    channels (list): A list of channel indices to be corrupted.
    start (int): The starting index for corruption.
    end (int): The ending index for corruption.
    size (float): The magnitude of the jump to be added.

    Returns:
    numpy.ndarray: The corrupted array with specified channels having a jump between start and end indices.
    """
    corrupted_arr = arr.copy()
    normal_size = np.sqrt(np.median(window_variance(arr)[channels], axis=1))
    corrupted_arr[channels, start:end] += normal_size[:,None] * size
    mask = np.zeros(arr.shape, dtype=bool)
    mask[channels, start:end] = True
    return corrupted_arr, mask

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

def window_labels(mask, win=200):
    mask_windowed = to_windows(mask, win=win).mean(axis=-1) > 0.5
    return mask_windowed

def window_variance(arr, win=200):
    variance = to_windows(arr, win=win).var(axis=-1)
    return variance

def get_ratio(arr, damaged, channels):
    clean_var = window_variance(arr)
    dmg_var = window_variance(damaged)
    ratio = dmg_var[channels] / clean_var[channels]
    return ratio

if __name__ == "__main__":
    t = Trial("data/A01T")   # your path
    arr = t.arr
    damaged, mask = corrupt_flatline(arr, [3, 4], 1000, 1600)
    ratio = get_ratio(arr, damaged, [3, 4])
    print(f"Flatline ratio: {np.round(ratio[:,5:8], 2)}")

    damaged, mask = corrupt_noise(arr, [3, 4], 1000, 1600, level=3, seed=0)
    ratio = get_ratio(arr, damaged, [3, 4])
    print(f"Noise ratio: {np.round(ratio[:,5:8], 2)}")

    damaged, mask = corrupt_jump(arr, [3, 4], 1000, 1600, size=3)
    ratio = get_ratio(arr, damaged, [3, 4])
    print(f"Jump ratio: {np.round(ratio[:,5:8], 2)}")

    clean_var = window_variance(arr)
    print(arr[3].var())                  # variance of the whole channel
    print(np.median(clean_var[3]))       # typical window variance
    print(clean_var[3, 5:8])             # the three windows you damaged

    # damaged, mask = corrupt_flatline(arr, [3, 4], 1000, 1600)
    # labels = window_labels(mask, win=200)
    # print(np.where(labels))
    # print(labels.sum())
    # damaged, mask = corrupt_flatline(arr, [3, 4], 1100, 1700)
    # labels = window_labels(mask, win=200)
    # print(np.where(labels))
    # print(labels.sum())

    # print(damaged[3, 1000:1600].std())
    # print(damaged[4, 1000:1600].std())
    # print(np.array_equal(damaged[~mask], arr[~mask]))
    # print(mask.sum() == 2 * 600)

    # damaged, mask = corrupt_noise(arr, [3, 4], 1000, 1600, level=3, seed=0)
    # print((damaged-arr)[3:5, 1000:1600].std(axis=1)/arr[3:5].std(axis=1))
    # print(np.array_equal(damaged[~mask], arr[~mask]))
    # print(mask.sum() == 2 * 600)

    # damaged, mask = corrupt_jump(arr, [3, 4], 1000, 1600, size=3)
    # print(np.allclose((damaged - arr)[3:5, 1000:1600], 3 * arr[3:5].std(axis=1)[:, None]))
    # print(np.array_equal(damaged[~mask], arr[~mask]))

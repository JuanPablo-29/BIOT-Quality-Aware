import numpy as np
from trial import Trial
from detector import window_variance, detect, window_scores, window_labels



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

def evaluate(damaged, mask):
    flag = detect(*window_scores(damaged))
    labels = window_labels(mask)
    recall = (flag & labels).sum() / labels.sum()
    false_alarm = (flag & ~labels).sum() / (~labels).sum()
    return recall, false_alarm

if __name__ == "__main__":
    # pass
    t = Trial("data/A01T")   # your path
    arr = t.arr


    
    for level in [1, 2, 3, 5, 10]:
        damaged, mask = corrupt_noise(arr, [3, 4], 1000, 1600, level=level, seed=0)
        print(level, evaluate(damaged, mask))

    damaged, mask = corrupt_flatline(arr, [3, 4], 1000, 1600)
    print(evaluate(damaged, mask))

    damaged, mask = corrupt_jump(arr, [3, 4], 1000, 1600, size=3)
    print(evaluate(damaged, mask))


    var_score, mean_score = window_scores(arr)
    print(f"Var Score: {np.percentile(var_score, [1, 50, 99])}")
    print(f"Mean Score: {np.percentile(mean_score, [1, 50, 99])}")


    # damaged, mask = corrupt_noise(arr, [3, 4], 1000, 1600, level=3, seed=0)
    # diff = get_diff(arr, damaged)
    # print(f"Noise diff: {np.round(diff, 2)}")

    # damaged, mask = corrupt_jump(arr, [3, 4], 1000, 1600, size=3)
    # diff = get_diff(arr, damaged)
    # print(f"Jump diff: {np.round(diff, 2)}")

    # clean_mean = window_mean(arr)
    # print(arr[3].mean())                 # mean of the whole channel
    # print(np.median(clean_mean[3]))      # typical window mean
    # print(clean_mean[3, 5:8])            # the three windows you damaged

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

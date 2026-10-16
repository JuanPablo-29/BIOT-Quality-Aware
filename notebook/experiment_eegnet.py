from eegnet import EEGNet
from trial import Trial
from corrupt import corrupt_random
import numpy as np
import torch
from detector import detect, window_scores

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy(model, t):
    x = torch.tensor(t.get_trials(), dtype=torch.float32).to(device)
    y = torch.tensor(t.get_labels(), dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item()

def run_conditions(model, file):
    t = Trial(file)
    clean = t.arr
    damaged, mask = corrupt_random("noise", clean, n_segments=12000, level = 5)
    flag = detect(*window_scores(damaged))
    flag = np.repeat(flag, 200, axis=1)
    missing = damaged.shape[1] - flag.shape[1]
    flag = np.pad(flag, ((0,0), (0,missing)), constant_values=False)
    damaged_flag = damaged.copy()
    damaged_flag[flag] = 0
    damaged_mask = damaged.copy()
    damaged_mask[mask] = 0

    results = []
    for arr in [clean, damaged, damaged_flag, damaged_mask]:
        t.arr = arr
        results.append(accuracy(model, t))
    return results




if __name__ == "__main__":
    model = EEGNet(n_classes=4, chans=22, samples=800).to(device)

    all_results = []
    for i in range(1, 10):
        model.load_state_dict(torch.load(f"models/eegnet_A0{i}.pt", map_location=device))
        model.eval()
        all_results.append(run_conditions(model, f"data/A0{i}E"))
    print(np.mean(all_results, axis=0))
    np.save("results/noise5.npy", np.array(all_results))
   

    
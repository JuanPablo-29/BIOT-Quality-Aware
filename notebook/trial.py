import mne
import numpy as np
import scipy
import scipy.io as sio


CH_NAMES = ["Fz",
            "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2",
            "POz"]

class Trial:
    def __init__(self, file):
        raw = mne.io.read_raw_gdf(file + '.gdf', preload = True)
        events, event_id = mne.events_from_annotations(raw)
        raw.set_channel_types({'EOG-left':'eog', 'EOG-right':"eog", 'EOG-central':'eog'})
        raw.pick('eeg')
        self.arr = raw.get_data()
        mask = events[:,2] == event_id['768']
        start = events[mask]
        self.start = start[:,0]
        self.start = np.round(self.start * 4/5).astype(int)

        # print(raw)
        # print(raw.info)
        # print(raw.ch_names)
        # for i in [0, 7, 9, 11, 19]:
        #     print(i, CH_NAMES[i], raw.ch_names[i])

        labels = sio.loadmat(file + '.mat')
        class_labels = labels['classlabel']
        self.class_labels = class_labels[:, 0]
        self.arr = self.resample(self.arr)
        self.sampling_rate = 200
        self.offset = int(2 * self.sampling_rate)
        self.window = int(4 * self.sampling_rate)

        print(len(self.start))
        print(len(self.class_labels))


    def resample(self, a):
        new_arr = scipy.signal.resample_poly(a, 4, 5, axis=1)
        return new_arr

    def get_trials(self):
        trials = np.zeros((len(self.start), self.arr.shape[0], self.window))
        for i, s in enumerate(self.start):
            a = s + self.offset
            x = self.arr[:, a : a + self.window]
            scale = np.quantile(np.abs(x), 0.95, axis=-1, keepdims=True)
            trials[i] = x / scale
        return trials

    def get_labels(self):
        return self.class_labels - 1

if __name__ == "__main__":
    # E_trials = {}
    # T_trials = {}
    # for i in range(9):
    #     E_trials[i+1] = Trial(f"data/A0{i+1}E")
    #     T_trials[i+1] = Trial(f"data/A0{i+1}T")
    #print(E_trials[1].get_trials().shape)

    t = Trial("data/A01T")
    x = t.get_trials()
    y = t.get_labels()
    print(x.shape, x.min(), x.max())
    print(np.unique(y))
    # encoded = t.get_encoded_labels()
    # print(t.class_labels[:5])
    # print(encoded[:5])

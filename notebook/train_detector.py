import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from trial import Trial
from detector import window_features, window_labels
from corrupt import corrupt_random
# import mne, warnings
# mne.set_log_level("ERROR")
# warnings.filterwarnings("ignore")

def make_table(clean, kind, level=5, size=5):
    damaged, mask = corrupt_random(kind, clean, n_segments=3000, size=size, level=level)
    feat = window_features(damaged).reshape(-1,4)
    key = window_labels(mask).reshape(-1)
    return feat, key

def build(subjects, session, level=5, size=5, kinds = ("noise", "flatline", "jump")):
    Xs, ys = [], []
    for i in subjects:
        clean = Trial(f"data/A0{i}{session}").arr
        for k in kinds:
            x, y = make_table(clean, k, level=level, size=size)
            Xs.append(x)
            ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)

if __name__ == "__main__":

    parts = [build(range(1, 7), "T", level=1, size=s) for l, s in [(3,2),(5,5),(8,8)]]

    X_train = np.concatenate([x for x, _ in parts])
    y_train = np.concatenate([y for _, y in parts])
    #X_test, y_test = build(range(7,10), "E", level=3, size=2)
    # print(X_train.shape, y_train.shape, y_train.mean())
    #print(X_test.shape, y_test.shape, y_test.mean())
    
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(X_train, y_train)


    # Xk, yk = build(range(7, 10), "E", level=3, size=2, kinds=("noise",))
    # pk = model.predict_proba(Xk)[:, 1]
    # top = np.argsort(pk[~yk])[-5:]
    # print("\n=== TOP 5 CLEAN WINDOWS THE MODEL CALLS BAD ===")
    # print(Xk[~yk][top])

    rows = []
    for k in ("noise", "flatline", "jump"):
        Xk, yk = build(range(7, 10), "E", level=3, size=2, kinds=(k,))
        pk = model.predict_proba(Xk)[:, 1]
        cut = np.quantile(pk[~yk], 0.988)      # cutoff that gives 1.2% false alarms
        pred = pk > cut
        rows.append((k, cut, pred[yk].mean(), pred[~yk].mean()))

    m = (X_train[:, 0] > 7) & (X_train[:, 0] < 14)    # windows with variance score near 10
    print("\n=== TRAIN WINDOWS WITH VARIANCE SCORE 7 TO 14 ===")
    print("count:", m.sum(), " share that are bad:", y_train[m].mean())

    print("\n=== MODEL AT 1.2% FALSE ALARMS (kind, cutoff, recall, false alarm) ===")
    for k, cut, r, fa in rows:
        print(f"{k:9s} {cut:.3f} {r:.3f} {fa:.4f}")
        # print(k, pk[yk].mean(), pk[~pk].mean())
        # rule = (Xk[:, 0] < 0.1) | (Xk[:, 0] > 9) | (Xk[:, 1] > 2)
        # print(k, rule[yk].mean(), rule[~yk].mean())

    
    # p = model.predict_proba(X_test)[:, 1]
    # print(p[:5])

    # pred = p > 0.5
    # recall = pred[y_test].mean()
    # false_alarm = pred[~y_test].mean()
    # print(recall, false_alarm)
    
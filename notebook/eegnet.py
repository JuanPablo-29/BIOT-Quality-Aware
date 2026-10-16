import torch
import torch.nn as nn
from trial import Trial

class EEGNet(nn.Module):
    def __init__(self, n_classes, chans, samples, kern=100, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classify = nn.Linear(F2 * (samples // 32), n_classes)

    def forward(self, x):                 # x: (batch, channels, samples)
        x = x[:, None]                    # -> (batch, 1, channels, samples)
        x = self.block2(self.block1(x))
        return self.classify(x.flatten(1))


def train_subject(i):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    
    t = Trial(f"data/A0{i}T")

    t_trials = t.get_trials()
    t_labels = t.get_labels()

    x_train, y_train = t_trials, t_labels

    xtr = torch.tensor(x_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)

    model = EEGNet(n_classes=4, chans=22, samples=800).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(60):
        # 3. train for one epoch
        model.train()
        perm = torch.randperm(len(x_train))            # new shuffle every epoch
        total_loss = 0
        for b in range(0, len(x_train), 32):
            idx = perm[b:b + 32]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            out = model(xb)                        # (batch, 4) scores
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        print(epoch, round(total_loss / len(x_train), 4))
    torch.save(model.state_dict(), f"models/eegnet_A0{i}.pt")
    

if __name__ == "__main__":
    for i in range(1, 10):
        train_subject(i)
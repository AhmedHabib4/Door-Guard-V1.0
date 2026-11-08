import torch, torch.nn as nn
from torchvision import datasets, transforms, models

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = transforms.Compose([
    transforms.Resize((160, 160)), transforms.ColorJitter(0.4, 0.4, 0.3),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
    transforms.ToTensor(), norm
])
tf_eval = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor(), norm])

train = datasets.ImageFolder("../data/vehicles/train", tf_train)
val = datasets.ImageFolder("../data/vehicles/val", tf_eval)
tl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
vl = torch.utils.data.DataLoader(val, batch_size=64)

m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
for p in m.parameters():
    p.requires_grad = False
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(train.classes)) # new head

def evaluate():
    m.eval(); ok = 0; n = 0
    with torch.no_grad():
        for x, y in vl:
            ok += (m(x).argmax(1) == y).sum().item()
            n += len(y)
    return ok / n

def fit(params, lr, epochs):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    for e in range(epochs):
        m.train()
        for x, y in tl:
            loss = nn.functional.cross_entropy(m(x), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        print("epoch", e, "val acc", round(evaluate(), 3))

# phase 1: only the new head
fit(m.classifier[3].parameters(), 1e-3, 6)
# phase 2: unfreeze the last blocks
for p in m.features[-3:].parameters():
    p.requires_grad = True
fit([p for p in m.parameters() if p.requires_grad], 1e-4, 6)

torch.save({"model": m.state_dict(), "classes": train.classes}, "../models/vehicle_v1.pt")
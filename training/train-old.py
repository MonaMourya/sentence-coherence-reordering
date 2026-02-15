import torch
import torch.nn as nn
from tqdm import tqdm

from models.sentence_encoder import SentenceEncoder
from models.coherence_model import CoherenceModel
from training.dataset_builder import build_dataset

device = "cpu"

encoder = SentenceEncoder()
model = CoherenceModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

# LOAD TRAINING DATA
with open("data/training data.txt", "r", encoding="utf-8") as f:
    paragraphs = f.readlines()

pairs = []
labels = []

for para in paragraphs:
    result = build_dataset(para)
    if result is None:
        continue

    coherent, incoherent = result

    emb_coh = encoder.encode(coherent)
    emb_inc = encoder.encode(incoherent)

    n = len(coherent)

    for i in range(n):
        for j in range(i + 1, n):
            # coherent
            pairs.append(torch.cat([emb_coh[i], emb_coh[j]]))
            labels.append(1)

            pairs.append(torch.cat([emb_coh[j], emb_coh[i]]))
            labels.append(0)

            # incoherent
            pairs.append(torch.cat([emb_inc[i], emb_inc[j]]))
            labels.append(0)

            pairs.append(torch.cat([emb_inc[j], emb_inc[i]]))
            labels.append(1)

# TRAIN
model.train()
for epoch in range(5):
    total_loss = 0
    for x, y in tqdm(zip(pairs, labels), total=len(labels)):
        optimizer.zero_grad()
        pred = model(x.unsqueeze(0))
        loss = loss_fn(pred.squeeze(), torch.tensor(float(y)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/trained/coherence_model.pt")
print("✅ Model saved")
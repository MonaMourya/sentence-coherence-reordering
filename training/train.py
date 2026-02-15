import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset

from models.sentence_encoder import SentenceEncoder
from models.coherence_model import CoherenceModel
from training.dataset_builder import build_dataset

# -----------------------
# CONFIGURATION
# -----------------------
DEVICE = "cpu"
MAX_SAMPLES = 1000        # enough for good accuracy, CPU-safe
EPOCHS = 5
LEARNING_RATE = 1e-4

# -----------------------
# INITIALIZE MODELS
# -----------------------
encoder = SentenceEncoder()
model = CoherenceModel().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss()

# -----------------------
# LOAD ROCSTORIES DATASET
# -----------------------
print("📥 Loading ROCStories dataset...")
dataset = load_dataset("mintujupally/ROCStories", split="train")

print("✅ Dataset loaded")

# Testing
# print(dataset[0])
# exit()

# -----------------------
# BUILD TRAINING PAIRS
# -----------------------
pairs = []
labels = []

print("🧠 Building training samples...")

for idx in tqdm(range(min(len(dataset), MAX_SAMPLES))):
    sample = dataset[idx]

    # SAFELY extract text
    if "text" in sample:
        story_text = sample["text"]
    else:
        continue  # skip unknown formats

    result = build_dataset(story_text)
    if result is None:
        continue

    coherent, incoherent = result

    emb_coh = encoder.encode(coherent)
    emb_inc = encoder.encode(incoherent)

    n = len(coherent)

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(torch.cat([emb_coh[i], emb_coh[j]]))
            labels.append(1)

            pairs.append(torch.cat([emb_coh[j], emb_coh[i]]))
            labels.append(0)

            pairs.append(torch.cat([emb_inc[i], emb_inc[j]]))
            labels.append(0)

            pairs.append(torch.cat([emb_inc[j], emb_inc[i]]))
            labels.append(1)

print(f"✅ Total training pairs: {len(pairs)}")

# -----------------------
# TRAIN MODEL
# -----------------------
print("🚀 Training started...\n")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in tqdm(zip(pairs, labels), total=len(labels)):
        optimizer.zero_grad()
        pred = model(x.unsqueeze(0))
        loss = loss_fn(pred.squeeze(), torch.tensor(float(y)))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f}")

# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "models/trained/coherence_model.pt")
print("\n✅ Model saved to models/trained/coherence_model.pt")
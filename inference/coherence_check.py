import torch

def compute_probability_matrix(sentences, encoder, model):
    embeddings = encoder.encode(sentences)
    n = len(sentences)

    prob = [[0.0] * n for _ in range(n)]

    model.eval()
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                if i != j:
                    x = torch.cat([embeddings[i], embeddings[j]])
                    prob[i][j] = model(x.unsqueeze(0)).item()

    return prob
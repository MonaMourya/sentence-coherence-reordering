def reorder_sentences(prob_matrix):
    """
    Robust sentence reordering using global scores.
    Avoids graph cycles.
    """
    n = len(prob_matrix)
    scores = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += prob_matrix[i][j]

    # Higher score = should come earlier
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return order
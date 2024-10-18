import torch.nn as nn

def compute_perplexity(loss):
    return torch.exp(loss)

# During evaluation
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in valid_iterator:
        inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
        outputs, hidden = model(inputs, None)
        loss = criterion(outputs, targets.reshape(-1))
        total_loss += loss.item()

perplexity = compute_perplexity(total_loss / len(valid_iterator))
print(f'Validation Perplexity: {perplexity}')

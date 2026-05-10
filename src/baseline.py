from tqdm import tqdm
import matplotlib.pyplot as plt

from data.load import load_training_set
from data.features import build_relevance_scores

from evaluate import compute_loss
from predict import random

df_train = load_training_set()

df_train = build_relevance_scores(df_train)

losses = []
for _ in tqdm(range(1000), desc="Evaluation"):
    df_eval = random(df_train)
    loss = compute_loss(df_eval)
    losses.append(loss)

avg_loss = sum(losses) / len(losses)
print(f"Average Loss: {avg_loss}")

plt.figure(figsize=(10, 6))
plt.hist(losses, bins=20, alpha=0.7, edgecolor='black', label='Loss frequency')
plt.axvline(x=avg_loss, color='r', linestyle='--', label=f'Average: {avg_loss:.4f}')
plt.xlabel('Loss value')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Frequency Distribution')
plt.savefig('loss_plot.png')
plt.show()

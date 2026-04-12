# scripts/plot_training.py
import pandas as pd
import matplotlib.pyplot as plt
import os

log = pd.read_csv('experiments/logs/train_log.csv')
os.makedirs('experiments/plots', exist_ok=True)

plt.figure(figsize=(6,4))
plt.plot(log['epoch'], log['train_loss'], label='train_loss')
plt.plot(log['epoch'], log['val_loss'], label='val_loss')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
plt.savefig('experiments/plots/loss.png', dpi=150)

plt.figure(figsize=(6,4))
plt.plot(log['epoch'], log['train_acc'], label='train_acc')
plt.plot(log['epoch'], log['val_acc'], label='val_acc')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
plt.savefig('experiments/plots/acc.png', dpi=150)

print('Saved plots to experiments/plots/')

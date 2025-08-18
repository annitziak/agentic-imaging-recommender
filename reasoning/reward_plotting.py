import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
trainer_state_json_path = ""  # Path to the your trainer state JSON file
window = 25  # Smoothing window for rolling mean

# Load the trainer state from JSON of trainer logs
with open(trainer_state_json_path, "r", encoding="utf-8") as f:
    trainer_state = json.load(f)

log_history = trainer_state['log_history']
df = pd.DataFrame(log_history)  # Convert to DataFrame for easier manipulation

# Epochs are logged every 100 steps
df["epoch"] = df['step'] / 100

# Plotting the rewards
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(14, 6))


# pay attention to the labels in the legend, change with different reward components
plt.plot(df['step'], df['reward'].rolling(window).mean(), label='Total Reward', color="#4C72B0")
if 'rewards/format_reward/mean' in df.columns:
    plt.plot(df['step'], df['rewards/format_reward/mean'].rolling(window).mean(),
             label='Format Reward', color="#C44E52")
if 'rewards/correctness_answer/mean' in df.columns:
    plt.plot(df['step'], df['rewards/correctness_answer/mean'].rolling(window).mean(),
             label='Answer Reward', color="#55A868")
    
# Uncomment if you have additional reward components   (the lambda is always the additional reward component) 
#if 'rewards/<lambda>/mean' in df.columns:
#    plt.plot(df['step'], df['rewards/<lambda>/mean'].rolling(window).mean(),
#             label='LLM Reward', color="#8172B2")

# Plotting
min_epoch = df["epoch"].min()
max_epoch = df["epoch"].max()
epoch_ticks = np.arange(np.floor(min_epoch), np.ceil(max_epoch) + 0.5, 0.5)
step_ticks = [df.iloc[(df["epoch"] - etick).abs().argsort()[:1]]["step"].values[0] for etick in epoch_ticks]
plt.xticks(step_ticks, [f"{etick:.1f}" for etick in epoch_ticks])

plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.legend()
plt.title('Smoothed Reward Components Over Training')
plt.tight_layout()
#plt.savefig("figures/reward_components.png", bbox_inches='tight')
plt.show()

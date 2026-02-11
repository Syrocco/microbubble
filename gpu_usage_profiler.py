import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# Adjust filename to match your JobID
df = pd.read_csv('gpu_usage_115636.csv', skipinitialspace=True)

# Clean column names (nvidia-smi sometimes adds leading spaces)
df.columns = [c.strip() for c in df.columns]

# Convert timestamp to a readable format or just use index for "Time Steps"
df['seconds'] = range(0, len(df) * 5, 5)  # Because we used '-l 5' in nvidia-smi

plt.figure(figsize=(10, 5))

# Plot GPU Utilization
plt.plot(df['seconds'], df['utilization.gpu [%]'], label='GPU Util (%)', color='blue')

# Plot Memory Utilization
plt.plot(df['seconds'], df['utilization.memory [%]'], label='VRAM Util (%)', color='green', linestyle='--')

plt.title('GPU Performance Profile (Job 115636)')
plt.xlabel('Time (seconds)')
plt.ylabel('Percentage (%)')
plt.legend()
plt.grid(True)

plt.savefig('gpu_profile.png')
plt.show()  
import matplotlib.pyplot as plt
import numpy as np

# Data from your runs
models = ['LogReg', 'DT', 'XGB']
acs_f1 = [0.7136, 0.7092, 0.7398]
nsl_f1 = [0.7435, 0.7606, 0.7858]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, acs_f1, width, label='ACS Income (OOD)')
bars2 = ax.bar(x + width/2, nsl_f1, width, label='NSL-KDD (OOD)')

ax.set_ylabel('F1-score')
ax.set_title('Baseline Model Performance on OOD Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/baseline_f1_comparison.png')
plt.show()

print("Plot saved to results/baseline_f1_comparison.png")
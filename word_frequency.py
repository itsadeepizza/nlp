from loader import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Skipping words  double the execution time
THREESHOLD = 5e-4
BASE_SKIP = 1.2 # Increase for skipping fewer words (default is 1.2)
OFFSET_SKIP = 8 # Increase for skipping fewer words  (default is 8)
wc = train.word_frequency
freqs = np.array(list(wc.values()))
ratio = np.clip(THREESHOLD / freqs * BASE_SKIP**(np.log(freqs)+OFFSET_SKIP), 0, 1)
ratio[freqs < THREESHOLD] = 1

skip_prob = np.sum((1 - ratio) * freqs)
new_freqs = ratio * freqs
new_freqs = new_freqs / new_freqs.sum()
fig, ax = plt.subplots(2, 1, figsize=(5,9))

sns.ecdfplot(freqs, ax=ax[0])
sns.ecdfplot(new_freqs, ax=ax[0])
ax[0].set_xscale('log')
ax[0].grid()
ax[1].set_xlabel("Probability to be chosen")
print(f"Probability to skip a word is {skip_prob*100:.1f}%. Number of iteration are "
      f"increased of {1/(1-skip_prob)*100 - 100:.0f}%")


ax[1].plot(np.linspace(0, 1, len(freqs))*100, np.cumsum(freqs[::-1])*100, label="base")
ax[1].plot(np.linspace(0, 1, len(new_freqs))*100, np.cumsum(new_freqs[::-1])*100, label="skip words")
ax[1].set_xlabel("Ratio of words (%)")
ax[1].set_ylabel("Probability to be chosen (%)")
ax[1].grid()
ax[1].legend()
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
fig.show()


from loader import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Skipping words  double the execution time
THREESHOLD = 3e-5 # Never skip below this threeshold
BASE_SKIP = 1.9 # Increase for skipping more words (default is 1.2, always greater than 1)
OFFSET_SKIP = 0.025 # Decrease for skipping more words  (default is 0.00358)
wc = train.word_frequency
freqs = np.array(list(wc.values()))
ratio = np.clip(OFFSET_SKIP / freqs * BASE_SKIP**(np.log(freqs)), 0, 1)
ratio[freqs < THREESHOLD] = 1

skip_prob = np.sum((1 - ratio) * freqs)
new_freqs = ratio * freqs
new_freqs = new_freqs / new_freqs.sum()
fig, ax = plt.subplots(3, 1, figsize=(5,13))

old_prob = np.exp(np.linspace(-5.5,-2,100)*np.log(10))
ratio_prob = np.clip(OFFSET_SKIP / old_prob * BASE_SKIP**(np.log(old_prob)), 0, 1)
ratio_prob[old_prob < THREESHOLD] = 1
new_prob = old_prob * ratio_prob / (1 - skip_prob)
ax[0].plot(old_prob, new_prob)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel("Old probability")
ax[0].set_ylabel("New probability")
ax[0].set_title("Probability variation after skipping")
ax[0].grid()

sns.ecdfplot(freqs, ax=ax[1], complementary=False)
sns.ecdfplot(new_freqs, ax=ax[1], complementary=False)
ax[1].set_xscale('log')
# ax[1].set_yscale('log')
ax[1].grid()
ax[1].set_xlabel("Probability to be chosen")
print(f"Probability to skip a word is {skip_prob*100:.1f}%. Number of iteration are "
      f"increased of {1/(1-skip_prob)*100 - 100:.0f}%")

ax[2].plot(np.linspace(0, 1, len(freqs))*100, np.cumsum(freqs[::-1])*100, label="base")
ax[2].plot(np.linspace(0, 1, len(new_freqs))*100, np.cumsum(new_freqs[::-1])*100, label="skip "
                                                                                        "words")
ax[2].set_xlabel("Ratio of words (%)")
ax[2].set_ylabel("Probability to be chosen (%)")
ax[2].grid()
ax[2].legend()
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
fig.show()


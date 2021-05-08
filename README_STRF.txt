STRF notebook can run perception, production, and visread.

Each model has a penalty of 1,000,000

For the training data:
X is the (denoised) spectrogram, each feature (frequency band) is zscored
If perc, pad over production with average (of -500ms to 700ms)
If prod, pad before production with average (of 0ms to 110ms)
If visread, no pad.

y is the ecog, each trial is convolved (smoothed) and zscored

data.mat is a dictionary where:
the keys are labels 
the values are a tuple of (scores, kernel data)
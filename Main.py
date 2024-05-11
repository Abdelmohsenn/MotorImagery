import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load data for Subject 1
signals1 = pd.read_csv('Subject1_Signals.csv')
labels1 = pd.read_csv('Subject1_Labels.csv')
trials1 = pd.read_csv('Subject1_Trial.csv')

signals2 = pd.read_csv('Subject2_Signals.csv')
labels2 = pd.read_csv('Subject2_Labels.csv')
trials2 = pd.read_csv('Subject2_Trial.csv')

signals3 = pd.read_csv('Subject3_Signals.csv')
labels3 = pd.read_csv('Subject3_Labels.csv')
trials3 = pd.read_csv('Subject3_Trial.csv')

# # Load the labels data
# labels_df = pd.read_csv('Subject1_Labels.csv', header=None, names=['Label'])

# # Load the trial data
# trial_df = pd.read_csv('SubjectX_Trial.csv', header=None, names=['Trial'])

# Extract the signal data for each electrode

#electrode_data = signals.to_string(index=False, max_rows=None, max_cols=None) # This will give you a numpy array of shape (number of samples, number of electrodes)

electrode_data = signals1.shape[0]

signals = pd.read_csv('Subject1_Signals.csv')

# # Iterate over each column (electrode) and print the data
# for col in signals.columns[3]:
#     #print(f"Electrode {col+1} Signals:")
#     print(signals[col].to_string(index=False))  # Print the signals in the column without index
#     print("\n")  # Add a newline for better readability
    
    
    
plt.figure(figsize=(12, 6))
for i in range(3):
    # Before CAR
    plt.subplot(3, 2, 2*i+1)
    f, Pxx = welch(signals.iloc[:, i], fs=512, nperseg=1024)
    plt.plot(f, 10*np.log10(Pxx))
    plt.title(f'Channel {i+1} Spectrum (Before CAR)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)

plt.tight_layout()
plt.show()

#print(electrode_data)




import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data for all subjects
# def readsubject1():
#     signals1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Signals.csv')
#     labels1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Labelscsv')
#     trials1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Trials.csv')
#     return signals1, labels1, trials1
# def readsubject2():
#     signals2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Signals.csv')
#     labels2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Labelscsv')
#     trials2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Trials.csv')
#     return signals2, labels2, trials2
# def readsubject3():
#     signals3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Signals.csv')
#     labels3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Labelscsv')
#     trials3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Trials.csv')
#     return signals3, labels3, trials3


def readsubject1():
    signals1 = pd.read_csv('Subject1_Signals.csv',header=None)
    labels1 = pd.read_csv('Subject1_Labels.csv', header=None)
    trials1 = pd.read_csv('Subject1_Trial.csv', header=None)
    return signals1, labels1, trials1
def readsubject2():
    signals2 = pd.read_csv('Subject2_Signals.csv',header=None)
    labels2 = pd.read_csv('Subject2_Labels.csv',header=None)
    trials2 = pd.read_csv('Subject2_Trial.csv',header=None)
    return signals2, labels2, trials2
def readsubject3():
    signals3 = pd.read_csv('Subject3_Signals.csv',header=None)
    labels3 = pd.read_csv('Subject3_Labels.csv', header=None)
    trials3 = pd.read_csv('Subject3_Trial.csv',header=None)
    return signals3, labels3, trials3

signals1,labels1,trials1= readsubject1()
signals2,labels2,trials2=readsubject2()
signals3,labels3,trials3=readsubject3()




Fs = 512
step = 1/Fs
#electrode_data = signals.to_string(index=False, max_rows=None, max_cols=None) # This will give you a numpy array of shape (number of samples, number of electrodes)
electrode_data = signals1.to_string(index=False, max_rows=None, max_cols=None)
# electrode_data2 = signals2.to_string(index=False, max_rows=None, max_cols=None)
# electrode_data3 = signals3.to_string(index=False, max_rows=None, max_cols=None)


# CAR filters
def ComputeCARfilterSignal1():
    average_signal = signals1.mean(axis=1)
    CommonAvFiltered = signals1.sub(average_signal, axis=0)
    return CommonAvFiltered

def ComputeCARfilterSignal2():
    average_signal = signals2.mean(axis=1)
    CommonAvFiltered = signals2.sub(average_signal, axis=0)
    return CommonAvFiltered

def ComputeCARfiltersignal3():  
    average_signal = signals3.mean(axis=1)
    CommonAvFiltered = signals3.sub(average_signal, axis=0)
    return CommonAvFiltered



#the filtered data and the non filtered comparing func (3 channels only)
def comparing(filtered, nonfiltered):
    plt.figure(figsize=(10, 6))
    for i in range(min(3, filtered.shape[1])):  
        plt.subplot(3, 1, i + 1)
        plt.plot(filtered.index, filtered.iloc[:, i])
        plt.title(f'Filtered Signal of Channel {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(min(3, nonfiltered.shape[1])):  
        plt.subplot(3, 1, i + 1)
        plt.plot(nonfiltered.index, nonfiltered.iloc[:, i])
        plt.title(f'Original Signal of Channel {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

# Example usage

def bandpassFilter(signal,lowcut,highcut,order):
    nyq=0.5*Fs
    low=lowcut/nyq
    high=highcut/nyq
    sig1,sig2 = butter(order,[low,high],btype='band')
    sig=filtfilt(sig1,sig2,signal)
    return sig

def CARandNon_Spectrum3channelsONLY():
    plt.figure(figsize=(10, 10))
    for i in range(1):

        signal_col = signals1.iloc[:, i]  
        n = len(signals1)
        fourier = np.fft.fftfreq(n, step)
        spectrum = np.fft.fft(signal_col)
        

        plt.subplot(6, 1, i + 1)
        plt.plot(fourier[:n // 2], np.abs(spectrum)[:n // 2]) 
        plt.title(f'Spectrum of Channel {i + 1} before CAR')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
       ## plt.xlim(50, 200)  # Adjust the limits as needed
        
    # CAR spectrum of 1st 3 channels
        signal_col_filtered = ComputeCARfilterSignal1().iloc[:, i]
        n = len(signal_col_filtered)
        fourier_filtered = np.fft.fftfreq(n, step)
        spectrum_filtered = np.fft.fft(signal_col_filtered)
        
        plt.subplot(6, 1, i + 4)
        plt.plot(fourier_filtered[:n // 2], np.abs(spectrum_filtered)[:n // 2])
        plt.title(f'Spectrum of CAR Filtered subject 2 Channel {i + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        ##plt.xlim(50, 200)  # Adjust the limits as needed

    plt.tight_layout()
    plt.show()
    

    
#comparing(ComputeCARfilterSignal2(), signals2)

# comparing(ComputeCARfilterSignal1(), signals1)
# comparing(ComputeCARfiltersignal3(), signals3)

##CARandNon_Spectrum3channelsONLY()

# def muband():
#     mubandexample = bandpassFilter(signals1.iloc[:,1],8,13,5)
# def betaband():
#     betabandexample=bandpassFilter(signals1.iloc[:,1],13,30,5)
    
np.set_printoptions(threshold=np.inf)
def bandpass_signals(sig):
    mu_band_signals = []
    beta_band_signals = []
    for i in range(sig.shape[1]):  # Iterate over electrodes
        signalcol = sig.iloc[:, i]
        muBand_signal = bandpassFilter(signalcol, 8, 13, 5)
        betaBand_signal = bandpassFilter(signalcol, 13, 30, 5)
        mu_band_signals.append(muBand_signal)
        beta_band_signals.append(betaBand_signal)
    return mu_band_signals, beta_band_signals

mu_band_signals, beta_band_signals = bandpass_signals(signals1)

mu_band_signalsARR = np.array(mu_band_signals)
beta_band_signalsARR= np.array(beta_band_signals)

labels1arr = np.array(labels1)
labels2arr= np.array(labels2)
labels3arr = np.array(labels3)

Trials1Arr = np.array(trials1)
Trials2Arr = np.array(trials2)
Trials3Arr = np.array(trials3)

# for electrode, signal in mu_band_signals.items():
#     print(f"Mu for {electrode}: {signal}")
#     print() 

# for electrode, signal in beta_band_signals.items():
#     print(f"Beta for {electrode}: {signal}")
#     print() 
    
#print(signals1.index) 

# Choose the electrode index to plot
electrodeidx = 7

# # Plot original and filtered signals for the chosen electrode
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(signals1.index, signals1.iloc[:, electrodeidx], label='Original Signal')
# plt.title(f'Original Signal - Electrode {electrodeidx + 1}')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(signals1.index, mu_band_signals[f'Electrode{electrodeidx + 1}'], label='Mu Band Filtered Signal', color='orange')
# plt.plot(signals1.index, beta_band_signals[f'Electrode{electrodeidx + 1}'], label='Beta Band Filtered Signal', color='green')
# plt.title(f'Filtered Signals - Electrode {electrodeidx + 1}')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.tight_layout()
# plt.show()

# mubandexample = muband()
# betabandexample = betaband()

# plt.figure(figsize=(10,10))
# plt.subplot(6,1,1)
# plt.plot(mu_band_signals[f'Electrode{electrodeidx + 1}'])
# plt.title("Muband")
# plt.xlabel("time")
# plt.ylabel("Signals")

# plt.subplot(6,1,3)
# plt.plot(signals1.iloc[:, electrodeidx])
# plt.title("ORIGINAL Signal")
# plt.xlabel("time")
# plt.ylabel("Signals")

# plt.subplot(6,1,5)
# plt.plot(beta_band_signals[f'Electrode{electrodeidx + 1}'])
# plt.title("Beta band")
# plt.xlabel("time")
# plt.ylabel("Signals")

# plt.tight_layout()
# plt.show()

# Example function to find relative change in Mu band power for each trial
def rel_changes(sigs, trial_starts):
    relchanges={}
    for electrode, signal in sigs.items():
        relative_changes = []
        for start in trial_starts:
            startSample = int(float(start)) ##startof the trial
            
            trial_data = signal[startSample:startSample + Fs * 5] ##fisrt 5 secs
            preonset_data = signal[startSample - Fs * 5:startSample] #bef trial 5 sec

            power_trial = np.mean(trial_data ** 2)
            power_pre_onset = np.mean(preonset_data ** 2)

            relative_change = (power_trial - power_pre_onset) / power_pre_onset ##rel change
            relative_changes.append(relative_change)

        relchanges[electrode] = relative_changes

    return relchanges

# Example usage
# relativeChanges_mu = rel_changes(mu_band_signals, trials1)
# relativeChanges_beta = rel_changes(beta_band_signals, trials1)

# print("Relative changes in Mu band power for each electrode and trial:", relativeChanges_mu)
# print("Relative changes in Mu band power for each electrode and trial:", relativeChanges_beta)

# Plot the accuracies
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), mu_band_acc, label='Mu Band')
# plt.plot(range(1, 11), beta_band_acc, label='Beta Band')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.title('KNN Classifier Accuracy for Mu and Beta Bands')
# plt.xticks(range(1, 11))
# plt.legend()
# plt.show()


print(mu_band_signalsARR.shape, beta_band_signalsARR.shape, labels1arr.shape)

#KNN CLASSIFIER

X_train_mu, X_test_mu, y_train, y_test = train_test_split(mu_band_signalsARR, labels1arr.ravel()[:15], test_size=0.2, random_state=42)
X_train_beta, X_test_beta, _, _ = train_test_split(beta_band_signalsARR, labels1arr.ravel()[:15], test_size=0.2, random_state=42)

# Initialize lists to store accuracy for Mu and Beta bands
mu_band_acc = []
beta_band_acc = []

##Loop over values of K from 1 to 10
for k in range(1, 11):
    
    # Initialize KNN classifier with K neighbors for Mu band
    knn_mu = KNeighborsClassifier(n_neighbors=k)
    knn_mu.fit(X_train_mu, y_train)
    mu_band_acc.append(knn_mu.score(X_test_mu, y_test))

    # Initialize KNN classifier with K neighbors for Beta band
    knn_beta = KNeighborsClassifier(n_neighbors=k)
    knn_beta.fit(X_train_beta, y_train)
    beta_band_acc.append(knn_beta.score(X_test_beta, y_test))
    

print (mu_band_acc)
print (beta_band_acc)


# def KnnErrors(mu, beta, y, k_values=range(1, 11), n_splits=10):
#     MuErrors = []
#     BetaErros = []
    
#     y=y.ravel()
    
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     for k in k_values:
#         FoldErrorsMu = []
#         FoldErrorsBeta = []
        
#         ##Splitting train and testing data
#         for train_index, test_index in kf.split(mu):
#             X_train_mu, X_test_mu = mu[train_index], mu[test_index]
#             X_train_beta, X_test_beta = beta[train_index], beta[test_index]
#             y_train, y_test = y[train_index], y[test_index]  

#         ##KNN FOR BETA
#             KnnMu = KNeighborsClassifier(n_neighbors=k)
#             KnnMu.fit(X_train_mu, y_train)
#             ypredmu = KnnMu.predict(X_test_mu)
#             FoldErrorsMu.append(1 - accuracy_score(y_test, ypredmu))

#         ##KNN FOR BETA
#             KnnBeta = KNeighborsClassifier(n_neighbors=k)
#             KnnBeta.fit(X_train_beta, y_train)
#             ypredbeta = KnnBeta.predict(X_test_beta)
#             FoldErrorsBeta.append(1 - accuracy_score(y_test, ypredbeta))

#         MuErrors.append(np.mean(FoldErrorsMu))
#         BetaErros.append(np.mean(FoldErrorsBeta))

#     classifErrors = {
#         'Mu band': MuErrors,
#         'Beta band': BetaErros
#     }

#     return classifErrors

# ClassifErr = KnnErrors(mu_band_signalsARR, beta_band_signalsARR, labels1arr)
# print(ClassifErr)


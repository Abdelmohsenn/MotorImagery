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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

# Load data for all subjects
def readsubject1():
    signals1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Signals.csv')
    labels1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Labels.csv', header=None)
    trials1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Trial.csv')
    return signals1, labels1, trials1
def readsubject2():
    signals2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Signals.csv')
    labels2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Labels.csv', header=None)
    trials2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Trial.csv')
    return signals2, labels2, trials2
def readsubject3():
    signals3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Signals.csv')
    labels3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Labels.csv', header=None)
    trials3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Trial.csv')
    return signals3, labels3, trials3


# def readsubject1():
#     signals1 = pd.read_csv('Subject1_Signals.csv')
#     labels1 = pd.read_csv('Subject1_Labels.csv',header=None)
#     trials1 = pd.read_csv('Subject1_Trial.csv')
#     return signals1, labels1, trials1
# def readsubject2():
#     signals2 = pd.read_csv('Subject2_Signals.csv')
#     labels2 = pd.read_csv('Subject2_Labels.csv',header = None)
#     trials2 = pd.read_csv('Subject2_Trial.csv')
#     return signals2, labels2, trials2
# def readsubject3():
#     signals3 = pd.read_csv('Subject3_Signals.csv')
#     labels3 = pd.read_csv('Subject3_Labels.csv' ,header = None)
#     trials3 = pd.read_csv('Subject3_Trial.csv')
#     return signals3, labels3, trials3

signals1,labels1,trials1= readsubject1()
signals2,labels2,trials2=readsubject2()
signals3,labels3,trials3=readsubject3()
allLeasterrors = float
Fs = 512
step = 1/Fs

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

#change the signals
signals1=ComputeCARfilterSignal1()
signals2=ComputeCARfilterSignal1()
signals3=ComputeCARfilterSignal1()

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


def CARandNon_Spectrum3channelsONLY():
    plt.figure(figsize=(10, 10))
    for i in range(3):

        signal_col = signals1.iloc[:, i]  
        n = len(signals1)
        fourier = np.fft.fftfreq(n, step)
        spectrum = np.fft.fft(signal_col)
        

        plt.subplot(6, 1, i + 1)
        plt.plot(fourier[:n // 2], np.abs(spectrum)[:n // 2]) 
        plt.title(f'Spectrum of Channel {i + 1} before CAR')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        
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

    plt.tight_layout()
    plt.show()
    

def bandpassFilter(signal, lowcut, highcut, order):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    sig1, sig2 = butter(order, [low, high], btype='band')
    sig = filtfilt(sig1, sig2, signal)
    return sig

def bandpass_signals(sig):
    mu_band_signals = {}
    beta_band_signals = {}
    for i in range(sig.shape[1]):  # Iterate over (electrodes)
        signalcol = sig.iloc[:, i]
        muBand_signal = bandpassFilter(signalcol, 8, 13, 5)
        betaBand_signal = bandpassFilter(signalcol, 13, 30, 5)
        mu_band_signals[f'Electrode{i+1}'] = muBand_signal
        beta_band_signals[f'Electrode{i+1}'] = betaBand_signal
    return mu_band_signals, beta_band_signals

# Compute relative changes
def rel_changes(signal, trial_starts):
    relative_changes = []
    for start in trial_starts:
        startSample = int(float(start))
        trial_data = signal[startSample:startSample + Fs * 5]
        preonset_data = signal[startSample - Fs * 5:startSample]

        power_trial = np.mean(trial_data ** 2)
        power_pre_onset = np.mean(preonset_data ** 2)

        relative_change = (power_trial - power_pre_onset) / power_pre_onset
        relative_changes.append(abs(relative_change))

    return np.array(relative_changes)

def rel_changes_concatenated(sigs, trial_starts, Fs):
    relchanges = {}
    for electrode, signal in sigs.items():
        relative_changes = []
        for start in trial_starts:
            startSample = int(float(start))
            trial_data = signal[startSample:startSample + Fs * 5]
            preonset_data = signal[startSample - Fs * 5:startSample]

            power_trial = np.mean(trial_data ** 2)
            power_pre_onset = np.mean(preonset_data ** 2)

            relative_change = (power_trial - power_pre_onset) / power_pre_onset
            relative_changes.append(abs(relative_change))

        relchanges[electrode] = relative_changes

    df = pd.DataFrame(relchanges)
    return df

# muband = bandpassFilter(signals1.iloc[:,0], 8, 13, 5)
# relchangeofoneelectrode=rel_changes(muband, trials1)

# print(relchangeofoneelectrode)
# KNN with cross_val_predict
def knn_crossval_predict(X, y, cv=10):
    predictions = []

    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        if X.ndim == 1:
            y_pred = cross_val_predict(knn, X.reshape(-1, 1), y, cv=cv)
        else:
            y_pred = cross_val_predict(knn, X, y, cv=cv)
        predictions.append(y_pred)

    return predictions

# Evaluate predictions
def Evaluate10foldErrors(predictions, y):
    errors = []

    for y_pred in predictions:
        accuracy = accuracy_score(y, y_pred)
        error = 1 - accuracy
        errors.append(error)

    return errors


# predictval=knn_crossval_predict(relchangeofoneelectrode,labels1ID)
# errors=Evaluate10foldErrors(predictval,labels1ID)

# print(errors)
def Subjectloop(Signals,Trials,Labels):

    Labels = Labels.values.flatten()
    least_error = float('inf')
    best_electrode = None
    best_band = None
    best_k = None

    for electrode in range(Signals.shape[1]):
        # print(f"Processing Electrode {electrode + 1}")

        mu_band_signal = bandpassFilter(Signals.iloc[:, electrode], 8, 13, 5)
        beta_band_signal = bandpassFilter(Signals.iloc[:, electrode], 13, 30, 5)

        rel_changes_mu = rel_changes(mu_band_signal, Trials)
        rel_changes_beta = rel_changes(beta_band_signal, Trials)

        # print(f"Relative changes for Electrode {electrode + 1} computed")

        mu_predictions = knn_crossval_predict(rel_changes_mu, Labels)
        beta_predictions = knn_crossval_predict(rel_changes_beta, Labels)

        # print(f"KNN predictions for Electrode {electrode + 1} computed")

        mu_errors_cv = Evaluate10foldErrors(mu_predictions, Labels)
        beta_errors_cv = Evaluate10foldErrors(beta_predictions, Labels)

        # print(f"KNN errors for Electrode {electrode + 1} evaluated")

        min_mu_error = min(mu_errors_cv)
        min_beta_error = min(beta_errors_cv)

        if min_mu_error < least_error:
            least_error = min_mu_error
            best_electrode = electrode + 1
            best_band = 'Mu'
            best_k = mu_errors_cv.index(min_mu_error) + 1

        if min_beta_error < least_error:
            least_error = min_beta_error
            best_electrode = electrode + 1
            best_band = 'Beta'
            best_k = beta_errors_cv.index(min_beta_error) + 1

        # print(f"Electrode {electrode + 1} - Mu Band Errors: {mu_errors_cv}")
        # print(f"Electrode {electrode + 1} - Beta Band Errors: {beta_errors_cv}")

    print(f"Best Electrode: {best_electrode}")
    print(f"Best Band: {best_band}")
    print(f"Best K: {best_k}")
    print(f"Least 10-fold Classification Error: {least_error}")
    return best_electrode, best_band, best_k, least_error


#da el concatenated
def Concatenated_Error(Signals, Trials, Labels):

    Labels = Labels.values.flatten()

    Concatenatedmu1, Concatenatedbeta1 = bandpass_signals(Signals)
    

    # Compute relative changes for Mu band 
    ConcrelativeChanges_mu1 = rel_changes_concatenated(Concatenatedmu1, Trials, Fs)
   

    # Compute relative changes for Beta band 
    ConcrelativeChanges_beta1 = rel_changes_concatenated(Concatenatedbeta1, Trials, Fs)
    

    mu_predictions1 = knn_crossval_predict(ConcrelativeChanges_mu1, Labels)
    mu_errors_cv1 = Evaluate10foldErrors(mu_predictions1, Labels)

    beta_predictions1 = knn_crossval_predict(ConcrelativeChanges_beta1, Labels)
    beta_errors_cv1 = Evaluate10foldErrors(beta_predictions1, Labels)

    min_mu_error1 = min(mu_errors_cv1)
    min_beta_error1 = min(beta_errors_cv1)

    if min_mu_error1 < min_beta_error1:
        least_error = min_mu_error1
        best_band = 'Mu'
        best_k = mu_errors_cv1.index(min_mu_error1) + 1
    else:
        least_error = min_beta_error1
        best_band = 'Beta'
        best_k = beta_errors_cv1.index(min_beta_error1) + 1

    print(f"Best Band of concatenated sigs  {best_band}")
    print(f"Best K of the concatenated sigs: {best_k}")
    print(f"Least 10-fold Classification Error of the concatenated sigs: {least_error}")

    return best_band, best_k, least_error

# Run the Concatenated Error function

# returning the best
print("Subject 1: ")
best_electrode, best_band, best_k,least_error=Subjectloop(signals1,trials1,labels1)
print("Subject 2: ")
best_electrode2, best_band2, best_k2,least_error2=Subjectloop(signals2,trials2,labels2)
print("Subject 3: ")
best_electrode3, best_band3, best_k3,least_error3=Subjectloop(signals3,trials3,labels3)

# avgerror=least_error+least_error2+least_error3
# # average error
# avgerror=avgerror/3
# print(avgerror)
print("Concatenated:")
print("--------------------------------------------------------------------------------------------------------:")
Concatenated_Error(signals1,trials1,labels1)
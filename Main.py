import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load data for all subjects
def readsubject1():
    signals1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Signals.csv')
    labels1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Labelscsv')
    trials1 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject1_Trials.csv')
    return signals1, labels1, trials1
def readsubject2():
    signals2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Signals.csv')
    labels2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Labelscsv')
    trials2 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject2_Trials.csv')
    return signals2, labels2, trials2
def readsubject3():
    signals3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Signals.csv')
    labels3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Labelscsv')
    trials3 = pd.read_csv('/Users/muhammadabdelmohsen/Downloads/Project-5/Subject3_Trials.csv')
    return signals3, labels3, trials3

signals1,labels1,trials1=readsubject1()
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

def bandpassFilter(signal,lowcut,highcut,order):
    nyq=0.5*Fs
    low=lowcut/nyq
    high=highcut/nyq
    sig1,sig2 = butter(order,[low,high],btype='band')
    sig=filtfilt(sig1,sig2,signal)
    return sig


#the filtered data and the non filtered comparing func (3 channels only)
def comparing():
    plt.figure(figsize=(10, 6))
    for i in range(min(3, ComputeCARfiltersignal3().shape[1])):  
        plt.subplot(3, 1, i + 1)
        plt.plot(ComputeCARfiltersignal3().index, ComputeCARfiltersignal3().iloc[:, i])
        plt.title(f'CAR Filtered Signal of Channel {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.show()
    #the og data
    plt.figure(figsize=(10, 6))
    for i in range(min(3, signals3.shape[1])):  
        plt.subplot(3, 1, i + 1)
        plt.plot(signals3.index, signals3.iloc[:, i])
        plt.title(f'Original Signal of Channel {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


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

# comparing()
# CARandNon_Spectrum3channelsONLY()
#
def muband():
    mubandexample=bandpassFilter(signals1.iloc[:,i],8,13,5)
def betaband():
    betabandexample=bandpassFilter(signals1.iloc[:,1],13,30,5)

mubandexample=muband()
betabandexample=betaband()

plt.figure(figsize=(10,10))
plt.subplot(6,1,1)
plt.plot(mubandexample)
plt.title("Muband")
plt.xlabel("time")
plt.ylabel("Signals")
plt.subplot(6,1,3)
plt.plot(signals1.iloc[:, 1])
plt.title("ORIGINAL Signal")
plt.xlabel("time")
plt.ylabel("Signals")
plt.subplot(6,1,5)
plt.plot(betabandexample)
plt.title("Beta band")
plt.xlabel("time")
plt.ylabel("Signals")

plt.tight_layout()
plt.show()
# signals = pd.read_csv('Subject1_Signals.csv')

# # Iterate over each column (electrode) and print the data
# for col in signals.columns[3]:
#     #print(f"Electrode {col+1} Signals:")
#     print(signals[col].to_string(index=False))  # Print the signals in the column without index
#     print("\n")  # Add a newline for better readability

#print(electrode_data)



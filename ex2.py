import math
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.fft
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import stft
import scipy.signal as signal



def add_sine_watermark(data, rate, freq_hz, amplitude):
    """
    Adds a clean sinusoidal watermark to an audio signal.
    
    data: audio array
    rate: sample rate
    freq_hz: frequency of the watermark tone
    amplitude: amplitude of the sine wave (0..1 range suggested)
    """
    # convert stereo → mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float64)

    # normalize audio to [-1,1]
    data_norm = data / np.max(np.abs(data))

    # generate sine wave
    t = np.arange(len(data_norm)) / rate
    sine = amplitude * np.sin(2 * np.pi * freq_hz * t)

    # add watermark
    watermarked = data_norm + sine

    # normalize again to avoid clipping
    watermarked = watermarked / np.max(np.abs(watermarked))

    return watermarked

def save_audio(filename, rate, data):
    data_int16 = np.int16(data * 32767)
    wavfile.write(filename, rate, data_int16)


def make_bad(filename):
    rate, data = wavfile.read(filename)
    watermarked = add_sine_watermark(
        data, rate,
        freq_hz=1000,   # clearly audible
        amplitude=0.6   # strong and annoying
    )
    save_audio("bad_watermark.wav", rate, watermarked)


def make_good(filename):
    rate, data = wavfile.read(filename)
    watermarked = add_sine_watermark(
        data, rate,
        freq_hz=18000,   # very high pitch
        amplitude=0.2    # subtle enough to be inaudible to adults
    )
    save_audio("good_watermark.wav", rate, watermarked)


def normalize_audio(data):
    """Normalize audio data to the range [-1, 1]."""
    return data / np.max(np.abs(data))


def compute_stft(signal, fs, n_fft=1024, hop_length=512):
    """
    Compute the STFT of a signal using SciPy.
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)
    return f, t, magnitude


def process_all_audio_in_directory(target_path, audios_file):
    """Process all audio files in the specified directory."""
    array_spectograms = []
    for file_name in audios_file:
        array_spectograms.append(file_name)
    return array_spectograms


def process_zoomed_spectrograms(target_path, audios_file):
    """
    Plots zoomed-in spectrograms to reveal the watermark shape.
    Focuses on 15kHz-22kHz and the first 2 seconds.
    """
    num_files = len(audios_file)
    cols = 3
    rows = math.ceil(num_files / cols)
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i, file_name in enumerate(audios_file):
        full_path = os.path.join(target_path, file_name)
        
        try:
            rate, data = scipy.io.wavfile.read(full_path)
            
            # Convert to mono if needed
            if data.ndim > 1:
                data = data.mean(axis=1)
                
            plt.subplot(rows, cols, i + 1)
            
            # We use a smaller NFFT for better time resolution if needed, 
            # but 1024 is usually fine.
            plt.specgram(
                data, 
                Fs=rate, 
                NFFT=1024, 
                noverlap=512, 
                cmap='inferno'
            )
            
            plt.title(f"{file_name} (Zoomed)")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            
            # --- THE MAGIC PART: ZOOM IN ---
            # Look only at the high frequencies where the watermark lives
            plt.ylim(15000, 22000) 
            
            # Look only at the first 2 seconds to see the shape details clearly
            plt.xlim(0, 2) 
            
        except FileNotFoundError:
            print(f"Could not find file: {full_path}")
            
    plt.tight_layout()
    plt.show()

def smooth_signal(arr, window_size=20):
    """Simple moving average smoothing."""
    return np.convolve(arr, np.ones(window_size)/window_size, mode='same')

def load_audio(file_path):
    """Load WAV file and convert to mono float64."""
    rate, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return rate, data.astype(np.float64)

def extract_watermark_sine(file_path, min_freq=18000):
    """
    Detect the dominant sinusoidal watermark in the audio file.
    Returns dominant frequency, amplitude, and center time.
    """
    rate, data = load_audio(file_path)
    data = data / np.max(np.abs(data))  # normalize

    f, t, Zxx = compute_stft(data, rate, n_fft=4096, hop_length=1024)
    magnitude = np.abs(Zxx)

    # only high frequencies where watermark lives
    high_idx = np.where(f >= min_freq)[0]
    f_high = f[high_idx]
    mag_high = magnitude[high_idx, :]

    # strongest peak
    freq_idx, time_idx = np.unravel_index(np.argmax(mag_high), mag_high.shape)
    dominant_freq = f_high[freq_idx]
    amplitude = mag_high[freq_idx, time_idx]
    center_time = t[time_idx]

    return {
        "dominant_frequency": float(dominant_freq),
        "center_time": float(center_time)
    }

def classify_watermarks(folder_path):
    freq_magnitudes = []
    audio_files = []
    sine_features = []
    min_len = float('inf')

    # load all 9 watermarked files
    for i in range(9):
        file_name = f"{i}_watermarked.wav"
        file_path = os.path.join(folder_path, file_name)
        audio_files.append(file_name)

        sample_rate, data = load_audio(file_path)
        f, t, stft_mag = compute_stft(data, sample_rate)

        # get magnitude of target frequency (~20kHz)
        target_freq = 20000.0
        freq_idx = np.argmin(np.abs(f - target_freq))
        arr = stft_mag[freq_idx, :]

        # smooth and store
        arr = smooth_signal(arr)
        freq_magnitudes.append(arr)
        min_len = min(min_len, len(arr))

        # extract sine features
        features = extract_watermark_sine(file_path)
        sine_features.append(features)

    # make all arrays same length for clustering
    freq_magnitudes = [arr[:min_len] for arr in freq_magnitudes]
    X = np.stack(freq_magnitudes)

    # cluster into 3 groups
    clustering = AgglomerativeClustering(n_clusters=3)
    labels = clustering.fit_predict(X)

    # group audio files based on cluster labels
    groups_dict = {0: [], 1: [], 2: []}
    for i, label in enumerate(labels):
        groups_dict[label].append(audio_files[i])

    final_groups = {
        "Group 1": groups_dict[0],
        "Group 2": groups_dict[1],
        "Group 3": groups_dict[2],
    }

    # print results
    for group_name, files in final_groups.items():
        print(f"{group_name}: {files}")


    return final_groups, sine_features

def extract_shared_watermark_from_three(files, f_low=16000, f_high=20000):
    """
    Given 3 audio files that all contain the SAME sine watermark,
    this function extracts the watermark frequency, amplitude, and reconstructs
    the watermark waveform.

    Uses YOUR functions:
        - load_audio()
        - compute_stft()
        - smooth_signal()

    Returns dictionary with:
        - dominant_frequency
        - amplitude_envelope (interpolated to full signal)
        - watermark_signal (clean sine wave)
        - time_axis
    """
    signals = []
    rates = []

    # Load & normalize audio (your function)
    for file in files:
        rate, data = load_audio(file)
        data = data / np.max(np.abs(data))
        signals.append(data)
        rates.append(rate)

    # assume same rate
    fs = rates[0]

    # Cut all to same length
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    # Average the 3 signals → reduces noise, keeps watermark
    avg_signal = np.mean(signals, axis=0)

    # Compute STFT (your function)
    f, t, Zxx = compute_stft(avg_signal, fs)
    mag = np.abs(Zxx)

    # Restrict to frequency band
    idx = np.where((f >= f_low) & (f <= f_high))[0]
    f_band = f[idx]
    mag_band = mag[idx, :]

    # Find dominant frequency peak
    freq_idx, time_idx = np.unravel_index(np.argmax(mag_band), mag_band.shape)
    dom_freq = float(f_band[freq_idx])

    # Amplitude envelope over time
    amp_env = mag_band[freq_idx, :]
    amp_env = smooth_signal(amp_env, window_size=10)  # your smoothing

    # Interpolate to full length
    time_axis = np.arange(min_len) / fs
    amp_interp = np.interp(time_axis, t, amp_env)

    # Estimate phase using Hilbert transform
    analytic = signal.hilbert(avg_signal)
    phase = float(np.unwrap(np.angle(analytic))[np.argmax(amp_interp)])

    # Reconstruct watermark
    watermark = amp_interp * np.sin(2 * np.pi * dom_freq * time_axis + phase)

    return {
        "dominant_frequency": dom_freq,
        "amplitude_envelope": amp_interp,
        "watermark_signal": watermark,
        "time_axis": time_axis,
        "phase": phase
    }

def detect_watermark_peak(signal, fs, target_freq=20000, smooth_window=20):
    """Detect peak magnitude in the STFT around the target frequency."""
    f, t, stft_mag = compute_stft(signal, fs)
    freq_idx = np.argmin(np.abs(f - target_freq))
    mag_arr = stft_mag[freq_idx, :]
    # smooth signal
    mag_arr = np.convolve(mag_arr, np.ones(smooth_window)/smooth_window, mode='same')
    peaks, _ = find_peaks(mag_arr, height=np.max(mag_arr)*0.5, distance=10)
    return t[peaks], mag_arr[peaks]

def estimate_speedup(original_peaks, modified_peaks):
    """Estimate speedup factor x by comparing peak positions in time."""
    if len(original_peaks) == 0 or len(modified_peaks) == 0:
        return None
    # simple ratio using first peak
    x = (modified_peaks[0]) / (original_peaks[0])
    return x

def detect_speedup_method(file1_path, file2_path):
    """Detect which file is time-domain speedup vs frequency-domain speedup."""
    # load both files
    fs1, data1 = load_audio(file1_path)
    fs2, data2 = load_audio(file2_path)
    
    # detect watermark peaks
    t1_peaks, _ = detect_watermark_peak(data1, fs1)
    t2_peaks, _ = detect_watermark_peak(data2, fs2)
    
    # estimate relative speedup
    if len(t1_peaks) == 0 or len(t2_peaks) == 0:
        print("Could not detect peaks reliably.")
        return
    
    # time-domain speedup distorts peak spacing more
    time_domain_file = file1_path if not np.ptp(t1_peaks)>np.ptp(t2_peaks) else file2_path
    freq_domain_file = file2_path if time_domain_file==file1_path else file1_path
    
    # speedup factor estimation
    speedup_factor = (len(data1)/len(data2)) if time_domain_file==file1_path else (len(data2)/len(data1))
    
    print(f"Time-domain speedup: {time_domain_file}")
    print(f"Frequency-domain speedup: {freq_domain_file}")
    print(f"Estimated speedup factor: {speedup_factor:.3f}")


if __name__ == "__main__":

    make_bad("Task 1/task1.wav")
    make_good("Task 1/task1.wav")
    audios_file = ["0_watermarked.wav", "1_watermarked.wav", "2_watermarked.wav", "3_watermarked.wav", 
                   "4_watermarked.wav", "5_watermarked.wav", "6_watermarked.wav",
                   "7_watermarked.wav", "8_watermarked.wav"]
    # features = process_watermark_group("Task 2", audios_file)
    process_zoomed_spectrograms("Task 2", audios_file)
    classify_watermarks("/cs/usr/karin.malka/ImageProcessing/ex2/Exercise Inputs-20251118/Task 2/")


    file1 = "/cs/usr/karin.malka/ImageProcessing/ex2/Exercise Inputs-20251118/Task 3/task3_watermarked_method1.wav"
    file2 = "/cs/usr/karin.malka/ImageProcessing/ex2/Exercise Inputs-20251118/Task 3/task3_watermarked_method2.wav"
    detect_speedup_method(file1, file2)   

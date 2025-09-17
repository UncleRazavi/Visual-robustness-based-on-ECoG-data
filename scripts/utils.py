#!/usr/bin/env python3
"""
Utility functions for MVPA analysis of ECoG data
"""

import os
import pickle
import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import scipy.stats as stats
import scipy.signal as signal

try:
    from nilearn import plotting
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

warnings.filterwarnings('ignore')


def load_ecog_data(filepath, subject_idx, session_idx):
    """Load ECoG data for a specific subject and session from .npz file."""
    try:
        alldat = np.load(filepath, allow_pickle=True)['dat']
        return alldat[subject_idx][session_idx]
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except IndexError:
        print(f"Invalid subject/session index: {subject_idx}/{session_idx}")
        return None


def create_mne_raw_object(session_data, sfreq):
    """Create an MNE Raw object with montage and annotations."""
    data = session_data['V'].T
    n_channels = data.shape[0]
    ch_names = [f"ECOG_{i+1:03}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['ecog'] * n_channels)
    raw = mne.io.RawArray(data, info)
    locs = session_data['locs']
    ch_pos = {ch_names[i]: locs[i] for i in range(n_channels)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='mni_tal')
    raw.set_montage(montage)
    onsets = session_data['t_on'] / sfreq
    durations = np.full_like(onsets, 1.0)
    stim_cat = session_data['stim_cat']
    stim_noise = session_data['stim_noise']
    descriptions = [
        f"{'house' if c == 1 else 'face'}/noise_{np.round(np.squeeze(stim_noise[i])/100, 2):.2f}"
        for i, c in enumerate(stim_cat)
    ]
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    return raw


def preprocess_ecog(raw, params):
    """Preprocess MNE Raw object: resample, drop channels, filter, notch, rereference."""
    raw = raw.copy()
    if params.get('resample_rate'):
        raw.resample(params['resample_rate'])
    if params.get('exclude_channels'):
        to_drop = [ch for ch in params['exclude_channels'] if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)
    l_freq = params.get('l_freq')
    h_freq = params.get('h_freq')
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    if params.get('notch_freqs'):
        raw.notch_filter(freqs=params['notch_freqs'])
    if params.get('rereference') == 'car':
        raw.set_eeg_reference('average', projection=False)
    return raw


def extract_broadband_power(raw):
    """Extract high-gamma broadband power as a new MNE Raw object."""
    sfreq = raw.info['sfreq']
    V = raw.get_data().T.astype('float32')
    b, a = signal.butter(3, [50], btype='high', fs=sfreq)
    V = signal.filtfilt(b, a, V, axis=0)
    V = np.abs(V) ** 2
    b, a = signal.butter(3, [10], btype='low', fs=sfreq)
    V = signal.filtfilt(b, a, V, axis=0)
    V = V / V.mean(axis=0)
    info = raw.info.copy()
    info['description'] = 'Broadband Power'
    raw_broadband = mne.io.RawArray(V.T, info)
    raw_broadband.set_annotations(raw.annotations)
    return raw_broadband


def generate_output_folder_name(config):
    """Generate output folder name based on config."""
    parts = ["mvpa_results"]
    mode = config.get('processing_mode', 'ecog')
    parts.append("high_gamma" if mode == 'high_gamma' else "ecog")
    pre = config.get('preprocess_params', {})
    l_freq = pre.get('l_freq')
    h_freq = pre.get('h_freq')
    if l_freq is not None and h_freq is not None:
        parts.append(f"{l_freq}to{h_freq}Hz")
    elif l_freq is not None:
        parts.append(f"hp{l_freq}Hz")
    elif h_freq is not None:
        parts.append(f"lp{h_freq}Hz")
    else:
        parts.append("raw")
    if pre.get('resample_rate'):
        parts.append(f"{pre['resample_rate']}Hz")
    noise_range = config.get('noise_range')
    if noise_range:
        parts.append(f"noise{noise_range[0]:.2f}to{noise_range[1]:.2f}")
    win = config.get('window_length_ms')
    stride = config.get('stride_ms')
    if win and stride:
        parts.append(f"win{win}ms_stride{stride}ms")
    return " - ".join(parts)


class MVPAAnalyzer:
    """MVPA analysis for ECoG data using sliding window SVM."""

    def __init__(self, config):
        self.config = config
        self.sfreq = config.get('resample_rate', config.get('sampling_rate', 1000))
        self.window_length_samples = int(config['window_length_ms'] * self.sfreq / 1000)
        self.stride_samples = int(config['stride_ms'] * self.sfreq / 1000)
        self.baseline_start_samples = int(config['baseline_start_ms'] * self.sfreq / 1000)
        self.analysis_start_samples = int(config['analysis_start_ms'] * self.sfreq / 1000)
        self.analysis_end_samples = int(config['analysis_end_ms'] * self.sfreq / 1000)
        print(f"MVPA: mode={config.get('processing_mode','ecog')}, sfreq={self.sfreq}Hz, "
              f"win={self.window_length_samples} samples, stride={self.stride_samples} samples, "
              f"reps={config['n_repetitions']}, test_size={config['test_size']}")

    def analyze_subject(self, raw_data, subject_idx):
        """Analyze all channels for a single subject."""
        noise_range = self.config.get('noise_range')
        epochs_data, labels, times = self.extract_epochs_and_labels(raw_data, noise_range)
        if epochs_data is None:
            print("No valid epochs found!")
            return None
        electrode_locs = self.extract_electrode_locations(raw_data)
        subject_results = {
            'subject_idx': subject_idx,
            'n_channels': epochs_data.shape[1],
            'n_trials': epochs_data.shape[0],
            'processing_mode': self.config.get('processing_mode', 'ecog'),
            'noise_range': noise_range,
            'electrode_locations': electrode_locs,
            'channel_results': {},
            'all_channels_result': None,
        }
        for channel_idx, channel_name in enumerate(tqdm(raw_data.ch_names, desc="Channels")):
            res = self.analyze_channel(epochs_data, labels, channel_idx, channel_name)
            if res is not None:
                subject_results['channel_results'][channel_name] = res
        all_channels_result = self.analyze_all_channels(epochs_data, labels)
        if all_channels_result is not None:
            subject_results['all_channels_result'] = all_channels_result
        return subject_results

    def extract_epochs_and_labels(self, raw, noise_range=None):
        """Extract epochs and labels from MNE Raw object."""
        annotations = raw.annotations
        onsets = annotations.onset
        descriptions = annotations.description
        labels, valid_onsets = [], []
        for i, desc in enumerate(descriptions):
            if '/' in desc and 'noise_' in desc:
                cat, noise = desc.split('/')
                try:
                    noise_val = float(noise.split('_')[1])
                except Exception:
                    continue
                if noise_range:
                    if not (noise_range[0] <= noise_val <= noise_range[1]):
                        continue
                if 'house' in cat:
                    labels.append(0)
                    valid_onsets.append(onsets[i])
                elif 'face' in cat:
                    labels.append(1)
                    valid_onsets.append(onsets[i])
        labels = np.array(labels)
        valid_onsets = np.array(valid_onsets)
        if len(labels) == 0:
            print("No valid stimulus events found!")
            return None, None, None
        tmin = self.config['baseline_start_ms'] / 1000
        tmax = self.config['analysis_end_ms'] / 1000
        events = np.array([[int(onset * self.sfreq), 0, label + 1] for onset, label in zip(valid_onsets, labels)])
        event_id = {'house': 1, 'face': 2}
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        epochs_data = epochs.get_data()
        epoch_labels = epochs.events[:, 2] - 1
        times = epochs.times
        return epochs_data, epoch_labels, times

    def extract_electrode_locations(self, raw_data):
        """Extract electrode locations from MNE Raw object."""
        locs = {}
        if raw_data.info['dig'] is not None:
            for ch_name in raw_data.ch_names:
                ch_idx = raw_data.ch_names.index(ch_name)
                if ch_idx < len(raw_data.info['chs']):
                    locs[ch_name] = raw_data.info['chs'][ch_idx]['loc'][:3]
        return locs

    def sliding_window_mvpa_single_channel(self, channel_data, labels, random_seed):
        """Sliding window MVPA for a single channel."""
        n_trials, n_timepoints = channel_data.shape
        start = max(0, self.analysis_start_samples - self.baseline_start_samples)
        end = min(n_timepoints, self.analysis_end_samples - self.baseline_start_samples)
        n_windows = (end - start - self.window_length_samples) // self.stride_samples + 1
        if n_windows <= 0:
            return []
        accuracies = []
        try:
            train_idx, test_idx = train_test_split(
                range(n_trials), test_size=self.config['test_size'],
                random_state=random_seed, stratify=labels
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                range(n_trials), test_size=self.config['test_size'],
                random_state=random_seed
            )
        for window_idx in range(n_windows):
            s = start + window_idx * self.stride_samples
            e = s + self.window_length_samples
            X = channel_data[:, s:e].reshape(n_trials, -1)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            if len(np.unique(y_train)) < 2:
                accuracies.append(0.5)
                continue
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            svm = SVC(kernel=self.config['svm_kernel'], C=self.config['svm_c'], random_state=random_seed)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        return accuracies

    def analyze_channel(self, epochs_data, labels, channel_idx, channel_name):
        """Analyze a single channel with multiple random seeds."""
        channel_data = epochs_data[:, channel_idx, :]
        all_accuracies = []
        for rep in range(self.config['n_repetitions']):
            acc = self.sliding_window_mvpa_single_channel(channel_data, labels, rep)
            if acc:
                all_accuracies.append(acc)
        if not all_accuracies:
            return None
        all_accuracies = np.array(all_accuracies)
        mean_accuracy = np.mean(all_accuracies, axis=0)
        std_accuracy = np.std(all_accuracies, axis=0)
        start = max(0, self.analysis_start_samples - self.baseline_start_samples)
        window_times = []
        for i in range(len(mean_accuracy)):
            center = start + i * self.stride_samples + self.window_length_samples // 2
            time_ms = (center + self.baseline_start_samples) * 1000 / self.sfreq
            window_times.append(time_ms)
        return {
            'channel_name': channel_name,
            'channel_idx': channel_idx,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'all_accuracies': all_accuracies,
            'window_times': np.array(window_times),
            'n_repetitions': self.config['n_repetitions'],
            'n_windows': len(mean_accuracy),
            'peak_accuracy': np.max(mean_accuracy),
            'peak_time': window_times[np.argmax(mean_accuracy)]
        }

    def sliding_window_mvpa_all_channels(self, epochs_data, labels, random_seed):
        """Sliding window MVPA using all channels together."""
        n_trials, n_channels, n_timepoints = epochs_data.shape
        start = max(0, self.analysis_start_samples - self.baseline_start_samples)
        end = min(n_timepoints, self.analysis_end_samples - self.baseline_start_samples)
        n_windows = (end - start - self.window_length_samples) // self.stride_samples + 1
        if n_windows <= 0:
            return []
        accuracies = []
        try:
            train_idx, test_idx = train_test_split(
                range(n_trials), test_size=self.config['test_size'],
                random_state=random_seed, stratify=labels
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                range(n_trials), test_size=self.config['test_size'],
                random_state=random_seed
            )
        for window_idx in range(n_windows):
            s = start + window_idx * self.stride_samples
            e = s + self.window_length_samples
            X = epochs_data[:, :, s:e].reshape(n_trials, -1)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            if len(np.unique(y_train)) < 2:
                accuracies.append(0.5)
                continue
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            svm = SVC(kernel=self.config['svm_kernel'], C=self.config['svm_c'], random_state=random_seed)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        return accuracies

    def analyze_all_channels(self, epochs_data, labels):
        """Analyze using all channels together with multiple random seeds."""
        all_accuracies = []
        for rep in tqdm(range(self.config['n_repetitions']), desc="All channels"):
            acc = self.sliding_window_mvpa_all_channels(epochs_data, labels, rep)
            if acc:
                all_accuracies.append(acc)
        if not all_accuracies:
            return None
        all_accuracies = np.array(all_accuracies)
        mean_accuracy = np.mean(all_accuracies, axis=0)
        std_accuracy = np.std(all_accuracies, axis=0)
        start = max(0, self.analysis_start_samples - self.baseline_start_samples)
        window_times = []
        for i in range(len(mean_accuracy)):
            center = start + i * self.stride_samples + self.window_length_samples // 2
            time_ms = (center + self.baseline_start_samples) * 1000 / self.sfreq
            window_times.append(time_ms)
        return {
            'channel_name': 'ALL_CHANNELS',
            'channel_idx': -1,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'all_accuracies': all_accuracies,
            'window_times': np.array(window_times),
            'n_repetitions': self.config['n_repetitions'],
            'n_windows': len(mean_accuracy),
            'peak_accuracy': np.max(mean_accuracy),
            'peak_time': window_times[np.argmax(mean_accuracy)]
        }
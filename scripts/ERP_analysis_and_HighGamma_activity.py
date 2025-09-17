import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt

# --- Configuration ---
SUBJECT_INDEX = 1
SESSION_INDEX = 1
SAMPLING_RATE = 1000
NOISE_LEVEL_TO_PLOT = 'all'  # or a float like 0.5
FACE_CH_IDX = 46
HOUSE_CH_IDX = 43

def load_ecog_data(filepath, subject_idx, session_idx):
    """Load ECoG data for a subject and session from a .npz file."""
    try:
        alldat = np.load(filepath, allow_pickle=True)['dat']
        return alldat[subject_idx][session_idx]
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def create_mne_raw_object(session_data, sfreq):
    """Create MNE Raw object with montage and stimulus annotations."""
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
        f"{'house' if c == 1 else 'face'}/noise_{np.squeeze(stim_noise[i])/100:.1f}"
        for i, c in enumerate(stim_cat)
    ]
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    return raw

def preprocess_ecog(raw, params):
    """Apply band-pass, notch filter, and rereference to ECoG data."""
    raw = raw.copy()
    if 'l_freq' in params and 'h_freq' in params:
        raw.filter(l_freq=params['l_freq'], h_freq=params['h_freq'])
    if 'notch_freqs' in params:
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
    V = np.abs(V)**2
    b, a = signal.butter(3, [10], btype='low', fs=sfreq)
    V = signal.filtfilt(b, a, V, axis=0)
    V = V / V.mean(axis=0)
    info = raw.info.copy()
    info['description'] = 'Broadband Power'
    raw_broadband = mne.io.RawArray(V.T, info)
    raw_broadband.set_annotations(raw.annotations)
    return raw_broadband

def plot_broadband_evoked(raw_broadband, tmin=-0.2, tmax=0.4, n_channels_to_plot=50, noise_level='all'):
    """Plot evoked broadband response for each channel."""
    events, event_id = mne.events_from_annotations(raw_broadband)
    epochs = mne.Epochs(raw_broadband, events, event_id, tmin=tmin, tmax=tmax, preload=True, baseline=(None, 0))
    try:
        if noise_level == 'all':
            evoked_face = epochs['face'].average()
            evoked_house = epochs['house'].average()
            title = "(All Noise Levels)"
        else:
            face_query = f'face/noise_{noise_level:.1f}'
            house_query = f'house/noise_{noise_level:.1f}'
            evoked_face = epochs[face_query].average()
            evoked_house = epochs[house_query].average()
            title = f"(Noise Level: {noise_level:.1f})"
    except KeyError:
        print(f"Noise level {noise_level} not found.")
        return
    plt.figure(figsize=(20, 10))
    for i in range(n_channels_to_plot):
        ax = plt.subplot(5, 10, i + 1)
        ax.plot(evoked_house.times * 1000, evoked_house.data[i], label='House')
        ax.plot(evoked_face.times * 1000, evoked_face.data[i], label='Face')
        ax.set_title(raw_broadband.ch_names[i])
        ax.set_ylim([0, 4])
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Normalized Power')
            ax.set_xlabel('Time (ms)')
    plt.suptitle(f"Average Broadband Response to Faces vs. Houses {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_broadband_epochs_image(raw_broadband, face_ch_idx, house_ch_idx, tmin=-0.2, tmax=0.4, noise_level='all'):
    """Plot heatmaps of all trials for specified channels."""
    events, event_id = mne.events_from_annotations(raw_broadband)
    epochs = mne.Epochs(raw_broadband, events, event_id, tmin=tmin, tmax=tmax, preload=True, baseline=(None, 0))
    epochs_to_plot = epochs
    title = " (All Noise Levels)"
    if noise_level != 'all':
        query = f'noise_{noise_level:.2f}'
        if any(query in desc for desc in epochs.event_id):
            epochs_to_plot = epochs[query]
            title = f" (Noise Level: {noise_level:.2f})"
        else:
            print(f"Noise level {noise_level} not found. Plotting all noise levels.")
    face_title = f'Face-Selective Channel (Index: {face_ch_idx}){title}'
    epochs_to_plot.plot_image(picks=[face_ch_idx], order=np.argsort(epochs_to_plot.events[:, 2]), vmin=0, vmax=7, cmap='magma', title=face_title)
    house_title = f'House-Selective Channel (Index: {house_ch_idx}){title}'
    epochs_to_plot.plot_image(picks=[house_ch_idx], order=np.argsort(epochs_to_plot.events[:, 2]), vmin=0, vmax=7, cmap='magma', title=house_title)
    plt.show()

def plot_channel_erps(raw, tmin=-0.2, tmax=1.0, n_channels_to_plot=50, noise_level='all'):
    """Plot ERP for each channel, comparing faces and houses."""
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, preload=True, baseline=(None, 0))
    try:
        if noise_level == 'all':
            evoked_face = epochs['face'].average()
            evoked_house = epochs['house'].average()
            title = "(All Noise Levels)"
        else:
            face_query = f'face/noise_{noise_level:.1f}'
            house_query = f'house/noise_{noise_level:.1f}'
            evoked_face = epochs[face_query].average()
            evoked_house = epochs[house_query].average()
            title = f"(Noise Level: {noise_level:.1f})"
    except KeyError:
        print(f"Noise level {noise_level} not found.")
        return
    plt.figure(figsize=(20, 10))
    for i in range(n_channels_to_plot):
        ax = plt.subplot(5, 10, i + 1)
        ax.plot(evoked_house.times * 1000, evoked_house.data[i] * 1e6, label='House')
        ax.plot(evoked_face.times * 1000, evoked_face.data[i] * 1e6, label='Face')
        ax.set_title(raw.ch_names[i])
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Voltage (µV)')
            ax.set_xlabel('Time (ms)')
    plt.suptitle(f"Event-Related Potentials (ERPs) to Faces vs. Houses {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    filepath = r'faceshouses.npz'
    session_data = load_ecog_data(filepath, SUBJECT_INDEX, SESSION_INDEX)
    if session_data:
        raw = create_mne_raw_object(session_data, SAMPLING_RATE)
        raw_broadband = extract_broadband_power(raw)
        plot_broadband_evoked(raw_broadband, n_channels_to_plot=50, noise_level=NOISE_LEVEL_TO_PLOT)
        plot_broadband_epochs_image(raw_broadband, FACE_CH_IDX, HOUSE_CH_IDX, noise_level=NOISE_LEVEL_TO_PLOT)
        preprocess_params = {
            'l_freq': 0.5,
            'h_freq': 300.0,
            'notch_freqs': np.arange(60, 241, 60),
            'rereference': 'car'
        }
        raw_preprocessed = preprocess_ecog(raw, preprocess_params)
        raw_preprocessed.compute_psd().plot()
        input()
        plot_channel_erps(raw_preprocessed, n_channels_to_plot=50, noise_level=NOISE_LEVEL_TO_PLOT)

import numpy as np
import mne

def load_ecog_data(filepath, subject_idx, session_idx):
    """Load ECoG data for a subject and session from a .npz file."""
    try:
        alldat = np.load(filepath, allow_pickle=True)['dat']
        return alldat[subject_idx][session_idx]
    except (FileNotFoundError, IndexError):
        return None

def create_mne_raw_object(session_data, sfreq):
    """Convert session data to an MNE Raw object with montage and annotations."""
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
    """Preprocess MNE Raw object: drop channels, filter, notch, rereference."""
    raw = raw.copy()
    if params.get('exclude_channels'):
        to_drop = [ch for ch in params['exclude_channels'] if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)
    if params.get('l_freq') is not None and params.get('h_freq') is not None:
        raw.filter(l_freq=params['l_freq'], h_freq=params['h_freq'])
    if params.get('notch_freqs'):
        raw.notch_filter(freqs=params['notch_freqs'])
    if params.get('rereference') == 'car':
        raw.set_eeg_reference('average', projection=False)
    return raw

# Example usage
if __name__ == '__main__':
    FILEPATH = r'faceshouses.npz'
    SAMPLING_RATE = 1000
    PREPROCESS_PARAMS = {
        'l_freq': 0.5,
        'h_freq': 290.0,
        'exclude_channels': ['ECoG1', 'ECoG2', 'ECoG3']
    }
    N_SUBJECTS, N_SESSIONS = 2, 2
    all_preprocessed_data = {}
    for sub_idx in range(N_SUBJECTS):
        all_preprocessed_data[sub_idx] = {}
        for ses_idx in range(N_SESSIONS):
            session_data = load_ecog_data(FILEPATH, sub_idx, ses_idx)
            if session_data:
                raw = create_mne_raw_object(session_data, SAMPLING_RATE)
                preprocessed = preprocess_ecog(raw, PREPROCESS_PARAMS)
                all_preprocessed_data[sub_idx][ses_idx] = preprocessed
#!/usr/bin/env python3
"""
MVPA Analysis for ECoG Data

Main script for running a sliding window SVM classification analysis.
Iterates through different stimulus noise levels, performs MVPA for each,
and saves results in separate folders.
"""

import os
import numpy as np
from utils import load_subject_data, MVPAAnalyzer, generate_output_folder_name

def main():
    # ------------------- CONFIGURATION -------------------
    config = {
        'filepath': r'C:\Users\Mohammadreza\Desktop\Neuromatch CN\Data\faceshouses.npz',
        'original_sampling_rate': 1000,
        'n_subjects': 7,
        'session_idx': 1,
        'preprocess_params': {
            'resample_rate': 600,
            'l_freq': 0.5,
            'h_freq': 290,
            'notch_freqs': None,
            'rereference': None,
            'exclude_channels': []
        },
        'processing_mode': 'ecog',  # or 'high_gamma'
        'window_length_ms': 50,
        'stride_ms': 10,
        'n_repetitions': 10,
        'test_size': 0.3,
        'svm_kernel': 'rbf',
        'svm_c': 1.0,
        'baseline_start_ms': -200,
        'baseline_end_ms': 0,
        'analysis_start_ms': -200,
        'analysis_end_ms': 500,
        'alpha': 0.05,
        'generate_brain_videos': False,
    }

    # ------------------- NOISE RANGES --------------------
    noise_ranges = [
        [0.0, 0.2],
        [0.2, 0.4],
        [0.4, 0.6],
        [0.6, 0.8],
        [0.8, 1.0]
    ]

    print("Starting the MVPA analysis pipeline...")

    for noise_range in noise_ranges:
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR NOISE RANGE: {noise_range}")
        print(f"{'='*80}\n")

        # Prepare config for this run
        current_config = config.copy()
        current_config['noise_range'] = noise_range
        preproc = current_config['preprocess_params']
        current_config['sampling_rate'] = preproc['resample_rate'] or config['original_sampling_rate']
        current_config['output_dir'] = generate_output_folder_name(current_config)
        os.makedirs(current_config['output_dir'], exist_ok=True)

        print(f"Mode: {current_config['processing_mode']}, "
              f"Sampling rate: {current_config['sampling_rate']} Hz, "
              f"Noise: {current_config['noise_range']}, "
              f"Output: {current_config['output_dir']}\n")

        analyzer = MVPAAnalyzer(current_config)

        for subject_idx in range(current_config['n_subjects']):
            print(f"\n--- Subject {subject_idx} ---")
            result_file = os.path.join(current_config['output_dir'], f'subject_{subject_idx}_results.pkl')
            if os.path.exists(result_file):
                print("Results already exist. Skipping.")
                continue

            print("Loading and preprocessing data...")
            raw_data = load_subject_data(current_config, subject_idx)
            if raw_data is None:
                print("Failed to load/process data. Skipping.")
                continue

            subject_results = analyzer.analyze_subject(raw_data, subject_idx)
            if subject_results:
                analyzer.save_subject_results(subject_results, subject_idx)
                analyzer.generate_subject_plots(subject_results, subject_idx)
                if current_config['generate_brain_videos']:
                    analyzer.generate_brain_video(subject_results, subject_idx)
            else:
                print("No results for this subject.")

    print(f"\n{'='*80}")
    print("MVPA ANALYSIS COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
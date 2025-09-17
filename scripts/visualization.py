import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import pandas as pd
import warnings

try:
    from nilearn import plotting
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available for brain visualization")

warnings.filterwarnings('ignore')

class NoiseRangeVisualizer:
    def __init__(self, base_results_dir, config):
        """
        Initialize the noise range visualizer
        
        Args:
            base_results_dir: Base directory containing all noise range results
            config: Configuration dictionary with visualization parameters
        """
        self.base_results_dir = Path(base_results_dir)
        self.config = config
        
        # Find all noise range directories
        self.noise_dirs = self._find_noise_directories()
        print(f"Found {len(self.noise_dirs)} noise range directories.")
        
        # Load all results
        self.all_results = self._load_all_results()
        
        # Create output directory
        self.output_dir = self.base_results_dir / "NoiseRange_Visualizations"
        self.output_dir.mkdir(exist_ok=True)

    def _find_noise_directories(self):
        """Find directories matching noise range pattern."""
        noise_dirs = {}
        for dir_path in self.base_results_dir.glob("*noise*to*"):
            if dir_path.is_dir():
                try:
                    noise_part = [part for part in dir_path.name.split(' - ') if 'noise' in part][0]
                    min_val, max_val = noise_part.replace('noise', '').split('to')
                    noise_dirs[(float(min_val), float(max_val))] = dir_path
                except Exception:
                    continue
        return dict(sorted(noise_dirs.items()))

    def _load_all_results(self):
        """Load results from all noise range directories."""
        all_results = {}
        for noise_range, dir_path in self.noise_dirs.items():
            results_files = list(dir_path.glob("subject_*_results.pkl"))
            noise_results = {}
            for results_file in results_files:
                try:
                    subject_idx = int(results_file.stem.split('_')[1])
                    with open(results_file, 'rb') as f:
                        noise_results[subject_idx] = pickle.load(f)
                except Exception:
                    continue
            all_results[noise_range] = noise_results
        return all_results

    def _apply_smoothing_1d(self, data, sigma=1.0):
        """Apply 1D Gaussian smoothing along time axis."""
        if not self.config.get('apply_smoothing', True):
            return data
        data = np.array(data)
        if data.ndim == 1:
            return ndimage.gaussian_filter1d(data, sigma=sigma)
        return ndimage.gaussian_filter1d(data, sigma=sigma, axis=-1)

    def _detect_significant_timepoints(self, accuracies, threshold=0.55, min_duration=3):
        """Detect significant time points where accuracy is above threshold."""
        above = accuracies > threshold
        mask = np.zeros_like(above, dtype=bool)
        if np.any(above):
            diff = np.diff(np.concatenate(([False], above, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for start, end in zip(starts, ends):
                if end - start >= min_duration:
                    mask[start:end] = True
        return mask

    def generate_level1_visualizations(self):
        """Level 1: Per noise level and per subject."""
        print("\n" + "="*50)
        print("LEVEL 1: Per noise level and per subject")
        print("="*50)
        level1_dir = self.output_dir / "level1_per_subject_per_noise"
        level1_dir.mkdir(exist_ok=True)
        for noise_range, noise_data in self.all_results.items():
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            noise_dir = level1_dir / f"noise_{noise_str}"
            noise_dir.mkdir(exist_ok=True)
            for subject_idx, subject_data in noise_data.items():
                self._plot_channel_heatmap(subject_data, subject_idx, noise_range, noise_dir)
                self._plot_channel_grid(subject_data, subject_idx, noise_range, noise_dir)
                self._plot_all_channels_timecourse(subject_data, subject_idx, noise_range, noise_dir)

    def _plot_channel_heatmap(self, subject_data, subject_idx, noise_range, output_dir):
        """Heatmap for all channels of a subject."""
        channel_results = subject_data.get('channel_results')
        if not channel_results:
            return
        channel_names = list(channel_results.keys())
        n_channels = len(channel_names)
        first_channel = next(iter(channel_results.values()))
        time_points = first_channel.get('window_times')
        if time_points is None:
            return
        n_times = len(time_points)
        acc_matrix = np.zeros((n_channels, n_times))
        for i, ch in enumerate(channel_names):
            acc = np.array(channel_results[ch].get('mean_accuracy', np.zeros(n_times)))
            acc_matrix[i, :] = self._apply_smoothing_1d(acc, self.config.get('smoothing_sigma', 1.0))
        plt.figure(figsize=(15, max(8, n_channels * 0.3)))
        plt.imshow(acc_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0.4, vmax=0.8)
        plt.xlabel('Time (ms)')
        plt.ylabel('Channel')
        plt.title(f'Subject {subject_idx} - Channel Decoding\nNoise: {noise_range[0]:.1f}-{noise_range[1]:.1f}')
        plt.colorbar(label='Decoding Accuracy')
        plt.tight_layout()
        plt.savefig(output_dir / f'subject_{subject_idx}_channels_heatmap.png', dpi=300)
        plt.close()

    def _plot_channel_grid(self, subject_data, subject_idx, noise_range, output_dir):
        """6x10 grid of individual channel accuracies."""
        channel_results = subject_data.get('channel_results')
        if not channel_results:
            return
        channel_names = list(channel_results.keys())
        fig, axes = plt.subplots(6, 10, figsize=(20, 12))
        axes = axes.flatten()
        for i in range(60):
            ax = axes[i]
            if i < len(channel_names):
                ch = channel_names[i]
                ch_data = channel_results[ch]
                times = np.array(ch_data.get('window_times', []))
                acc = np.array(ch_data.get('mean_accuracy', []))
                if times.size and acc.size:
                    acc = self._apply_smoothing_1d(acc, self.config.get('smoothing_sigma', 1.0))
                    sig = self._detect_significant_timepoints(acc)
                    ax.plot(times, acc, 'b-', linewidth=1, alpha=0.8)
                    if np.any(sig):
                        ax.fill_between(times, 0.4, 0.8, where=sig, alpha=0.3, color='green')
                    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
                    ax.axvline(0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
                    ax.set_title(ch, fontsize=8)
                    ax.set_ylim(0.4, 0.8)
                    ax.set_xlim(times[0], times[-1])
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')
            else:
                ax.axis('off')
        plt.suptitle(f'Subject {subject_idx} - Channel Accuracies\nNoise: {noise_range[0]:.1f}-{noise_range[1]:.1f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'subject_{subject_idx}_channel_grid.png', dpi=300)
        plt.close()

    def _plot_all_channels_timecourse(self, subject_data, subject_idx, noise_range, output_dir):
        """Time course plot for all-channels analysis."""
        all_channels = subject_data.get('all_channels_result')
        if not all_channels:
            return
        times = np.array(all_channels.get('window_times', []))
        acc = np.array(all_channels.get('mean_accuracy', []))
        std = np.array(all_channels.get('std_accuracy', np.zeros_like(acc)))
        if not times.size or not acc.size:
            return
        acc = self._apply_smoothing_1d(acc, self.config.get('smoothing_sigma', 1.0))
        std = self._apply_smoothing_1d(std, self.config.get('smoothing_sigma', 1.0))
        plt.figure(figsize=(12, 6))
        plt.plot(times, acc, 'b-', linewidth=2, label='All Channels')
        plt.fill_between(times, acc - std, acc + std, alpha=0.3, color='blue')
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.axvline(0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Decoding Accuracy')
        plt.title(f'Subject {subject_idx} - All Channels Decoding\nNoise: {noise_range[0]:.1f}-{noise_range[1]:.1f}')
        plt.ylim(0.4, 0.8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'subject_{subject_idx}_all_channels_timecourse.png', dpi=300)
        plt.close()

    def generate_level2_visualizations(self):
        """Level 2: Per noise level across subjects (all-channels only)."""
        print("\n" + "="*50)
        print("LEVEL 2: Per noise level across subjects")
        print("="*50)
        level2_dir = self.output_dir / "level2_per_noise_across_subjects"
        level2_dir.mkdir(exist_ok=True)
        for noise_range, noise_data in self.all_results.items():
            times, accs, subj_idxs = None, [], []
            for subject_idx, subject_data in noise_data.items():
                all_channels = subject_data.get('all_channels_result')
                if all_channels and 'window_times' in all_channels and 'mean_accuracy' in all_channels:
                    times = np.array(all_channels['window_times'])
                    acc = np.array(all_channels['mean_accuracy'])
                    acc = self._apply_smoothing_1d(acc, self.config.get('smoothing_sigma', 1.0))
                    accs.append(acc)
                    subj_idxs.append(subject_idx)
            if accs and times is not None:
                self._plot_noise_level_summary(accs, subj_idxs, times, noise_range, level2_dir)

    def _plot_noise_level_summary(self, accs, subj_idxs, times, noise_range, output_dir):
        """Summary plot for a single noise level across subjects."""
        arr = np.array(accs)
        mean_acc = arr.mean(axis=0)
        sem_acc = arr.std(axis=0) / np.sqrt(len(accs))
        plt.figure(figsize=(12, 6))
        for i, (idx, acc) in enumerate(zip(subj_idxs, accs)):
            plt.plot(times, acc, alpha=0.6, linewidth=1, label=f'Subject {idx}')
        plt.plot(times, mean_acc, 'k-', linewidth=3, label='Mean')
        plt.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc, alpha=0.2, color='black')
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.axvline(0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Decoding Accuracy')
        plt.ylim(0.4, 0.8)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'All Subjects - Noise: {noise_range[0]:.1f}-{noise_range[1]:.1f}\nMean ± SEM (n={len(accs)})')
        plt.tight_layout()
        plt.savefig(output_dir / f'noise_{noise_range[0]:.1f}-{noise_range[1]:.1f}_across_subjects.png', dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Generate all levels of visualizations."""
        print("Starting noise range analysis...")
        print(f"Base directory: {self.base_results_dir}")
        print(f"Output directory: {self.output_dir}")
        self.generate_level1_visualizations()
        self.generate_level2_visualizations()
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*50)

def main():
    config = {
        'apply_smoothing': True,
        'smoothing_sigma': 1.0,
        'figure_dpi': 300,
    }
    base_results_dir = "."
    if not os.path.exists(base_results_dir):
        print(f"Error: Base results directory not found: {base_results_dir}")
        return
    visualizer = NoiseRangeVisualizer(base_results_dir, config)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
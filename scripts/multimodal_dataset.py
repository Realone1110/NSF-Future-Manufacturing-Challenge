"""
Memory-bounded, epoch-aware dataset for training the dual-encoder U-Net.

Design summary
--------------
Every call to set_epoch(epoch) throws away the previous epoch's loaded
samples and pulls a brand new set through get_multimodal_bundle_at_x, so at
any moment only n_per_epoch samples (resized down to small fixed shapes)
live in memory. Nothing accumulates across epochs.

repeatable=False (the default) draws fresh random x locations every time
set_epoch is called, so successive epochs and separate runs all see
different data, which is what you want for the actual training set.

repeatable=True seeds the draw from (seed, epoch) instead, so the same
epoch number always produces the same set of x locations across separate
runs. This is meant for the Track_21 test set, where you want a fixed,
reproducible evaluation set rather than a moving target.

Disk reads for the underlying track height/thermal arrays are already
cached per track by get_track_data, this module reuses that. SEM tile reads
are wrapped here with an lru_cache so repeated samples that land on the
same physical tile don't re-decode the image file from disk every time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from PIL import Image

from nsf_fmrg_data import get_sem_tile_paths, load_sem_tile
from frame_extraction import get_track_data
from sem_multimodal_bundle import get_multimodal_bundle_at_x


# ---------------------------------------------------------------------
# cached wrappers around the SEM disk reads
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cached_tile_paths(sem_dir, track_id):
    return tuple(get_sem_tile_paths(sem_dir, track_id))


def cached_get_sem_tile_paths(sem_dir, track_id):
    """Drop-in replacement for get_sem_tile_paths with per-track caching."""
    return list(_cached_tile_paths(sem_dir, track_id))


@lru_cache(maxsize=32)
def _cached_tile_array(path):
    return load_sem_tile(path)


def cached_load_sem_tile(path):
    """
    Drop-in replacement for load_sem_tile. Returns a copy of the cached
    array so nothing downstream can accidentally mutate the cached version.
    Cache size is capped at 32 distinct tiles, raise it if a track has more
    tiles than that and you're seeing repeated disk hits.
    """
    return _cached_tile_array(path).copy()


# ---------------------------------------------------------------------
# array preprocessing
# ---------------------------------------------------------------------

def _resize_array(arr, size_hw, fill_value=0.0, resample=Image.BILINEAR):
    """
    Resizes a 2D array to (height, width) = size_hw. NaN entries get filled
    with fill_value before resizing.

    For the height map, NaN doesn't mean missing data, it means the pixel
    sits outside the laser track, where height genuinely is zero. So the
    fill here should be a real physical value (0.0), not an estimate like a
    local mean, which would blur exactly the boundary that carries the
    local width information.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=fill_value)
    img = Image.fromarray(arr)
    img = img.resize((size_hw[1], size_hw[0]), resample=resample)
    return np.array(img, dtype=np.float32)  # np.array copies, np.asarray may not


def _resize_mask(mask, size_hw):
    """
    Resizes a binary coverage mask (1 = inside the track, 0 = outside)
    using nearest-neighbor so the result stays strictly 0/1 rather than
    picking up blended fractional values at the boundary the way bilinear
    resize would.
    """
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((size_hw[1], size_hw[0]), resample=Image.NEAREST)
    return (np.array(img, dtype=np.float32) > 127).astype(np.float32)


def _normalize(arr):
    """Per-sample min-max scaling to 0 to 1. Swap for dataset-wide stats
    later if you find per-sample scaling washes out real signal."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _x_fraction(track_id, x_mm):
    """Position along the track, expressed as a 0 to 1 fraction of its
    common x range rather than a raw millimeter value, so the same number
    means roughly the same thing on tracks of different lengths."""
    td = get_track_data(track_id)
    span = td.common_x_max - td.common_x_min
    if span <= 0:
        return 0.0
    return float(np.clip((x_mm - td.common_x_min) / span, 0.0, 1.0))


def _sample_x_locations(track_id, n_samples, rng, margin_mm=0.6):
    """
    Draws n_samples random x locations from a track's common x range,
    staying margin_mm clear of both edges so the width_window_mm crop
    around each sample never runs past the edge of available data.
    """
    td = get_track_data(track_id)
    lo = td.common_x_min + margin_mm
    hi = td.common_x_max - margin_mm
    return rng.uniform(lo, hi, size=n_samples)


def _split_count(total, n_groups):
    """Splits an integer total as evenly as possible across n_groups."""
    base = total // n_groups
    counts = [base] * n_groups
    for i in range(total - base * n_groups):
        counts[i] += 1
    return counts


# ---------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------

class MultimodalHeightDataset(Dataset):
    def __init__(
        self,
        track_ids,
        n_per_epoch,
        SEM_DIR,
        SEM_TILE_WIDTH_MM,
        get_sem_tile_paths_fn=cached_get_sem_tile_paths,
        load_sem_tile_fn=cached_load_sem_tile,
        width_window_mm=1.0,
        thermal_size=(128, 128),
        sem_size=(128, 128),
        height_size=(64, 64),
        sem_method="box",
        sem_box_mode="envelope",
        sem_mask_mode="cover_track",
        repeatable=False,
        seed=0,
        max_retries_per_sample=3,
        height_mean=None,
        height_std=None,
    ):
        self.track_ids = list(track_ids)
        self.n_per_epoch = n_per_epoch
        self.SEM_DIR = SEM_DIR
        self.SEM_TILE_WIDTH_MM = SEM_TILE_WIDTH_MM
        self.get_sem_tile_paths_fn = get_sem_tile_paths_fn
        self.load_sem_tile_fn = load_sem_tile_fn
        self.width_window_mm = width_window_mm
        self.thermal_size = thermal_size
        self.sem_size = sem_size
        self.height_size = height_size
        self.sem_method = sem_method
        self.sem_box_mode = sem_box_mode
        self.sem_mask_mode = sem_mask_mode
        self.repeatable = repeatable
        self.seed = seed
        self.max_retries_per_sample = max_retries_per_sample

        # normalizes the height target so the loss reflects proportional
        # error rather than raw millimeter scale. Pass these in explicitly
        # (computed once on the training set) so a test/eval dataset uses
        # the exact same scale rather than recomputing its own.
        self.height_mean = height_mean
        self.height_std = height_std

        self._samples = []
        self._epoch = -1

    def fit_height_normalization(self, n_calib_per_track=30):
        """
        Estimates height_mean and height_std from real (unfilled) height
        pixels across this dataset's tracks, and stores them on the
        dataset. Call this once on your training dataset before training,
        then pass the same two numbers into any other dataset (validation,
        test) via the height_mean/height_std constructor args, so every
        split gets normalized on the same scale.
        """
        rng = np.random.default_rng(self.seed)
        values = []
        for track_id in self.track_ids:
            td = get_track_data(track_id)
            for x_val in _sample_x_locations(track_id, n_calib_per_track, rng):
                try:
                    bundle = get_multimodal_bundle_at_x(
                        float(x_val),
                        track_id=track_id,
                        width_window_mm=self.width_window_mm,
                        display=False,
                        track_data=td,
                        include_sem=False,
                    )
                except Exception:
                    continue
                raw = np.asarray(bundle["height_Z_crop"], dtype=np.float32)
                valid = raw[~np.isnan(raw)]
                if valid.size:
                    values.append(valid)
        if not values:
            raise RuntimeError("couldn't sample any valid height pixels for normalization")
        all_valid = np.concatenate(values)
        self.height_mean = float(np.mean(all_valid))
        self.height_std = float(np.std(all_valid)) or 1.0
        return self.height_mean, self.height_std

    def unnormalize_target(self, arr):
        """Converts a normalized height prediction/target back to real mm,
        for plotting or physical-unit metrics."""
        if self.height_mean is None or self.height_std is None:
            raise RuntimeError("height_mean/height_std not set, call fit_height_normalization first")
        return arr * self.height_std + self.height_mean

    def set_epoch(self, epoch):
        """
        Call this before each training epoch. Replaces self._samples with a
        freshly loaded set, the previous epoch's arrays are dropped and
        garbage collected once this returns.
        """
        self._epoch = epoch
        rng = (
            np.random.default_rng(self.seed + epoch)
            if self.repeatable
            else np.random.default_rng()
        )

        per_track = _split_count(self.n_per_epoch, len(self.track_ids))

        # one TrackData per track, shared across every sample from that
        # track this epoch, so the raw height/thermal files are only read
        # from disk once per track no matter how many locations get sampled
        track_cache = {tid: get_track_data(tid) for tid in set(self.track_ids)}

        plan = []
        for track_id, n_track in zip(self.track_ids, per_track):
            for x_val in _sample_x_locations(track_id, n_track, rng):
                plan.append((track_id, float(x_val)))

        loaded = []
        for track_id, x_val in plan:
            sample = self._load_one(track_id, x_val, track_cache[track_id], rng)
            if sample is not None:
                loaded.append(sample)

        self._samples = loaded

    def _load_one(self, track_id, x_val, track_data, rng):
        """Loads a single sample, retrying with a fresh x location a few
        times if the bundle extraction fails for that particular spot
        (edge cases near tile boundaries etc)."""
        for attempt in range(self.max_retries_per_sample):
            try:
                bundle = get_multimodal_bundle_at_x(
                    x_val,
                    track_id=track_id,
                    width_window_mm=self.width_window_mm,
                    display=False,
                    track_data=track_data,
                    include_sem=True,
                    SEM_DIR=self.SEM_DIR,
                    get_sem_tile_paths=self.get_sem_tile_paths_fn,
                    load_sem_tile=self.load_sem_tile_fn,
                    SEM_TILE_WIDTH_MM=self.SEM_TILE_WIDTH_MM,
                    sem_method=self.sem_method,
                    sem_box_mode=self.sem_box_mode,
                    sem_mask_mode=self.sem_mask_mode,
                )
                return self._bundle_to_tensors(bundle, track_id)
            except Exception:
                x_val = float(_sample_x_locations(track_id, 1, rng)[0])
        return None

    def _bundle_to_tensors(self, bundle, track_id):
        thermal = _normalize(_resize_array(bundle["thermal_frame"], self.thermal_size))
        sem_img = _normalize(_resize_array(bundle["sem_masked_image"], self.sem_size))

        raw_height = np.asarray(bundle["height_Z_crop"], dtype=np.float32)
        # NaN here means outside the track, not a missing reading, so the
        # mask is captured before the fill below turns those spots into 0.0
        coverage_raw = ~np.isnan(raw_height)
        target = _resize_array(raw_height, self.height_size, fill_value=0.0)
        coverage = _resize_mask(coverage_raw, self.height_size)

        if self.height_mean is not None and self.height_std is not None:
            target = (target - self.height_mean) / self.height_std

        x_mm = bundle["x_requested_mm"]
        x_frac = _x_fraction(track_id, x_mm)

        return {
            "thermal": torch.from_numpy(thermal).unsqueeze(0),    # (1, H, W)
            "sem": torch.from_numpy(sem_img).unsqueeze(0),        # (1, H, W)
            "target": torch.from_numpy(target).unsqueeze(0),      # (1, H, W)
            "track_mask": torch.from_numpy(coverage).unsqueeze(0),  # (1, H, W), 1 = inside track
            "x_frac": torch.tensor([x_frac], dtype=torch.float32),  # (1,), 0 to 1 position along track
            "track_id": track_id,
            "x_mm": x_mm,
        }

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def make_epoch_loader(dataset, epoch, batch_size=8, shuffle=True, num_workers=0):
    """Refreshes the dataset for this epoch, then wraps it in a DataLoader.
    num_workers=0 by default since the bundle extraction already does its
    own disk-read caching, multiple worker processes would just duplicate
    that cache rather than help."""
    dataset.set_epoch(epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ---------------------------------------------------------------------
# example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    NUM_EPOCHS = 20

    train_ds = MultimodalHeightDataset(
        track_ids=["Track_8", "Track_10", "Track_14"],
        n_per_epoch=100,
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        repeatable=False,   # fresh random locations every epoch
    )
    height_mean, height_std = train_ds.fit_height_normalization()
    print(f"height normalization: mean={height_mean:.5f}mm  std={height_std:.5f}mm")

    # fixed, reproducible evaluation set on the held-out track, sharing the
    # same normalization stats as the training set
    test_ds = MultimodalHeightDataset(
        track_ids=["Track_21"],
        n_per_epoch=100,
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        repeatable=True,
        seed=42,
        height_mean=height_mean,
        height_std=height_std,
    )
    test_loader = make_epoch_loader(test_ds, epoch=0, batch_size=8, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        train_loader = make_epoch_loader(train_ds, epoch, batch_size=8, shuffle=True)
        for batch in train_loader:
            thermal = batch["thermal"]
            sem = batch["sem"]
            target = batch["target"]
            # forward pass, loss, backward pass, optimizer step go here
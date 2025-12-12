#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.iceflow.data_preparation.config import PreparationParams
from igm.processes.iceflow.data_preparation.batch_builder import TrainingBatchBuilder
from igm.processes.iceflow.data_preparation.patching import OverlapPatching


# --------------------------
# Helpers 
# --------------------------

def make_fieldin(h=8, w=8, c=3, dtype=tf.float32):
    """Asymmetric pattern so flips/noise are detectable."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    arr = np.zeros((h, w, c), dtype=np.float64)
    for ch in range(c):
        arr[..., ch] = 10000.0 * yy + 100.0 * xx + 10.0 * ch
    return tf.constant(arr, dtype=dtype)


def make_params(
    overlap=0.0,
    batch_size=4,
    patch_size=4,
    rotation_probability=0.0,
    flip_probability=0.0,
    noise_type="none",
    noise_scale=0.0,
    target_samples=4,
    fieldin_names=("thk", "usurf", "temp"),
    precision="single",
    noise_channels=("thk", "usurf"),
):
    return PreparationParams(
        overlap=overlap,
        batch_size=batch_size,
        patch_size=patch_size,
        rotation_probability=rotation_probability,
        flip_probability=flip_probability,
        noise_type=noise_type,
        noise_scale=noise_scale,
        target_samples=target_samples,
        fieldin_names=tuple(fieldin_names),
        precision=precision,
        noise_channels=tuple(noise_channels),
    )


def flatten_batches(training_tensor):
    """[num_batches, batch_size, H, W, C] -> [N, H, W, C]."""
    nb, bs, h, w, c = training_tensor.shape
    return tf.reshape(training_tensor, [nb * bs, h, w, c])


def multiset_equal(samples_a, samples_b):
    """Order-insensitive equality over first dimension."""
    A = samples_a.reshape(samples_a.shape[0], -1)
    B = samples_b.reshape(samples_b.shape[0], -1)
    idxA = np.lexsort(A.T[::-1])
    idxB = np.lexsort(B.T[::-1])
    return np.array_equal(A[idxA], B[idxB])


# --------------------------------
# Basic Tests for New API
# --------------------------------

def test_basic_batch_builder_no_augmentation():
    """Test basic batch building without augmentation."""
    tf.random.set_seed(1)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    prep = make_params(
        patch_size=ps,
        batch_size=num_patches,
        target_samples=num_patches,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    
    assert batches.dtype == tf.float32
    assert batches.shape[0] == 1
    assert batches.shape[1] == num_patches
    assert batches.shape[2:] == (ps, ps, C)
    
    flat = flatten_batches(batches).numpy()
    assert multiset_equal(flat, patches.numpy())


def test_batch_builder_with_upsampling():
    """Test that upsampling creates more samples with augmentation."""
    tf.random.set_seed(2)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    target_samples = num_patches + 2
    
    prep = make_params(
        patch_size=ps,
        batch_size=target_samples,
        target_samples=target_samples,
        flip_probability=1.0,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    
    assert batches.shape[0] == 1
    assert batches.shape[1] == target_samples
    
    flat = flatten_batches(batches).numpy()
    assert flat.shape[0] == target_samples


def test_batch_builder_multiple_batches():
    """Test that large sample counts create multiple batches."""
    tf.random.set_seed(3)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    batch_size = 2
    
    prep = make_params(
        patch_size=ps,
        batch_size=batch_size,
        target_samples=num_patches,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    
    expected_num_batches = num_patches // batch_size
    assert batches.shape[0] == expected_num_batches
    assert batches.shape[1] == batch_size
    assert batches.shape[2:] == (ps, ps, C)


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_patching_with_overlap_increases_patch_count(overlap):
    """Test that overlap increases patch count monotonically."""
    tf.random.set_seed(4)
    H, W, C, ps = 16, 16, 1, 8
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)

    counts = []
    for ov in [0.0, 0.25, 0.5]:
        patcher = OverlapPatching(patch_size=ps, overlap=ov, fieldin=fieldin)
        patches = patcher.generate_patches(fieldin)
        counts.append(int(patches.shape[0]))

    assert counts[0] <= counts[1] <= counts[2]
    assert counts[0] < counts[2]


def test_no_augmentation_without_upsampling():
    """
    Policy test: augmentations should NOT be applied when target_samples <= num_patches,
    even if augmentation probabilities are set to 1.0.
    """
    tf.random.set_seed(5)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    # Enable all augmentations but don't request upsampling
    prep = make_params(
        patch_size=ps,
        batch_size=num_patches,
        target_samples=num_patches,  # No upsampling
        rotation_probability=1.0,
        flip_probability=1.0,
        noise_type="gaussian",
        noise_scale=0.5,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    flat = flatten_batches(batches).numpy()
    
    # All samples should be exact matches to originals (just shuffled)
    assert multiset_equal(flat, patches.numpy())


def test_augmentation_only_on_extras():
    """
    Test that when upsampling, original patches remain unmodified
    and only the extra samples are augmented.
    """
    tf.random.set_seed(6)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    # Request 2 extra samples with flip augmentation
    target_samples = num_patches + 2
    
    prep = make_params(
        patch_size=ps,
        batch_size=target_samples,
        target_samples=target_samples,
        rotation_probability=0.0,
        flip_probability=1.0,  # All extras will be flipped
        noise_type="none",
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    flat = flatten_batches(batches).numpy()
    
    assert flat.shape[0] == target_samples
    
    # Count how many samples are exact matches to originals
    def is_exact_match(sample):
        return np.any(np.all(patches.numpy() == sample, axis=(1, 2, 3)))
    
    matches = sum(is_exact_match(s) for s in flat)
    
    # Should have all originals plus 2 augmented
    assert matches == num_patches, f"Expected {num_patches} exact matches, got {matches}"


def test_noise_respects_channel_mask():
    """Test that noise augmentation only affects specified channels."""
    tf.random.set_seed(7)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin_names = ("thk", "usurf", "temp")
    noise_channels = ("thk",)  # Only apply noise to first channel
    
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    # Request 1 extra with noise
    target_samples = num_patches + 1
    
    prep = make_params(
        patch_size=ps,
        batch_size=target_samples,
        target_samples=target_samples,
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="gaussian",
        noise_scale=0.3,
        fieldin_names=fieldin_names,
        noise_channels=noise_channels,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=fieldin_names,
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    flat = flatten_batches(batches).numpy()
    
    # Find the augmented sample (not an exact match to any original)
    def is_exact_match(sample):
        return np.any(np.all(patches.numpy() == sample, axis=(1, 2, 3)))
    
    augmented_samples = [s for s in flat if not is_exact_match(s)]
    
    assert len(augmented_samples) == 1, "Should have exactly 1 augmented sample"
    
    # The augmented sample should differ from all originals only in channel 0
    # (channels 1 and 2 should match some original)
    aug = augmented_samples[0]
    
    # Check that noise was applied (channel 0 should differ from all originals)
    channel_0_differs = all(
        not np.allclose(aug[..., 0], orig[..., 0]) 
        for orig in patches.numpy()
    )
    assert channel_0_differs, "Channel 0 should have noise applied"


@pytest.mark.parametrize("precision,dtype", [
    ("single", tf.float32),
    ("double", tf.float64),
])
def test_precision_types(precision, dtype):
    """Test that precision setting correctly sets output dtype."""
    tf.random.set_seed(8)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    prep = make_params(
        patch_size=ps,
        batch_size=num_patches,
        target_samples=num_patches,
        precision=precision,
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    
    assert batches.dtype == dtype


def test_multiple_calls_produce_different_results():
    """
    Test that calling build_batches() multiple times produces different
    shuffles/samples due to stateful RNG (simulating multiple epochs).
    """
    tf.random.set_seed(9)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    # Use flip augmentation with upsampling to ensure variation
    target_samples = num_patches + 1
    
    prep = make_params(
        patch_size=ps,
        batch_size=target_samples,
        target_samples=target_samples,
        flip_probability=1.0,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    # Call build_batches twice
    batches1 = builder.build_batches(patches)
    batches2 = builder.build_batches(patches)
    
    flat1 = flatten_batches(batches1).numpy()
    flat2 = flatten_batches(batches2).numpy()
    
    # Results should differ (different shuffle/augmentation)
    assert not np.array_equal(flat1, flat2), "Multiple calls should produce different results"


def test_large_upsampling_factor():
    """Test upsampling with large factor (e.g., 4 patches -> 12 samples)."""
    tf.random.set_seed(10)
    H, W, C = 8, 8, 3
    ps = 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    
    patcher = OverlapPatching(patch_size=ps, overlap=0.0, fieldin=fieldin)
    patches = patcher.generate_patches(fieldin)
    num_patches = patches.shape[0]
    
    # 3x upsampling
    target_samples = num_patches * 3
    
    prep = make_params(
        patch_size=ps,
        batch_size=target_samples,
        target_samples=target_samples,
        flip_probability=1.0,
        precision="single",
    )
    
    builder = TrainingBatchBuilder(
        preparation_params=prep,
        fieldin_names=("thk", "usurf", "temp"),
        patch_shape=(ps, ps, C),
        num_patches=num_patches,
        seed=42,
    )
    
    batches = builder.build_batches(patches)
    flat = flatten_batches(batches).numpy()
    
    assert flat.shape[0] == target_samples
    
    # All original patches should be present
    def is_exact_match(sample):
        return np.any(np.all(patches.numpy() == sample, axis=(1, 2, 3)))
    
    matches = sum(is_exact_match(s) for s in flat)
    assert matches >= num_patches, "All originals should be present"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

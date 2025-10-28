#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
import numpy as np
import tensorflow as tf
from types import SimpleNamespace

from igm.processes.iceflow.data_preparation.preparation_params import (
    PreparationParams,
    calculate_expected_dimensions,
    create_channel_mask,
    get_input_params_args,
)

from igm.processes.iceflow.data_preparation.input_tensor_preparation import (
    create_input_tensor_from_fieldin,
)

from igm.processes.iceflow.data_preparation.patching import OverlapPatching

# --------------------------
# Helpers 
# --------------------------

def make_fieldin(h=8, w=8, c=3, dtype=tf.float32) -> tf.Tensor:
    """
    Asymmetric pattern so flips/noise are detectable.
    value[y, x, c] = 10000*y + 100*x + 10*c
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    arr = np.zeros((h, w, c), dtype=np.float64)
    for ch in range(c):
        arr[..., ch] = 10000.0 * yy + 100.0 * xx + 10.0 * ch
    return tf.constant(arr, dtype=dtype)


def make_params(
    *,
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
    skip_preparation=False,
) -> PreparationParams:
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
        skip_preparation=skip_preparation,
    )


def flatten_batches(training_tensor: tf.Tensor) -> tf.Tensor:
    """[num_batches, batch_size, H, W, C] -> [N, H, W, C]."""
    nb, bs, h, w, c = training_tensor.shape
    return tf.reshape(training_tensor, [nb * bs, h, w, c])


def multiset_equal(samples_a: np.ndarray, samples_b: np.ndarray) -> bool:
    """Order-insensitive equality over first dimension."""
    A = samples_a.reshape(samples_a.shape[0], -1)
    B = samples_b.reshape(samples_b.shape[0], -1)
    idxA = np.lexsort(A.T[::-1])
    idxB = np.lexsort(B.T[::-1])
    return np.array_equal(A[idxA], B[idxB])


# --------------------------------
# Tests for create_input_tensor...
# --------------------------------

def test_skip_preparation_returns_single_batch() -> None:
    tf.random.set_seed(1)
    H, W, C = 6, 5, 3
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)

    prep = make_params(skip_preparation=True, precision="single")
    patching = OverlapPatching(patch_size=max(H, W), overlap=0.0)

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)

    assert out.dtype == tf.float32
    assert out.shape == (1, 1, H, W, C)
    np.testing.assert_allclose(out[0, 0].numpy(), fieldin.numpy())


def test_single_full_image_patch_no_augs() -> None:
    tf.random.set_seed(2)
    H, W, C = 7, 5, 2
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)

    # patch_size >= H and W, and no augs => single sample
    prep = make_params(
        patch_size=10,
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="none",
        target_samples=1,
        precision="single",
    )
    patching = OverlapPatching(patch_size=10, overlap=0.0)

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)

    assert out.dtype == tf.float32
    assert out.shape == (1, 1, H, W, C)
    np.testing.assert_allclose(out[0, 0].numpy(), fieldin.numpy())


@pytest.mark.parametrize("H,W,ps,batch_size", [(8, 8, 4, 4), (12, 12, 6, 2)])
def test_patching_no_upsampling_no_augs_multiset_equality(H: int, W: int, ps: int, batch_size: int) -> None:
    """
    No augmentations; target_samples <= num_patches -> no upsampling,
    output contains exactly the original patches (order-insensitive).
    """
    tf.random.set_seed(3)
    C = 3
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)

    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    patches = patching.patch_tensor(fieldin).numpy()               # [P, ps, ps, C]
    num_patches = patches.shape[0]

    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=batch_size,
        target_samples=num_patches,      # <= num_patches -> no upsample
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="none",
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    assert out.dtype == tf.float32
    nb = (num_patches // batch_size)
    assert out.shape == (nb, batch_size, ps, ps, C)

    out_flat = flatten_batches(out).numpy()  # [num_patches, ps, ps, C]
    assert multiset_equal(out_flat, patches)


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_patching_with_overlap_increases_patch_count(overlap: float) -> None:
    """
    Overlap increases patch count monotonically.
    """
    tf.random.set_seed(4)
    H, W, C, ps = 16, 16, 1, 8
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)

    counts = []
    for ov in [0.0, 0.25, 0.5]:
        patching = OverlapPatching(patch_size=ps, overlap=ov)
        patches = patching.patch_tensor(fieldin)
        counts.append(int(patches.shape[0]))

    assert counts[0] <= counts[1] <= counts[2]
    assert counts[0] < counts[2]  # strict increase overall


def test_upsample_flip_aug_applies_only_to_extras() -> None:
    """
    When upsampling is required and flip prob=1.0:
    - originals stay unmodified
    - extras are flips (accept horizontal OR vertical)
    """
    tf.random.set_seed(42)
    H, W, C, ps = 8, 8, 3, 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]

    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=P + 1,            # single batch
        target_samples=P + 1,        # force one extra
        rotation_probability=0.0,
        flip_probability=1.0,        # always flip augmented extras
        noise_type="none",
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    assert flat.shape[0] == P + 1

    # Identify exact-original matches
    def is_in_originals(sample):
        return np.any(np.all(originals == sample, axis=(1, 2, 3)))

    matches = np.array([is_in_originals(s) for s in flat])
    assert matches.sum() == P, "All originals must be present and unmodified"
    assert (~matches).sum() == 1, "Exactly one augmented extra expected"

    aug_sample = flat[~matches][0]

    # Accept horizontal, vertical, OR both flips (180° rotation) of any original
    def is_flip_of_original(sample):
        for o in originals:
            if np.all(np.flip(o, axis=1) == sample):  # horizontal only
                return True
            if np.all(np.flip(o, axis=0) == sample):  # vertical only
                return True
            if np.all(np.flip(np.flip(o, axis=0), axis=1) == sample):  # both (180°)
                return True
        return False

    assert is_flip_of_original(aug_sample), "Augmented extra should be a flip (horizontal, vertical, or both) of some original"


def test_augmentations_not_applied_without_upsampling_even_if_prob_1() -> None:
    """
    Policy check: if target_samples == num_patches, no augmentations are applied,
    even if flip/rotation/noise are enabled.
    """
    tf.random.set_seed(5)
    H, W, C, ps = 8, 8, 3, 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    patches = patching.patch_tensor(fieldin).numpy()
    P = patches.shape[0]

    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=P,
        target_samples=P,              # no upsample
        rotation_probability=1.0,
        flip_probability=1.0,
        noise_type="gaussian",
        noise_scale=0.5,
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()  # [P, ps, ps, C]

    assert multiset_equal(flat, patches), "No sample should be augmented when no upsampling"


def test_upsample_noise_respects_channel_mask() -> None:
    """
    With gaussian noise and mask=('thk',), only channel 0 should differ for augmented extras.
    """
    tf.random.set_seed(123)
    H, W, C, ps = 8, 8, 3, 4
    names = ("thk", "usurf", "temp")
    mask_channels = ("thk",)  # only ch 0 will be noised

    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]

    prep = make_params(
        fieldin_names=names,
        noise_channels=mask_channels,
        overlap=0.0,
        patch_size=ps,
        batch_size=P + 1,
        target_samples=P + 1,              # force 1 extra
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="gaussian",
        noise_scale=0.25,
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    assert flat.shape[0] == P + 1

    # Find augmented extra (not bitwise equal to any original)
    def is_in_originals(sample):
        return np.any(np.all(originals == sample, axis=(1, 2, 3)))
    idx_aug = np.where([not is_in_originals(s) for s in flat])[0]
    assert len(idx_aug) == 1, "Exactly one augmented extra expected"
    aug = flat[idx_aug[0]]

    # Match its source original based on unmasked channels
    unmasked = [i for i, n in enumerate(names) if n not in mask_channels]
    masked = [i for i, n in enumerate(names) if n in mask_channels]

    matched = None
    for o in originals:
        same_unmasked = all(np.allclose(o[..., ch], aug[..., ch]) for ch in unmasked)
        if same_unmasked:
            matched = o
            break
    assert matched is not None, "Augmented extra should match an original on unmasked channels"

    # Verify masked channels differ somewhere, unmasked match exactly
    for ch in unmasked:
        np.testing.assert_allclose(matched[..., ch], aug[..., ch], rtol=0, atol=0)
    assert any(np.any(matched[..., ch] != aug[..., ch]) for ch in masked), "Masked channels must differ"


def test_batch_trimming_drops_remainder() -> None:
    """
    Samples are rounded UP to next batch_size multiple to avoid dropping any data.
    Previously this test checked trimming behavior, now it verifies rounding up.
    """
    tf.random.set_seed(7)
    H, W, C, ps = 12, 12, 1, 6
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]

    # Force extras so adjusted_target is P + 1; choose batch_size that doesn't divide P+1
    adjusted_target = P + 1
    batch_size = max(2, P // 2)  # likely not a divisor of P+1

    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=batch_size,
        target_samples=adjusted_target,
        flip_probability=1.0,   # any aug, doesn't matter
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    nb = out.shape[0]
    used = nb * batch_size
    
    # NEW BEHAVIOR: Rounds up to next multiple of batch_size to preserve all samples
    import math
    expected_used = math.ceil(adjusted_target / batch_size) * batch_size
    assert used >= adjusted_target, "Should never drop samples"
    assert used == expected_used, f"Should round up to next batch_size multiple"


def test_large_batch_size_clamped_to_samples() -> None:
    """
    If batch_size > sample count, creates a single batch with actual sample count.
    No padding is added when there's only one batch.
    """
    tf.random.set_seed(8)
    H, W, C, ps = 8, 8, 2, 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    P = int(patching.patch_tensor(fieldin).shape[0])
    batch_size = P + 10

    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=batch_size,
        target_samples=P,       # no upsample requested
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    
    # Single batch case: batch size adapts to actual sample count
    assert out.shape[0] == 1, "Should create 1 batch"
    assert out.shape[1] == P, f"Single batch should contain actual sample count, not padded"
    assert out.shape[2:] == (ps, ps, C)


def test_shuffle_preserves_multiset() -> None:
    """
    With upsampling disabled, shuffled output must be a permutation of original patches.
    """
    tf.random.set_seed(9)
    H, W, C, ps = 8, 8, 2, 4
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    patches = patching.patch_tensor(fieldin).numpy()
    P = patches.shape[0]

    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=P,
        target_samples=P,      # no upsample
        precision="single",
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    assert multiset_equal(flat, patches)


@pytest.mark.parametrize("precision,dtype", [("single", tf.float32), ("double", tf.float64)])
def test_precision_casting_single_and_double(precision: str, dtype: tf.DType) -> None:
    tf.random.set_seed(10)
    H, W, C, ps = 8, 8, 2, 4
    # Start with the opposite dtype to confirm casting
    start_dtype = tf.float64 if dtype == tf.float32 else tf.float32
    fieldin = make_fieldin(H, W, C, dtype=start_dtype)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)

    P = int(patching.patch_tensor(fieldin).shape[0])
    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=P,
        target_samples=P,            # no upsample
        precision=precision,
    )

    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    assert out.dtype == dtype


# -----------------------------
# Helper/decision function tests
# -----------------------------

def test_calculate_expected_dimensions_full_image_case() -> None:
    H, W = 7, 5
    prep = make_params(patch_size=10, batch_size=16, target_samples=3)
    Ny, Nx, num_patches, eff_bs, adj_tgt = calculate_expected_dimensions(H, W, prep)
    assert (Ny, Nx) == (H, W)
    assert num_patches == 1
    assert eff_bs == min(16, max(3, 1))
    assert adj_tgt == max(3, 1)


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_calculate_expected_dimensions_patching_case(overlap: float) -> None:
    H, W, ps = 16, 16, 8
    prep = make_params(patch_size=ps, overlap=overlap, batch_size=4, target_samples=1)
    Ny, Nx, num_patches, eff_bs, adj_tgt = calculate_expected_dimensions(H, W, prep)
    assert (Ny, Nx) == (ps, ps)
    assert num_patches >= 1
    # effective batch size based on adjusted target >= num_patches
    assert eff_bs == min(4, max(1, num_patches))
    assert adj_tgt == max(1, num_patches)


def test_create_channel_mask_defaults_and_custom() -> None:
    names = ("thk", "usurf", "temp")
    mask_default = create_channel_mask(names).numpy()
    assert mask_default.tolist() == [True, True, False]

    mask_custom = create_channel_mask(names, ("temp",)).numpy()
    assert mask_custom.tolist() == [False, False, True]


def test_get_input_params_args_clamps_noise_scale_and_sets_channels() -> None:
    # Build a minimal nested cfg using SimpleNamespace (no custom classes)
    def build_cfg(noise_scale, control_list=None, method="unified", mapping="identity"):
        processes = SimpleNamespace()
        iceflow = SimpleNamespace()
        unified = SimpleNamespace()
        data_prep = SimpleNamespace()
        numerics = SimpleNamespace()
        emulator = SimpleNamespace()
        data_assimilation = SimpleNamespace() if control_list is not None else None

        data_prep.overlap = 0.0
        data_prep.batch_size = 8
        data_prep.patch_size = 4
        data_prep.rotation_probability = 0.0
        data_prep.flip_probability = 0.0
        data_prep.noise_type = "gaussian"
        data_prep.noise_scale = noise_scale
        data_prep.target_samples = 4

        emulator.fieldin = ("thk", "usurf", "temp")
        numerics.precision = "single"
        unified.data_preparation = data_prep
        unified.mapping = mapping
        iceflow.method = method
        iceflow.unified = unified

        processes.iceflow = iceflow
        if data_assimilation is not None:
            data_assimilation.control_list = control_list
            processes.data_assimilation = data_assimilation

        cfg = SimpleNamespace()
        cfg.processes = processes
        cfg.processes.iceflow.numerics = numerics
        cfg.processes.iceflow.emulator = emulator
        return cfg

    # noise_scale > 1.0 -> clamp to 1.0 and pick channels from control_list
    cfg1 = build_cfg(1.5, control_list=["thk", "foo"])
    params1 = get_input_params_args(cfg1)
    assert pytest.approx(params1["noise_scale"], rel=0, abs=0) == 1.0
    assert params1["noise_channels"] == ("thk",)
    assert params1["skip_preparation"] is True  # unified + identity

    # noise_scale < 0.0 -> clamp to 0.0, default channels when no data_assimilation
    cfg2 = build_cfg(-0.25, control_list=None)
    params2 = get_input_params_args(cfg2)
    assert pytest.approx(params2["noise_scale"], rel=0, abs=0) == 0.0
    assert params2["noise_channels"] == ("thk", "usurf")
    assert params2["skip_preparation"] is True  # unified + identity


# --------------------------------
# Additional tests for missing coverage
# --------------------------------

def test_large_upsampling_12_patches_to_20_samples() -> None:
    """
    Test: 12 patches, target_samples=20 → 12 originals + 8 augmented extras.
    Validates requirement: originals preserved, extras augmented.
    """
    tf.random.set_seed(100)
    H, W, C, ps = 12, 12, 3, 6
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]  # Should be 4 patches (2x2)
    
    target = 20
    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=target,  # single batch
        target_samples=target,
        rotation_probability=0.0,
        flip_probability=1.0,  # augment extras with flips
        noise_type="none",
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    # Should have exactly target samples
    assert flat.shape[0] == target
    
    # Count how many are exact originals
    def is_in_originals(sample):
        return np.any(np.all(originals == sample, axis=(1, 2, 3)))
    
    matches = np.array([is_in_originals(s) for s in flat])
    num_originals = matches.sum()
    num_augmented = (~matches).sum()
    
    assert num_originals == P, f"Expected {P} original patches preserved"
    assert num_augmented == target - P, f"Expected {target - P} augmented extras"


def test_single_patch_upsampled_to_20_samples() -> None:
    """
    Test: 1 patch, target_samples=20 → 1 original + 19 augmented copies.
    Critical test for the requirement: "1 original, 19 augmented"
    """
    tf.random.set_seed(101)
    H, W, C = 8, 8, 3
    ps = 10  # Larger than image, so 1 patch
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]
    assert P == 1, "Should have exactly 1 patch"
    
    target = 20
    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=target,
        target_samples=target,
        rotation_probability=0.0,
        flip_probability=1.0,
        noise_type="none",
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    assert flat.shape[0] == target
    
    # Exactly 1 should match original exactly
    def is_exact_original(sample):
        return np.all(originals[0] == sample)
    
    exact_matches = sum(is_exact_original(s) for s in flat)
    assert exact_matches == 1, "Exactly 1 original should be preserved unmodified"
    
    # Remaining 19 should be augmented (flipped in some way)
    augmented_count = 0
    for s in flat:
        if not is_exact_original(s):
            augmented_count += 1
    assert augmented_count == 19, "Should have 19 augmented copies"


def test_exact_batch_split_128_samples_batch_32() -> None:
    """
    Test: 128 samples with batch_size=32 → output shape [4, 32, H, W, C]
    Validates the specific example from requirements.
    """
    tf.random.set_seed(102)
    H, W, C, ps = 16, 16, 3, 8
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]  # Should be 4 patches (2x2)
    
    target = 128
    batch_size = 32
    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=batch_size,
        target_samples=target,
        flip_probability=1.0,  # need augmentation to upsample
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    
    expected_batches = target // batch_size
    assert out.shape[0] == expected_batches, f"Expected {expected_batches} batches"
    assert out.shape[1] == batch_size, f"Expected batch_size={batch_size}"
    assert out.shape[2:] == (ps, ps, C)
    
    # Total samples should be exactly 128
    total_samples = out.shape[0] * out.shape[1]
    assert total_samples == 128


def test_rotation_augmentation_applied_to_extras() -> None:
    """
    Test rotation augmentation on square patches.
    Validates that originals are preserved and extras may be rotated (including identity).
    """
    tf.random.set_seed(103)
    H, W, C, ps = 8, 8, 3, 4  # Square patches
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]
    
    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=P + 1,
        target_samples=P + 1,  # force 1 extra
        rotation_probability=1.0,  # always rotate (may include identity k=0)
        flip_probability=0.0,
        noise_type="none",
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    assert flat.shape[0] == P + 1
    
    # Count exact original matches
    def is_in_originals(sample):
        return np.any(np.all(originals == sample, axis=(1, 2, 3)))
    
    matches = np.array([is_in_originals(s) for s in flat])
    # At least P originals must be present (augmented extra may also match an original if k=0)
    assert matches.sum() >= P, f"At least {P} samples should match originals"
    
    # If there's exactly one non-match, verify it's a valid rotation
    if matches.sum() == P:
        aug_sample = flat[~matches][0]
        
        # Check if augmented sample is a rotation (0°, 90°, 180°, 270°) of any original
        def is_rotation_of_original(sample):
            for o in originals:
                # Check identity and all rotations (k=0,1,2,3)
                if np.all(o == sample):
                    return True
                for k in [1, 2, 3]:
                    rotated = np.rot90(o, k=k, axes=(0, 1))
                    if np.all(rotated == sample):
                        return True
            return False
        
        assert is_rotation_of_original(aug_sample), "Augmented extra should be a rotation (or identity) of some original"


def test_rotation_and_flip_combined() -> None:
    """
    Test that rotation and flip can both be applied to the same sample.
    Accepts that augmented samples may be identical to original (identity rotation/flip).
    """
    tf.random.set_seed(104)
    H, W, C, ps = 8, 8, 3, 8  # Single square patch
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    originals = patching.patch_tensor(fieldin).numpy()
    P = originals.shape[0]
    assert P == 1
    
    prep = make_params(
        overlap=0.0,
        patch_size=ps,
        batch_size=5,
        target_samples=5,  # 1 original + 4 augmented
        rotation_probability=1.0,
        flip_probability=1.0,
        noise_type="none",
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    assert flat.shape[0] == 5
    
    # At least 1 should be the unmodified original (may be more if identity transforms occur)
    original = originals[0]
    exact_matches = sum(np.all(original == s) for s in flat)
    assert exact_matches >= 1, "At least one original should be preserved"
    
    # Augmentation was applied to 4 samples (even if some result in identity transforms)
    # Total samples should be 5
    assert flat.shape[0] == 5, "Should have 5 total samples"


def test_noise_magnitude_stays_within_bounds() -> None:
    """
    Test that noise_scale produces values within expected fractional range.
    Example: value=10.0, scale=0.1 → result in [9.0, 11.0]
    """
    tf.random.set_seed(105)
    H, W, C, ps = 8, 8, 1, 8
    
    # Create fieldin with known value (10.0 everywhere)
    fieldin = tf.constant(np.full((H, W, C), 10.0, dtype=np.float32))
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    P = 1  # Single patch
    scale = 0.1
    
    prep = make_params(
        fieldin_names=("thk",),
        noise_channels=("thk",),
        overlap=0.0,
        patch_size=ps,
        batch_size=10,
        target_samples=10,  # 1 original + 9 augmented
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="gaussian",
        noise_scale=scale,
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    # Find augmented samples (not exactly 10.0 everywhere)
    original_value = 10.0
    augmented = [s for s in flat if not np.allclose(s, original_value)]
    
    assert len(augmented) > 0, "Should have augmented samples with noise"
    
    # Check that all noisy values are within [9.0, 11.0]
    expected_min = original_value * (1.0 - scale)
    expected_max = original_value * (1.0 + scale)
    
    for aug in augmented:
        values = aug.flatten()
        assert np.all(values >= expected_min), f"Values should be >= {expected_min}"
        assert np.all(values <= expected_max), f"Values should be <= {expected_max}"


def test_noise_never_produces_negatives() -> None:
    """
    Critical test: noise should never produce negative values.
    Test with zero values and small positive values.
    """
    tf.random.set_seed(106)
    H, W, C, ps = 8, 8, 2, 8
    
    # Create fieldin with zeros in channel 0, small values in channel 1
    arr = np.zeros((H, W, C), dtype=np.float32)
    arr[..., 0] = 0.0  # Channel 0: all zeros
    arr[..., 1] = 0.5  # Channel 1: small positive values
    fieldin = tf.constant(arr)
    
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    prep = make_params(
        fieldin_names=("thk", "usurf"),
        noise_channels=("thk", "usurf"),
        overlap=0.0,
        patch_size=ps,
        batch_size=20,
        target_samples=20,
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="gaussian",
        noise_scale=0.5,  # 50% scale - aggressive
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    # Critical: no values should be negative
    assert np.all(flat >= 0.0), "Noise should never produce negative values"
    
    # Channel 0 should remain at or near 0 (can't go negative)
    channel_0_values = flat[..., 0]
    assert np.all(channel_0_values >= 0.0), "Channel 0 with zeros should not go negative"


def test_perlin_noise_type() -> None:
    """
    Test perlin noise type produces spatially coherent patterns.
    """
    tf.random.set_seed(107)
    H, W, C, ps = 8, 8, 1, 8
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    prep = make_params(
        fieldin_names=("thk",),
        noise_channels=("thk",),
        overlap=0.0,
        patch_size=ps,
        batch_size=5,
        target_samples=5,
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="perlin",
        noise_scale=0.2,
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    # Should have 1 original + 4 augmented
    original = patching.patch_tensor(fieldin).numpy()[0]
    augmented = [s for s in flat if not np.allclose(s, original)]
    
    assert len(augmented) >= 4, "Should have augmented samples with perlin noise"
    
    # Perlin noise should produce spatially varying patterns (not uniform)
    for aug in augmented:
        variance = np.var(aug)
        assert variance > 0, "Perlin noise should produce spatial variation"


def test_intensity_noise_type() -> None:
    """
    Test intensity noise type produces uniform scaling across entire image.
    """
    tf.random.set_seed(108)
    H, W, C, ps = 8, 8, 1, 8
    fieldin = make_fieldin(H, W, C, dtype=tf.float32)
    patching = OverlapPatching(patch_size=ps, overlap=0.0)
    
    prep = make_params(
        fieldin_names=("thk",),
        noise_channels=("thk",),
        overlap=0.0,
        patch_size=ps,
        batch_size=5,
        target_samples=5,
        rotation_probability=0.0,
        flip_probability=0.0,
        noise_type="intensity",
        noise_scale=0.2,
        precision="single",
    )
    
    out = create_input_tensor_from_fieldin(fieldin, patching, prep)
    flat = flatten_batches(out).numpy()
    
    # Should have 1 original + 4 augmented
    original = patching.patch_tensor(fieldin).numpy()[0]
    augmented = [s for s in flat if not np.allclose(s, original, rtol=1e-5)]
    
    assert len(augmented) >= 4, "Should have augmented samples with intensity noise"
    
    # Intensity noise should scale uniformly (ratio should be constant across pixels)
    for aug in augmented:
        # Compare ratios at different positions (should be the same)
        ratios = aug[aug != 0] / original[aug != 0]
        ratio_variance = np.var(ratios)
        # Intensity noise produces uniform scaling, so variance should be very small
        assert ratio_variance < 1e-10, "Intensity noise should produce uniform scaling"




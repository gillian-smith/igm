#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Unit tests for patching module (OverlapPatching and GridPatching).

These tests verify:
- Correct patch dimensions and counts
- Full coverage of input images
- Overlap approximation behavior
- Edge cases (images smaller/larger than patch_size)
- Padding minimization
- Non-square image handling
"""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.iceflow.data_preparation.patching import (
    Patching,
    OverlapPatching,
)


# --------------------------
# Test Helpers
# --------------------------


def create_test_image(height: int, width: int, channels: int = 3) -> tf.Tensor:
    """
    Create a test image with unique pixel IDs at each spatial location.

    This allows verification that all pixels are covered by patches.
    Each pixel gets a unique ID: pixel_id = y * width + x
    All channels get the same ID for simplicity.
    """
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    # Unique pixel ID for each (y, x) location
    pixel_ids = yy * width + xx
    # Broadcast to all channels
    arr = np.repeat(pixel_ids[:, :, np.newaxis], channels, axis=2).astype(
        np.float32
    )
    return tf.constant(arr, dtype=tf.float32)


def verify_full_coverage(original: tf.Tensor, patches: tf.Tensor) -> bool:
    """
    Verify that every pixel in the original image appears in at least one patch.

    This works by checking that all unique pixel IDs from the original image
    appear in the extracted patches. This is more robust than geometric
    calculations as it verifies the actual patch contents.

    Args:
        original: Original image tensor (H, W, C) with unique pixel IDs
        patches: Extracted patches (N, patch_size, patch_size, C)

    Returns:
        True if every pixel is covered, False otherwise
    """
    # Get all unique pixel IDs from original image (using first channel)
    original_ids = set(original.numpy()[:, :, 0].flatten())

    # Get all unique pixel IDs from all patches (using first channel)
    patch_ids = set(patches.numpy()[:, :, :, 0].flatten())

    # Check if all original IDs appear in patches
    return original_ids.issubset(patch_ids)


def calculate_actual_overlap(
    height: int,
    width: int,
    patch_size: int,
    n_patches_y: int,
    n_patches_x: int,
    stride_y: int,
    stride_x: int,
) -> tuple[float, float]:
    """
    Calculate actual overlap fractions achieved in each dimension.

    Returns:
        Tuple of (overlap_y, overlap_x) as fractions
    """
    if n_patches_y > 1 and stride_y > 0:
        overlap_y = (patch_size - stride_y) / patch_size
    else:
        overlap_y = 0.0

    if n_patches_x > 1 and stride_x > 0:
        overlap_x = (patch_size - stride_x) / patch_size
    else:
        overlap_x = 0.0

    return overlap_y, overlap_x


# Helper: generate patches using the constructor auto-initialization
def generate_patches_for_image(
    patcher: OverlapPatching, image: tf.Tensor
) -> tf.Tensor:
    """
    Convenience helper that expects patcher to already be initialized.
    With the new API, patcher should be constructed with fieldin parameter
    for auto-initialization, so this just calls generate_patches.
    """
    return patcher.generate_patches(image)


# --------------------------
# OverlapPatching Tests
# --------------------------


class TestOverlapPatching:
    """Tests for OverlapPatching class."""

    def test_initialization_valid_overlap(self):
        """Test that valid overlap values are accepted."""
        for overlap in [0.0, 0.25, 0.5, 0.75, 0.99]:
            patcher = OverlapPatching(patch_size=64, overlap=overlap)
            assert patcher.target_overlap == overlap
            assert patcher.patch_size == 64

    def test_initialization_invalid_overlap(self):
        """Test that invalid overlap values raise errors."""
        with pytest.raises(ValueError):
            OverlapPatching(patch_size=64, overlap=-0.1)

        with pytest.raises(ValueError):
            OverlapPatching(patch_size=64, overlap=1.0)

        with pytest.raises(ValueError):
            OverlapPatching(patch_size=64, overlap=1.5)

    def test_initialize_for_field_sets_metadata(self):
        """
        New: test that initialize_for_field correctly sets num_patches and patch_shape.
        """
        patcher = OverlapPatching(patch_size=64, overlap=0.25)
        image = create_test_image(256, 256, channels=3)

        patcher.initialize_for_field(image)

        # num_patches should be consistent with internal layout
        assert patcher.num_patches == patcher._n_patches_y * patcher._n_patches_x
        # patch_shape should be (patch_size, patch_size, C) in this case
        assert patcher.patch_shape == (64, 64, 3)

    def test_image_smaller_than_patch_size(self):
        """
        Test that images smaller than patch_size return original dimensions.

        A 250x150 image with patch_size=400 should return a single patch
        with the original 250x150 dimensions (not 400x400) when using
        generate_patches().
        """
        image = create_test_image(250, 150, channels=3)
        patcher = OverlapPatching(patch_size=400, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        # Should return single patch with original dimensions
        assert patches.shape[0] == 1, "Should return exactly one patch"
        assert patches.shape[1] == 250, "Height should match original"
        assert patches.shape[2] == 150, "Width should match original"
        assert patches.shape[3] == 3, "Channels should match original"

        # Verify content is identical
        np.testing.assert_array_equal(patches[0].numpy(), image.numpy())

    def test_image_larger_creates_multiple_patches(self):
        """
        Test that images larger than patch_size are split into multiple patches.

        A 250x150 image with patch_size=64 should create several 64x64 patches.
        """
        image = create_test_image(250, 150, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        # Should create multiple patches
        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64, "Patch height should be 64"
        assert patches.shape[2] == 64, "Patch width should be 64"
        assert patches.shape[3] == 3, "Channels should be preserved"

    def test_full_coverage_small_patch_size(self):
        """
        Test that every pixel is covered when using small patch_size.

        This verifies the fundamental requirement that no pixels are missed.
        """
        image = create_test_image(250, 150, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        # Verify full coverage
        is_covered = verify_full_coverage(image, patches)

        assert is_covered, "All pixels must be covered by at least one patch"

    def test_high_overlap_near_complete(self):
        """
        Test edge case with patch_size close to image dimension.

        A 250x150 image with patch_size=149 should result in 4 patches
        (2x2 grid) with very high overlap in one dimension.
        """
        image = create_test_image(250, 150, channels=3)
        patcher = OverlapPatching(patch_size=149, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        # Should create 2x2 grid = 4 patches
        assert patches.shape[0] >= 4, "Should create at least 4 patches for coverage"
        assert patches.shape[1] == 149, "Patch size should be 149"
        assert patches.shape[2] == 149, "Patch size should be 149"

        # Verify full coverage despite near-complete overlap
        is_covered = verify_full_coverage(image, patches)

        assert is_covered, "Must maintain full coverage even with high overlap"

    def test_overlap_approximation_accuracy(self):
        """
        Test that actual overlap approximates target overlap reasonably well.

        The adaptive algorithm should get close to the target overlap while
        maintaining full coverage.

        Uses the layout cached by initialize_for_field instead of calling
        the old _calculate_patching_parameters method.
        """
        target_overlap = 0.25
        patch_size = 64
        image = create_test_image(256, 256, channels=3)  # Nice divisible size
        patcher = OverlapPatching(patch_size=patch_size, overlap=target_overlap, fieldin=image)

        # Generate patches (patcher already initialized via constructor)
        patches = patcher.generate_patches(image)

        # Calculate actual overlap achieved using cached layout
        height, width = 256, 256
        n_patches_y = patcher._n_patches_y
        n_patches_x = patcher._n_patches_x
        stride_y = patcher._stride_y
        stride_x = patcher._stride_x

        overlap_y, overlap_x = calculate_actual_overlap(
            height,
            width,
            patch_size,
            n_patches_y,
            n_patches_x,
            stride_y,
            stride_x,
        )

        avg_overlap = (overlap_y + overlap_x) / 2.0

        # Should be reasonably close to target (within 0.15)
        assert abs(avg_overlap - target_overlap) < 0.15, (
            f"Average overlap {avg_overlap:.3f} should approximate "
            f"target {target_overlap} (within 0.15)"
        )

        # Sanity: patches really exist
        assert patches.shape[0] == n_patches_y * n_patches_x

    def test_various_overlap_values(self):
        """Test that different overlap values work correctly."""
        image = create_test_image(200, 200, channels=3)

        for target_overlap in [0.0, 0.25, 0.5, 0.75]:
            patcher = OverlapPatching(patch_size=64, overlap=target_overlap, fieldin=image)
            patches = generate_patches_for_image(patcher, image)

            # Should create patches
            assert (
                patches.shape[0] > 0
            ), f"Should create patches for overlap={target_overlap}"
            assert patches.shape[1] == 64
            assert patches.shape[2] == 64

            # Verify coverage
            is_covered = verify_full_coverage(image, patches)

            assert is_covered, f"Full coverage required for overlap={target_overlap}"

    def test_non_square_image_tall(self):
        """Test patching on tall non-square images."""
        image = create_test_image(400, 100, channels=3)  # Tall image
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64
        assert patches.shape[2] == 64

        # Verify coverage
        is_covered = verify_full_coverage(image, patches)

        assert is_covered, "Full coverage required for tall images"

    def test_non_square_image_wide(self):
        """Test patching on wide non-square images."""
        image = create_test_image(100, 400, channels=3)  # Wide image
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64
        assert patches.shape[2] == 64

        # Verify coverage
        is_covered = verify_full_coverage(image, patches)

        assert is_covered, "Full coverage required for wide images"

    def test_square_image(self):
        """Test patching on square images."""
        image = create_test_image(256, 256, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64
        assert patches.shape[2] == 64

        # Verify symmetric patching via cached layout (already initialized)
        n_patches_y = patcher._n_patches_y
        n_patches_x = patcher._n_patches_x

        # Square image should have same number of patches in each dimension
        assert (
            n_patches_y == n_patches_x
        ), "Square image should have symmetric patching"

    def test_minimal_padding(self):
        """
        Test that adaptive algorithm minimizes padding.

        The algorithm should prefer configurations with less padding.
        """
        image = create_test_image(200, 200, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        height, width = 200, 200
        padding_h = patcher._padding_h
        padding_w = patcher._padding_w

        # Padding should be relatively small compared to image size
        total_padding = padding_h + padding_w
        image_pixels = height + width

        assert total_padding < image_pixels * 0.1, (
            f"Padding {total_padding} should be < 10% of image perimeter {image_pixels}"
        )

    def test_zero_overlap(self):
        """Test that zero overlap works correctly (minimal overlap patches)."""
        image = create_test_image(256, 256, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.0, fieldin=image)

        patches = patcher.generate_patches(image)

        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64
        assert patches.shape[2] == 64

        # With zero overlap, should have minimal or no overlap
        height, width = 256, 256
        n_patches_y = patcher._n_patches_y
        n_patches_x = patcher._n_patches_x
        stride_y = patcher._stride_y
        stride_x = patcher._stride_x
        overlap_y, overlap_x = calculate_actual_overlap(
            height,
            width,
            64,
            n_patches_y,
            n_patches_x,
            stride_y,
            stride_x,
        )

        avg_overlap = (overlap_y + overlap_x) / 2.0

        # Should be very close to zero
        assert avg_overlap < 0.1, f"Overlap {avg_overlap:.3f} should be minimal"

    def test_single_patch_dimension(self):
        """
        Test image where one dimension needs multiple patches but other doesn't.

        E.g., 500x50 with patch_size=64 should patch only along height.
        """
        image = create_test_image(500, 50, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64
        assert patches.shape[2] == 64

        # Verify coverage
        is_covered = verify_full_coverage(image, patches)

        assert is_covered, "Full coverage required for asymmetric images"

    def test_exact_divisible_dimension(self):
        """
        Test image where dimensions are exactly divisible by patch_size.

        Should result in clean patching with predictable overlap.
        """
        image = create_test_image(256, 192, channels=3)  # 256 = 4*64, 192 = 3*64
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        assert patches.shape[0] > 1, "Should create multiple patches"
        assert patches.shape[1] == 64
        assert patches.shape[2] == 64

        # Verify coverage
        is_covered = verify_full_coverage(image, patches)

        assert is_covered, "Full coverage required for divisible dimensions"

    def test_channels_preserved(self):
        """Test that different channel counts are preserved correctly."""
        for n_channels in [1, 3, 5, 10]:
            image = create_test_image(200, 200, channels=n_channels)
            patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

            patches = generate_patches_for_image(patcher, image)

            assert patches.shape[3] == n_channels, (
                f"Channels should be preserved: expected {n_channels}, "
                f"got {patches.shape[3]}"
            )

    def test_very_small_image(self):
        """Test with very small images (smaller than typical patch_size)."""
        image = create_test_image(10, 15, channels=3)
        patcher = OverlapPatching(patch_size=64, overlap=0.25, fieldin=image)

        patches = generate_patches_for_image(patcher, image)

        # Should return single patch with original dimensions
        assert patches.shape[0] == 1
        assert patches.shape[1] == 10
        assert patches.shape[2] == 15
        assert patches.shape[3] == 3


# --------------------------
# GridPatching Tests
# --------------------------


# class TestGridPatching:
#     """Tests for GridPatching class."""

#     def test_initialization(self):
#         """Test that GridPatching initializes correctly."""
#         patcher = GridPatching(patch_size=64)
#         assert patcher.patch_size == 64

#     def test_basic_grid_division(self):
#         """Test that grid patching creates non-overlapping patches."""
#         patcher = GridPatching(patch_size=64)
#         image = create_test_image(256, 256, channels=3)

#         patches = patcher.patch_tensor(image)

#         # Should create multiple patches
#         assert patches.shape[0] > 1, "Should create multiple patches"
#         assert patches.shape[3] == 3, "Channels should be preserved"

#         # Patches may not be exactly patch_size (grid division)
#         assert patches.shape[1] > 0, "Patch height should be positive"
#         assert patches.shape[2] > 0, "Patch width should be positive"

#     def test_non_divisible_dimensions(self):
#         """
#         Test grid patching with dimensions not divisible by patch_size.

#         The algorithm should handle remainder pixels appropriately.
#         """
#         patcher = GridPatching(patch_size=64)
#         image = create_test_image(250, 150, channels=3)

#         patches = patcher.patch_tensor(image)

#         assert patches.shape[0] > 1, "Should create multiple patches"
#         assert patches.shape[3] == 3, "Channels should be preserved"

#     def test_image_smaller_than_patch_size(self):
#         """Test grid patching when image is smaller than patch_size."""
#         patcher = GridPatching(patch_size=100)
#         image = create_test_image(50, 60, channels=3)

#         patches = patcher.patch_tensor(image)

#         # Should still create patches (grid division)
#         assert patches.shape[0] > 0, "Should create at least one patch"
#         assert patches.shape[3] == 3, "Channels should be preserved"

#     def test_no_overlap_between_patches(self):
#         """
#         Verify that grid patches don't overlap.

#         Each pixel should appear in exactly one patch.
#         """
#         patcher = GridPatching(patch_size=64)
#         image = create_test_image(128, 128, channels=3)

#         patches = patcher.patch_tensor(image)

#         # For grid patching with no overlap, verify using pixel IDs
#         # Each unique pixel ID should appear exactly once across all patches
#         patch_ids = patches.numpy()[:, :, :, 0].flatten()
#         unique_ids, counts = np.unique(patch_ids, return_counts=True)

#         # All pixels should appear exactly once (no overlap)
#         assert np.all(
#             counts == 1
#         ), "Grid patches should not overlap - each pixel should appear exactly once"

#     def test_channels_preserved(self):
#         """Test that different channel counts are preserved correctly."""
#         for n_channels in [1, 3, 5, 10]:
#             patcher = GridPatching(patch_size=64)
#             image = create_test_image(200, 200, channels=n_channels)

#             patches = patcher.patch_tensor(image)

#             assert patches.shape[3] == n_channels, (
#                 f"Channels should be preserved: expected {n_channels}, "
#                 f"got {patches.shape[3]}"
#             )

#     def test_square_image(self):
#         """Test grid patching on square images."""
#         patcher = GridPatching(patch_size=64)
#         image = create_test_image(256, 256, channels=3)

#         patches = patcher.patch_tensor(image)

#         assert patches.shape[0] > 1, "Should create multiple patches"

#         # For square images with grid patching, verify symmetry by checking
#         # that we can arrange patches into a square grid
#         num_patches = patches.shape[0]

#         # Should be a perfect square number of patches for square images
#         sqrt_patches = int(np.sqrt(num_patches))
#         assert sqrt_patches * sqrt_patches == num_patches, (
#             f"Square image should produce square grid of patches, "
#             f"but got {num_patches} patches (sqrt={sqrt_patches})"
#         )

#     def test_non_square_image(self):
#         """Test grid patching on non-square images."""
#         patcher = GridPatching(patch_size=64)
#         image = create_test_image(200, 300, channels=3)

#         patches = patcher.patch_tensor(image)

#         assert patches.shape[0] > 1, "Should create multiple patches"
#         assert patches.shape[3] == 3, "Channels should be preserved"


# --------------------------
# Base Class Tests
# --------------------------


class TestPatchingBase:
    """Tests for Patching base class functionality."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Patching abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Patching(patch_size=64)

    def test_validate_input_valid_tensor(self):
        """Test input validation accepts valid 3D tensors."""
        patcher = OverlapPatching(patch_size=64, overlap=0.25)
        image = create_test_image(100, 100, channels=3)

        # Should not raise any errors
        patcher._validate_input(image)

    def test_validate_input_invalid_rank(self):
        """Test that input validation rejects non-3D tensors."""
        patcher = OverlapPatching(patch_size=64, overlap=0.25)

        # 2D tensor (missing channels)
        invalid_2d = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
            patcher._validate_input(invalid_2d)

        # 4D tensor (has batch dimension)
        invalid_4d = tf.constant([[[[1.0]]]])
        with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
            patcher._validate_input(invalid_4d)

    def test_get_patch_dimensions(self):
        """Test that dimension extraction works correctly."""
        patcher = OverlapPatching(patch_size=64, overlap=0.25)
        image = create_test_image(123, 456, channels=7)

        height, width = patcher._get_patch_dimensions(image)

        assert height.numpy() == 123
        assert width.numpy() == 456

    def test_extract_patch(self):
        """Test single patch extraction."""
        patcher = OverlapPatching(patch_size=64, overlap=0.25)
        image = create_test_image(200, 200, channels=3)

        # Extract patch from top-left corner
        patch = patcher._extract_patch(image, tf.constant(0), tf.constant(0))

        assert patch.shape == (64, 64, 3)

        # Verify content matches original
        np.testing.assert_array_equal(patch.numpy(), image.numpy()[:64, :64, :])

    def test_generate_patches_small_and_large(self):
        """
        New: smoke-test generate_patches for both small and large images.

        Ensures the base lifecycle (constructor auto-initialization + generate_patches)
        is consistent with num_patches/patch_shape.
        """
        # Small image: no patching, single patch, original dimensions
        img_small = create_test_image(10, 15, channels=3)
        patcher_small = OverlapPatching(patch_size=64, overlap=0.25, fieldin=img_small)
        patches_small = patcher_small.generate_patches(img_small)

        assert patcher_small.num_patches == 1
        assert patcher_small.patch_shape == (10, 15, 3)
        assert patches_small.shape == (1, 10, 15, 3)

        # Larger image: multiple patches, patch_shape equals (P, P, C)
        img_large = create_test_image(200, 200, channels=3)
        patcher_large = OverlapPatching(patch_size=64, overlap=0.25, fieldin=img_large)
        patches_large = patcher_large.generate_patches(img_large)

        assert patches_large.shape[0] == patcher_large.num_patches
        assert patches_large.shape[1:] == patcher_large.patch_shape
        assert patcher_large.patch_shape == (64, 64, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

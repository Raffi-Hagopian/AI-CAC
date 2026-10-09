import unittest

import numpy as np

from array_utils import center_crop_or_pad


def legacy_top_left(array, target_shape, dtype):
    # Behaviour of the loaders before issue #5 was fixed.
    output = np.zeros((*target_shape, *array.shape[2:]), dtype=dtype)
    output[: array.shape[0], : array.shape[1]] = array[
        : target_shape[0], : target_shape[1]
    ]
    return output


class CenterCropOrPadTests(unittest.TestCase):
    def test_center_crops_larger_array(self):
        array = np.arange(824 * 824, dtype=np.int32).reshape(824, 824)

        result = center_crop_or_pad(array, (512, 512))

        self.assertEqual(result.shape, (512, 512))
        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, array[156:668, 156:668])
        # Regression guard: the old loaders kept the top-left corner.
        self.assertFalse(
            np.array_equal(result, legacy_top_left(array, (512, 512), np.int32))
        )

    def test_center_pads_smaller_array(self):
        array = np.ones((256, 256), dtype=np.float32)

        result = center_crop_or_pad(array, (512, 512))

        expected = np.zeros((512, 512), dtype=np.float32)
        expected[128:384, 128:384] = array
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.sum(), 256 * 256)

    def test_crops_and_pads_independent_dimensions(self):
        array = np.arange(824 * 256, dtype=np.int32).reshape(824, 256)

        result = center_crop_or_pad(array, (512, 512))

        expected = np.zeros((512, 512), dtype=np.int32)
        expected[:, 128:384] = array[156:668, :]
        np.testing.assert_array_equal(result, expected)

        transposed = center_crop_or_pad(array.T, (512, 512))
        np.testing.assert_array_equal(transposed, expected.T)

    def test_non_square_target(self):
        array = np.arange(10 * 10).reshape(10, 10)

        result = center_crop_or_pad(array, (4, 8))

        np.testing.assert_array_equal(result, array[3:7, 1:9])

    def test_odd_difference_puts_extra_row_and_column_bottom_right(self):
        array = np.arange(1, 7 * 9 + 1, dtype=np.int16).reshape(7, 9)

        cropped = center_crop_or_pad(array, (4, 6))
        padded = center_crop_or_pad(array, (10, 12))

        # 7 -> 4 drops 1 row on top and 2 on the bottom; 9 -> 6 drops 1 left, 2 right.
        np.testing.assert_array_equal(cropped, array[1:5, 1:7])
        # 7 -> 10 pads 1 row on top and 2 on the bottom; 9 -> 12 pads 1 left, 2 right.
        expected = np.zeros((10, 12), dtype=np.int16)
        expected[1:8, 1:10] = array
        np.testing.assert_array_equal(padded, expected)
        self.assertTrue((padded[8:, :] == 0).all() and (padded[:, 10:] == 0).all())

    def test_identity_returns_equal_copy(self):
        array = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)

        result = center_crop_or_pad(array, (512, 512))

        np.testing.assert_array_equal(result, array)
        self.assertEqual(result.dtype, array.dtype)
        self.assertFalse(np.shares_memory(result, array))

    def test_three_dimensional_array_keeps_every_z_slice(self):
        array = np.arange(8 * 6 * 3, dtype=np.int16).reshape(8, 6, 3)

        result = center_crop_or_pad(array, (4, 4), dtype=float)

        self.assertEqual(result.shape, (4, 4, 3))
        self.assertEqual(result.dtype, np.float64)
        for z in range(array.shape[2]):
            np.testing.assert_array_equal(result[:, :, z], array[2:6, 1:5, z])

    def test_four_dimensional_array_keeps_trailing_dimensions(self):
        array = np.arange(5 * 5 * 2 * 3).reshape(5, 5, 2, 3)

        result = center_crop_or_pad(array, (3, 7))

        self.assertEqual(result.shape, (3, 7, 2, 3))
        np.testing.assert_array_equal(result[:, 1:6], array[1:4])

    def test_image_and_label_marker_lands_at_expected_position(self):
        image = np.zeros((9, 7), dtype=np.int16)
        label = np.zeros((9, 7))
        image[5, 4] = 130
        label[5, 4] = 1.0

        # Same dtypes as the training loader.
        centered_image = center_crop_or_pad(image, (6, 10), dtype=float)
        centered_label = center_crop_or_pad(label, (6, 10), dtype=int)

        # Rows 9 -> 6 start at 1; columns 7 -> 10 are padded by 1.
        self.assertEqual(list(zip(*np.nonzero(centered_image))), [(4, 5)])
        self.assertEqual(list(zip(*np.nonzero(centered_label))), [(4, 5)])
        self.assertEqual(centered_label.dtype, np.dtype(int))

    def test_dtype_argument_matches_legacy_assignment_cast(self):
        label = np.array([[0.0, 1.0, 2.7, -1.5]] * 4)
        image = np.arange(16, dtype=np.int16).reshape(4, 4) - 1024

        np.testing.assert_array_equal(
            center_crop_or_pad(label, (4, 4), dtype=int),
            legacy_top_left(label, (4, 4), int),
        )
        result = center_crop_or_pad(image, (4, 4), dtype=float)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, image.astype(float))

    def test_bool_input_and_read_only_input(self):
        array = np.ones((3, 3), dtype=bool)
        array.setflags(write=False)

        result = center_crop_or_pad(array, (5, 5))

        self.assertEqual(result.dtype, bool)
        self.assertTrue(result.flags.writeable)
        self.assertEqual(int(result.sum()), 9)

    def test_does_not_modify_input(self):
        array = np.arange(20 * 20, dtype=np.int16).reshape(20, 20)
        original = array.copy()

        center_crop_or_pad(array, (10, 30))

        np.testing.assert_array_equal(array, original)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            center_crop_or_pad(np.zeros(4), (2, 2))
        with self.assertRaises(ValueError):
            center_crop_or_pad(np.zeros((2, 2)), (2, 0))
        with self.assertRaises(ValueError):
            center_crop_or_pad(np.zeros((2, 2)), (2, 2, 2))
        with self.assertRaises(TypeError):
            center_crop_or_pad(np.zeros((2, 2)), (2.0, 2))


if __name__ == "__main__":
    unittest.main()

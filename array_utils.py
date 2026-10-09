import numpy as np


def center_crop_or_pad(array, target_shape, dtype=None):
    """Center-crop or zero-pad the first two (H, W) dimensions of an array.

    Each of H and W is handled independently: it is center-cropped when
    larger than the target and zero-padded on both sides when smaller.
    When the size difference is odd, the extra row/column is cropped or
    padded on the bottom/right. Any trailing dimensions (e.g. Z) are kept
    unchanged. The result has ``dtype`` if given, otherwise the input dtype.
    See issue #5.
    """
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("array must have at least two dimensions")
    if len(target_shape) != 2 or any(size <= 0 for size in target_shape):
        raise ValueError("target_shape must contain two positive dimensions")

    target_height, target_width = target_shape
    source_height, source_width = array.shape[:2]
    copy_height = min(source_height, target_height)
    copy_width = min(source_width, target_width)

    source_start_height = max((source_height - target_height) // 2, 0)
    source_start_width = max((source_width - target_width) // 2, 0)
    target_start_height = max((target_height - source_height) // 2, 0)
    target_start_width = max((target_width - source_width) // 2, 0)

    output = np.zeros(
        (target_height, target_width, *array.shape[2:]),
        dtype=array.dtype if dtype is None else dtype,
    )
    output[
        target_start_height : target_start_height + copy_height,
        target_start_width : target_start_width + copy_width,
        ...,
    ] = array[
        source_start_height : source_start_height + copy_height,
        source_start_width : source_start_width + copy_width,
        ...,
    ]
    return output

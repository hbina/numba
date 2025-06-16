import numpy as np
import numba


@numba.jit
def pad_img1(img, padding):
    if padding == 0:
        return img

    new_img = np.ones(
        (img.shape[0] + 2 * padding, img.shape[1] + 2 * padding, img.shape[2]),
        dtype=np.uint8,
    )
    new_img *= 255

    new_img[padding:-padding, padding:-padding, :] = img

    return new_img


first = pad_img1.py_func(np.ones((10, 10, 3), dtype=np.uint8), 1)
print(first)
second = pad_img1(np.ones((10, 10, 3), dtype=np.uint8), 1)
print(second)

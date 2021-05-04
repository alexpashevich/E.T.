import numpy as np
import alfred.gen.constants as constants

from PIL import Image
from io import BytesIO

def bbox_to_mask(bbox):
    '''
    bbox to rectangle pixelwise mask
    '''
    x1, y1, x2, y2 = bbox
    mask = np.zeros((constants.DETECTION_SCREEN_HEIGHT,
                     constants.DETECTION_SCREEN_WIDTH)).astype(int)
    mask[y1:y2, x1:x2] = 1
    return mask


def point_to_mask(point):
    '''
    single point to dense pixelwise mask
    '''
    x, y = point
    mask = np.zeros((constants.DETECTION_SCREEN_HEIGHT,
                     constants.DETECTION_SCREEN_WIDTH)).astype(int)
    mask[y, x] = 1
    return mask


def compress_mask(seg_mask):
    '''
    compress mask array
    '''
    run_len_compressed = []  # list of lists of run lengths for 1s, which are assumed to be less frequent.
    idx = 0
    curr_run = False
    run_len = 0
    for x_idx in range(len(seg_mask)):
        for y_idx in range(len(seg_mask[x_idx])):
            if seg_mask[x_idx][y_idx] == 1 and not curr_run:
                curr_run = True
                run_len_compressed.append([idx, None])
            if seg_mask[x_idx][y_idx] == 0 and curr_run:
                curr_run = False
                run_len_compressed[-1][1] = run_len
                run_len = 0
            if curr_run:
                run_len += 1
            idx += 1
    if curr_run:
        run_len_compressed[-1][1] = run_len
    return run_len_compressed


def compress_image(array):
    '''
    compress an numpy array with PNG
    '''
    assert array.dtype == 'int32'
    image = Image.fromarray(array)
    image_buffer = BytesIO()
    image.save(image_buffer, format='PNG')
    return image_buffer.getvalue()


def decompress_image(image_bytes):
    '''
    decompress a binary file compressed with PNG, returns a PIL image
    '''
    return Image.open(BytesIO(image_bytes))

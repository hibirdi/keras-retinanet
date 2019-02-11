import numpy as np


def compute_contains(detections, annotations):
    """
    Parameters
    ----------
    detections: list
        Contains detections in the form of a bounding box with its TL and BR coordinates
    annotations: list
        Contains annotations in the form of a bounding box with its TL and BR coordinates

    Returns
    -------
    overlaps: list
        all annotations that are covered by a detection by >70%
    """
    num_detections = detections.shape[0]
    num_annotations = annotations.shape[0]
    overlaps = np.zeros((num_detections, num_annotations), dtype=np.float64)

    for a in range(num_annotations):
        annotation_box = (annotations[a, 2] - annotations[a, 0] + 1) * (
            annotations[a, 3] - annotations[a, 1] + 1
        )
        for d in range(num_detections):

            width_diff = (
                min(detections[d, 2], annotations[a, 2])
                - max(detections[d, 0], annotations[a, 0])
                + 1
            )
            if width_diff > 0:
                height_diff = (
                    min(detections[d, 3], annotations[a, 3])
                    - max(detections[d, 1], annotations[a, 1])
                    + 1
                )
                if height_diff > 0:
                    intersection = width_diff * height_diff
                    if (intersection / annotation_box > 0.7):
                        overlaps[d, a] = intersection / annotation_box
    return overlaps

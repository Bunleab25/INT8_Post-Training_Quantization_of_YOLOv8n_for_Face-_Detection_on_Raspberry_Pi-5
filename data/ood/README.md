# Out-of-distribution (OOD) evaluation set

Required for Stage 1 Gate 4 and Stage 2 accuracy comparison.

Place labelled images here using either layout:

```
data/ood/images/*.jpg
data/ood/labels/*.txt    # YOLO format, class 0 = Face
```

or a flat directory of images with adjacent `.txt` labels.

Include crowds, profile views, occlusion, low light, small faces, and negatives
where licensing allows. Record licence notes next to the set.

Stage 2 never uses these images for INT8 calibration.

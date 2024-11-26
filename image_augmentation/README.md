# Image augmentation script

## How to use ?

Place the correct values in the `constants.yaml` file and run the script.

```bash
python image_augmentation.py
```

*Voilà*

## Credits

Largely inspired from [`muhammad-faizan-122 / yolo-data-augmentation`](https://github.com/muhammad-faizan-122/yolo-data-augmentation)

## Improvements

- [ ] `A.Normalize` outputs a black image when used, don't know why, but not using this option doesn't impact negatively the results.
- [ ] Can't install `pybboxes` to use `pybboxes.convert_bbox` to convert to bounding boxes from YOLO format to VOC format to print it for debug purposes. Dependencies issues.
# Sign language letters detection using YOLO

## Install

### Option 1 : Python Virtual Environment Install

Create a virtual environment :

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies :

```bash
pip install ultralytics tqdm opencv-python numpy albumentations pyside6
```

### Option 2 : Conda Install
```bash
conda install -c conda-forge ultralytics tqdm opencv numpy albumentations pyside6 openai
```

## Usage

To download and merge the datasets, see the `dataset_formatting/`.

To augment the images, see the `image_augmentation/` folder.

To train the model, see the `YOLO_train/` folder.
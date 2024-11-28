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
pip install ultralytics tqdm opencv-python numpy albumentations pyside6 openai
```

### Option 2 : Conda Install
```bash
conda install -c conda-forge ultralytics tqdm opencv numpy albumentations pyside6 openai
```

## Usage

To download and merge the datasets, see the `dataset_formatting/`.

To augment the images, see the `image_augmentation/` folder.

To train the model, see the `YOLO_train/` folder.

# Try the application  
To try the application with the already trained YOLO model, you can run `qt_app.py`. It opens a Qt HMI where you can see the detections, and concatenate the results to create words.

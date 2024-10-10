# GeoText-1652
An offical repo for ECCV 2024 Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching

- [x] Part I: Dataset
- [x] Part II: Annotation Pipeline
- [x] Part III: Data Feature
- [X] Part IV: Train and Test
- [ ] Part V: Visual Grounding


### Dataset
## Download Link
You could directly download the dataset from the google drive public link: https://drive.google.com/file/d/1vHjysm1VbJnmriKopIgnMxW-ZBR4mXQ1/view?usp=sharing. 

## Statistics of GeoText-1652

Training and test sets all include the image, global description, bbox-text pair and building numbers. We note that there is no overlap between the 33 universities of the training set and the 39 universities of the test sets. Three platforms are considered, i.e., drone, satellite, and ground cameras.

| Split                    | #Imgs  | #Global Descriptions | #Bbox-Texts | #Classes | #Universities |
|--------------------------|--------|----------------------|-------------|----------|---------------|
| **Training (Drone)**     | 37,854 | 113,562              | 113,367     | 701      | 33            |
| **Training (Satellite)** | 701    | 2,103                | 1,709       | 701      | 33            |
| **Training (Ground)**    | 11,663 | 34,989               | 14,761      | 701      | 33            |
| **Test (Drone)**         | 51,355 | 154,065              | 140,179     | 951      | 39            |
| **Test (Satellite)**     | 951    | 2,853                | 2,006       | 951      | 39            |
| **Test (Ground)**        | 2,921  | 8,763                | 4,023       | 793      | 39            |


The dataset 
## Dataset Structure

This repository includes training and testing data organized as follows:

### Directories

- **`train`**: Contains training images.
- **`test`**: Contains testing images.

### Files

- **`train.json`**
  - **Type**: JSON Source File
  - **Size**: 196,432 KB
  - Description: Contains the annotations and metadata for the training dataset.

- **`test_951_version.json`**
  - **Type**: JSON Source File
  - **Size**: 46,809 KB
  - Description: Contains the annotations and metadata for the test dataset.


## GeoText Dataset Official Structure

This dataset is designed to support the development and testing of models in geographical location recognition, providing images from multiple views at numerous unique locations.

### Directory Structure
```
GeoText_Dataset_Official/
├── test/
│ ├── gallery_no_train(250)/ // Contains images from 250 different locations, each with drone, street, and satellite views
│ │ ├── 0001/
│ │ │ ├── drone_view.jpg // Drone view image
│ │ │ ├── street_view.jpg // Street view image
│ │ │ ├── satellite_view.jpg // Satellite view image
│ │ ├── 0002/
│ │ ├── ... // More locations
│ │ ├── 0250/
│ ├── query(701)/ // Contains images from 701 different locations for query purposes, each with drone, street, and satellite views
│ │ ├── 0001/
│ │ │ ├── drone_view.jpg // Drone view image
│ │ │ ├── street_view.jpg // Street view image
│ │ │ ├── satellite_view.jpg // Satellite view image
│ │ ├── 0002/
│ │ ├── ... // More locations
│ │ ├── 0701/
├── train/ // Contains images from 701 different locations, each with drone, street, and satellite views for training
│ ├── 0001/
│ │ ├── drone_view.jpg // Drone view image
│ │ ├── street_view.jpg // Street view image
│ │ ├── satellite_view.jpg // Satellite view image
│ ├── 0002/
│ ├── ... // More locations
│ ├── 0701/
├── test_951_version.json // JSON file with annotations for the test dataset
├── train.json // JSON file with annotations for the training dataset
```
These files are critical for the machine learning models dealing with [specific task, e.g., image classification, object detection, etc.]. The JSON files include annotations necessary for training and testing the models.



### Example Entry in `train.json`

This entry provides a detailed description and annotations for a single image in the training dataset:

```json
{
  "image_id": "0839/image-43.jpeg",
  "image": "train/0839/image-43.jpeg",
  "caption": "In the center of the image is a large, modern office building with several floors. The building has a white facade with large windows that go all the way up to the top floor. There are several balconies on the upper floors, with white railings and green plants. The object in the center of the image is a large office building with several floors and a white facade. The building is surrounded by several other buildings, which are much smaller in size. On the right side of the building, there is a small parking lot with several cars parked in it. On the left side of the building, there is a street with cars driving on it. In the background, there are several trees and buildings that are further away.",
  "sentences": [
    "The object in the center of the image is a large office building with several floors and a white facade",
    "On the upper middle side of the building, there is a street with cars driving on it",
    "On the middle right side of the building, there is a small parking lot with several cars parked in it"
  ],
  "bboxes": [
    [0.408688485622406, 0.6883664131164551, 0.38859522342681885, 0.6234817504882812],
    [0.2420489490032196, 0.3855597972869873, 0.30488067865371704, 0.2891976535320282],
    [0.7388443350791931, 0.8320053219795227, 0.5213109254837036, 0.33447015285491943]
  ]
}
```
### Annotation Details

- **Caption**: Provides a global description for the entire image, framing the context for more detailed analyses.

- **Sentences**: Each sentence is aligned with a specific part of the image. These sentences are directly related to the bounding boxes, providing localized descriptions to enhance model training in tasks like image captioning and object detection.

- **Bounding Boxes (`bboxes`)**: Specified as arrays of coordinates in the format `[cx, cy, w, h]`, where:
  - `cx` (center x-coordinate) and `cy` (center y-coordinate) are the center coordinates of the bounding box relative to the image dimensions.
  - `w` (width) and `h` (height) represent the width and height of the bounding box, respectively. 

These annotations are crucial for tasks that require precise spatial localization, such as object detection and scene parsing, providing essential training data for machine learning models to accurately learn and predict.






# GeoText-1652 Setup and Usage Guide

This guide will walk you through the process of setting up the GeoText-1652 project environment.

## Prerequisites

- Git
- Git Large File Storage (LFS)
- Conda (we'll install Miniconda in the steps below)
- Internet connection for downloading necessary files

## Installation Steps

### 1. Clone the Repository

First, clone the GeoText-1652 repository:

```bash
git clone https://github.com/MultimodalGeo/GeoText-1652.git
```

### 2. Install Miniconda

Download and install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts to complete the Miniconda installation.

### 3. Create Conda Environment

Create a new conda environment for the project:

```bash
conda create -n gt python=3.8
```

Activate the environment:

```bash
conda activate gt
```

### 4. Install Requirements

Navigate to the project directory and install the required packages:

```bash
cd GeoText-1652
pip install -r requirements.txt
```

### 5. Install and Configure Git LFS

Install Git LFS:

```bash
apt install git-lfs
```

Set up Git LFS:

```bash
git lfs install
```

### 6. Download Dataset and Model

Clone the dataset and model repositories from Hugging Face:

```bash
git clone https://huggingface.co/datasets/truemanv5666/GeoText1652_Dataset
git clone https://huggingface.co/truemanv5666/GeoText1652_model
```

### 7. Extract Dataset Images

Navigate to the dataset images directory and extract the compressed files:

```bash
cd GeoText1652_Dataset/images
find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
```

This command will extract all `.tar.gz` files in the directory and remove the compressed files afterward.

### 8. Update Configuration File

You need to update the `reboot.yaml` configuration file with the correct paths. Open the file in a text editor and modify the following lines:

```yaml
train_file: ["/root/GeoText-1652/GeoText1652_Dataset/train.json"]
test_file: "/root/GeoText-1652/GeoText1652_Dataset/test_951_version.json"
image_root: '/root/GeoText-1652/GeoText1652_Dataset/images'
text_encoder: '/root/GeoText-1652/GeoText1652_model/bert'
```

Make sure these paths match your actual directory structure. If you cloned the repositories to a different location, adjust the paths accordingly.

### 9. Update SwinB Configuration

You also need to update the SwinB configuration file. Navigate to the `method/configs` directory and open the `config_swinB_384.json` file. Update the `ckpt` path as follows:

```json
"ckpt": "/root/GeoText-1652/GeoText1652_model/swin_base_patch4_window7_224_22k.pth"
```

Again, make sure this path matches your actual directory structure. If you cloned the model repository to a different location, adjust the path accordingly.

## Running the Model

After setting up your environment and configuring the files, you can run the model for evaluation or training. Navigate to the `method` directory to run these commands:

```bash
cd method
```

### Evaluation

To evaluate the model, use the following command:

```bash
python3 run.py --task "re_bbox" --dist "l4" --evaluate --output_dir "output/eva" --checkpoint "/root/GeoText-1652/GeoText1652_model/geotext_official_checkpoint.pth"
```

### Training

To train the model, use the following command:

```bash
nohup python3 run.py --task "re_bbox" --dist "l4" --output_dir "output/train" --checkpoint "/root/GeoText-1652/GeoText1652_model/geotext_official_checkpoint.pth" &
```

Note: The `nohup` command is used to run the training process in the background, allowing it to continue even if you close the terminal session.

### Adjusting GPU Settings

You can adjust the GPU settings in the `run.py` file if needed. Make sure to modify these settings according to your hardware capabilities and requirements.

## Next Steps

With these steps completed, you should be able to run both evaluation and training tasks for GeoText-1652. Remember to monitor your GPU usage and adjust settings as necessary for optimal performance.

## Troubleshooting

If you encounter any issues during setup or execution, please check the project's issue tracker on GitHub or reach out to the maintainers for support.

## Acknowledgements

We would like to express our gratitude to the creators of X-VLM (https://github.com/zengyan-97/X-VLM) for their excellent work, which has significantly contributed to this project.






















If you find GeoText-1652 useful for your work please cite:
```bib
@inproceedings{chu2024towards, 
      title={Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching}, 
      author={Chu, Meng and Zheng, Zhedong and Ji, Wei and Wang, Tingyu and Chua, Tat-Seng}, 
      booktitle={EECV}, 
      year={2024} 
      }
```

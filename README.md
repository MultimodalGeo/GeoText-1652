# 【ECCV2024】GeoText-1652 Benchmark

<div align="center">

**Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching**

Meng Chu¹, Zhedong Zheng²*, Wei Ji¹, Tingyu Wang³, Tat-Seng Chua¹

¹ School of Computing, National University of Singapore, Singapore  
² FST and ICI, University of Macau, China  
³ School of Communication Engineering, Hangzhou Dianzi University, China


[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2311.12751)
[![Poster](https://img.shields.io/badge/Poster-PDF-brightgreen)](https://drive.google.com/file/d/1QtLl3vtwUl-_rC_Ma48Gnaw-bHCMB7Qw/view?usp=share_link)
[![Project](https://img.shields.io/badge/Project-Website-blue)](https://multimodalgeo.github.io/GeoText/)
[![Dataset](https://img.shields.io/badge/Dataset-Download-yellow)](https://drive.google.com/file/d/1vHjysm1VbJnmriKopIgnMxW-ZBR4mXQ1/view?usp=sharing)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/truemanv5666/GeoText1652_Dataset)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blueviolet)](https://huggingface.co/truemanv5666/GeoText1652_model)




</div>

## 📰 Breaking News!!!

We have prepared 24G Test for CUDA OUT OF MEMORY users. You could find it in : https://huggingface.co/datasets/truemanv5666/GeoText1652_Dataset     and just use  test_24G_version.json

## 📚 About GeoText-1652

GeoText-1652 is a groundbreaking benchmark dataset for ECCV 2024, focusing on natural language-guided drone navigation with spatial relation matching. This dataset bridges the gap between natural language processing, computer vision, and robotics, paving the way for more intuitive and flexible drone control systems.

### 🌟 Key Features

- Multi-platform imagery: drone, satellite, and ground cameras
- Covers multiple universities with no overlap between train and test sets
- Rich annotations including global descriptions, bounding boxes, and spatial relations

## 📊 Dataset Statistics

Training and test sets all include the image, global description, bbox-text pair and building numbers. We note that there is no overlap between the 33 universities of the training set and the 39 universities of the test sets. Three platforms are considered, i.e., drone, satellite, and ground cameras.

| Split                    | #Imgs  | #Global Descriptions | #Bbox-Texts | #Classes | #Universities |
|--------------------------|--------|----------------------|-------------|----------|---------------|
| **Training (Drone)**     | 37,854 | 113,562              | 113,367     | 701      | 33            |
| **Training (Satellite)** | 701    | 2,103                | 1,709       | 701      | 33            |
| **Training (Ground)**    | 11,663 | 34,989               | 14,761      | 701      | 33            |
| **Test (Drone)**         | 51,355 | 154,065              | 140,179     | 951      | 39            |
| **Test (Satellite)**     | 951    | 2,853                | 2,006       | 951      | 39            |
| **Test (Ground)**        | 2,921  | 8,763                | 4,023       | 793      | 39            |

## 💾 Download Links

- Google Drive: [GeoText-1652 Dataset](https://drive.google.com/file/d/1vHjysm1VbJnmriKopIgnMxW-ZBR4mXQ1/view?usp=sharing)
- HuggingFace Hub:
  - Dataset: [truemanv5666/GeoText1652_Dataset](https://huggingface.co/datasets/truemanv5666/GeoText1652_Dataset)
  - Model: [truemanv5666/GeoText1652_model](https://huggingface.co/truemanv5666/GeoText1652_model)

## 📁 Dataset Structure

This dataset is designed to support the development and testing of models in geographical location recognition, providing images from multiple views at numerous unique locations.

### Directory Structure
```
GeoText_Dataset_Official/
├── test/
│ ├── gallery_no_train(250)/
│ │ ├── 0001/
│ │ │ ├── drone_view.jpg
│ │ │ ├── street_view.jpg
│ │ │ ├── satellite_view.jpg
│ │ ├── 0002/
│ │ ├── ... // More locations
│ │ ├── 0250/
│ ├── query(701)/
│ │ ├── 0001/
│ │ │ ├── drone_view.jpg
│ │ │ ├── street_view.jpg
│ │ │ ├── satellite_view.jpg
│ │ ├── 0002/
│ │ ├── ... // More locations
│ │ ├── 0701/
├── train/
│ ├── 0001/
│ │ ├── drone_view.jpg
│ │ ├── street_view.jpg
│ │ ├── satellite_view.jpg
│ ├── 0002/
│ ├── ... // More locations
│ ├── 0701/
├── test_951_version.json
├── train.json
```

### Annotation Details

Example entry in `train.json`:

```json
{
  "image_id": "0839/image-43.jpeg",
  "image": "train/0839/image-43.jpeg",
  "caption": "In the center of the image is a large, modern office building...",
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

- **Caption**: Provides a global description for the entire image.
- **Sentences**: Each sentence is aligned with a specific part of the image, related to the bounding boxes.
- **Bounding Boxes**: Specified as arrays of coordinates in the format `[cx, cy, w, h]`.

## 🛠️ Setup and Usage Guide

### Prerequisites

- Git
- Git Large File Storage (LFS)
- Conda

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/MultimodalGeo/GeoText-1652.git
   ```

2. Install Miniconda:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   sh Miniconda3-latest-Linux-x86_64.sh
   ```

3. Create and activate conda environment:
   ```bash
   conda create -n gt python=3.8
   conda activate gt
   ```

4. Install requirements:
   ```bash
   cd GeoText-1652
   pip install -r requirements.txt
   ```

5. Install and configure Git LFS:
   ```bash
   apt install git-lfs
   git lfs install
   ```

6. Download dataset and model:
   ```bash
   git clone https://huggingface.co/datasets/truemanv5666/GeoText1652_Dataset
   git clone https://huggingface.co/truemanv5666/GeoText1652_model
   ```

7. Extract dataset images:
   ```bash
   cd GeoText1652_Dataset/images
   find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
   ```

8. Update configuration files:
   - Update `re_bbox.yaml` with correct paths
   - Update `method/configs/config_swinB_384.json` with correct `ckpt` path



### Running the Model
From the `Method` directory:
```bash
cd Method
```

#### Evaluation
```bash
python3 run.py --task "re_bbox" --dist "l4" --evaluate --output_dir "output/eva" --checkpoint "/root/GeoText-1652/GeoText1652_model/geotext_official_checkpoint.pth"
```

Evaluation paths:
- Full test (951 cases): GeoText1652_Dataset/test_951_version.json
- 24GB GPU version (~190 cases): GeoText1652_Dataset/test_24G_version.json

24GB Version Results on Two 3090Ti:
```
| Text Query | Image Query |
|R@1  R@5  R@10|R@1  R@5  R@10|
|----|----|----|----|----|----| 
|29.9|46.3|54.1|50.1|81.2|90.3|
```
Full evaluation results are in the paper.

#### Training
```bash
nohup python3 run.py --task "re_bbox" --dist "l4" --output_dir "output/train" --checkpoint "/root/GeoText-1652/GeoText1652_model/geotext_official_checkpoint.pth" &
```





## 📄 Citation

If you find GeoText-1652 useful for your work, please cite:

```bibtex
@inproceedings{chu2024towards, 
  title={Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching}, 
  author={Chu, Meng and Zheng, Zhedong and Ji, Wei and Wang, Tingyu and Chua, Tat-Seng}, 
  booktitle={ECCV}, 
  year={2024} 
}
```

## 🙏 Acknowledgements

We would like to express our gratitude to the creators of [X-VLM](https://github.com/zengyan-97/X-VLM) for their excellent work, which has significantly contributed to this project.


<div align="center">
  Made with ❤️ by the GeoText-1652 Team
</div>

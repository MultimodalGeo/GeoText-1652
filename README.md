# GeoText-1652
An offical repo for ECCV 2024 Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching

- [x] Part I: Dataset
- [ ] Part II: Annotation Pipeline
- [ ] Part III: Data Feature
- [ ] Part IV: Train and Test
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
If you find GeoText-1652 useful for your work please cite:
```
@inproceedings{chu2024towards, 
      title={Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching}, 
      author={Chu, Meng and Zheng, Zhedong and Ji, Wei and Wang, Tingyu and Chua, Tat-Seng}, 
      booktitle={EECV}, 
      year={2024} 
      }
```
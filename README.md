# GeoText-1652
An offical repo for ECCV 2024 Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching

- [x] Part I: Dataset
- [ ] Part II: Annotation Pipeline
- [ ] Part III: Data Feature
- [ ] Part IV: Train and Test
- [ ] Part V: Visual Grounding

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


If you find GeoText-1652 useful for your work please cite:
```
@inproceedings{chu2024towards, 
      title={Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching}, 
      author={Chu, Meng and Zheng, Zhedong and Ji, Wei and Wang, Tingyu and Chua, Tat-Seng}, 
      booktitle={EECV}, 
      year={2024} 
      }
```
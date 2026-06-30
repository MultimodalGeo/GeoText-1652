# GeoText-1652 Result Candidates

This file tracks papers found through citation search that may contain GeoText-1652 results.
Only papers with public, sourceable six-metric results should be added to `leaderboard/data.json`.

Last checked: 2026-06-30

## Added to Leaderboard

| Paper | Venue / Year | Source | Status |
| --- | --- | --- | --- |
| Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching | ECCV 2024 | https://arxiv.org/abs/2311.12751 | Added official full-test and 24GB baselines |
| HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones | arXiv 2025 | https://arxiv.org/abs/2508.21539 | Added full-test result |
| Turning Generators into Retrievers: Unlocking MLLMs for Natural Language-Guided Geo-Localization | arXiv 2026 | https://arxiv.org/abs/2604.10721 | Added full-test and 24GB results |
| Seeing With Words: Interpretable Language-Guided Drone Geo-Localization via LLM-Enriched Semantic Attribute Alignment | IEEE TMM 2026 | https://doi.org/10.1109/TMM.2025.3642913 | Added SAA-DGL X-VLM and X2-VLM full-test and 24GB results from Table I of the local PDF |

## Needs Score Verification

These citing papers appear relevant, but the six GeoText-1652 leaderboard metrics were not publicly available from the accessible metadata/pages at the time of checking.

| Paper | Year | Link | Notes |
| --- | --- | --- | --- |
| Dual Set Contrastive Learning for Natural Language-Guided Drone Geo-Localization | 2026 | https://doi.org/10.1109/MAI69289.2026.11543778 | OpenAlex abstract says experiments on GeoText-1652; accessible IEEE page did not expose result tables. |
| Region-Level Cross-Modal Matching Framework for Text-Based Geo-Localization | 2026 | https://doi.org/10.1007/978-981-95-6123-0_53 | Springer abstract says SOTA on GeoText-1652; full results are behind access controls. Google Scholar snippet exposes partial main-test metrics: Image-to-Text R@1/R@5/R@10 = 27.6/54.7/67.4 and Text-to-Image R@1 = 13.7, but Text R@5/R@10 remain hidden. Do not add until all six values are confirmed. |
| Team Xiaomi EV-AD VLA: Caption-Guided Retrieval System for Cross-Modal Drone Navigation -- Technical Report for IROS 2025 RoboSense Challenge Track 4 | 2025 | https://arxiv.org/abs/2510.02728 | Reports RoboSense Challenge Track 4 single-direction Text-to-Image results (R@1/R@5/R@10 = 31.33/49.09/57.15), not the full six-metric GeoText leaderboard protocol. |
| A Parameter-Efficient MoE Framework for Cross-Modal Drone Navigation | 2025 | https://github.com/robosense2025/toolkit/blob/main/technical_reports/RoboSense2025_Track4_lineng.pdf | Found via Google Scholar/Robosense report. It is the winning RoboSense Track 4 solution, but the challenge uses GeoText-190 derived from GeoText-1652 and reports only Text-to-Image R@1/R@10. |
| HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Cross-Modal Drone Navigation | 2025 | https://github.com/robosense2025/toolkit/blob/main/technical_reports/RoboSense2025_Track4_rhao_hur.pdf | RoboSense Track 4 technical report variant of HCCM; not the same six-metric GeoText-1652 protocol. |

## Likely Not Leaderboard Entries

| Paper | Year | Link | Reason |
| --- | --- | --- | --- |
| VLGeo: Bridging Viewpoints in UAV-View Geo-Localization Using a Large Vision-Language Model | 2026 | https://doi.org/10.1109/TGRS.2026.3680280 | Introduces/evaluates UAV-VLL; OpenAlex abstract does not indicate GeoText-1652 metrics. |
| Learning Better UAV-Based Cross-View Object Geo-Localization from Multi-Modal Prompts: MoP-UAV Benchmark and MoPT Framework | AAAI 2026 | https://ojs.aaai.org/index.php/AAAI/article/view/38282 | Found by Scholar keyword search. Appears to introduce MoP-UAV rather than report GeoText-1652 retrieval leaderboard scores. |
| MMGeo: Multimodal Compositional Geo-Localization for UAVs | ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Ji_MMGeo_Multimodal_Compositional_Geo-Localization_for_UAVs_ICCV_2025_paper.html | Openaccess PDF checked. It cites GeoText-1652 in related work/dataset comparison but reports GTA-UAV-MM and UAV-VisLoc-MM results, not GeoText-1652 leaderboard metrics. |
| GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization | CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/html/Song_GeoBridge_A_Semantic-Anchored_Multi-View_Foundation_Model_Bridging_Images_and_Text_CVPR_2026_paper.html | Found by Scholar keyword search. Appears to be broader geo-localization foundation model work rather than GeoText-1652 leaderboard reporting. |
| CDM-Net: A Framework for Cross-View Geo-Localization With Multimodal Data | 2025 | https://doi.org/10.1109/TGRS.2025.3594544 | Found by Scholar keyword search. No accessible evidence of GeoText-1652 six-metric leaderboard results. |
| Road Maps as Free Geometric Priors: Weather-Invariant Drone Geo-Localization with GeoFuse | 2026 | https://arxiv.org/abs/2605.14925 | Found by Scholar keyword search. Drone geo-localization paper, but not a GeoText-1652 text-image leaderboard result. |
| Where am I? Cross-View Geo-localization with Natural Language Descriptions | ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Ye_Where_am_I_Cross-View_Geo-localization_with_Natural_Language_Descriptions_ICCV_2025_paper.html | CVG-Text paper; relevant field but different benchmark. |
| CityCube: Benchmarking Cross-view Spatial Reasoning on Vision-Language Models in Urban Environments | 2026 | https://arxiv.org/abs/2601.14339 | Uses GeoText-1652 imagery as a data source for a new benchmark; does not report GeoText-1652 retrieval leaderboard scores. |
| Global Cross-Modal Geo-Localization: A Million-Scale Dataset and a Physical Consistency Learning Framework | 2026 | https://arxiv.org/abs/2603.08491 | Discusses GeoText-1652 as a related dataset, but reports results on CORE/CVG-Text rather than GeoText-1652. |
| Vision-Language Modeling Meets Remote Sensing: Models, datasets, and perspectives | 2025 | https://arxiv.org/abs/2505.14361 | Survey paper, not a new GeoText-1652 result. |
| The 2nd/3rd Workshop on UAVs in Multimedia | 2024/2025 | https://doi.org/10.1145/3689095.3695932 | Workshop overview, not a model result. |

## Related Challenge Results

These are useful for the project page, but should not be mixed into the main GeoText-1652 full-test/24GB leaderboards because the split and metrics differ.

| Benchmark | Source | Result |
| --- | --- | --- |
| RoboSense 2025 Track 4 / roboText-190 Phase 2 | https://arxiv.org/abs/2601.05014 | lineng: R@1 38.31, R@10 61.32, score 49.82; rhao_hur: R@1 28.34, R@10 66.11, score 47.23; Xiaomi EV-AD VLA: R@1 31.33, R@10 57.15, score 44.24; X-VLM baseline: R@1 25.44, R@10 49.10, score 37.27. |

## Search Sources

- OpenAlex citation list for the ECCV 2024 GeoText-1652 paper: https://openalex.org/W4403947192
- OpenAlex arXiv record: https://openalex.org/W4388928006
- Google Scholar keyword search for `"GeoText-1652"` through the first 100 visible results and the citation cluster shown on the author profile (`cites=11632066698005565332`). Direct Google Scholar citation-list scraping was blocked by captcha, but keyword-search pages exposed additional candidate titles and snippets.
- Crossref DOI metadata for relevant citing papers.

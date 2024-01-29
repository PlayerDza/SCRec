# SCRec (Comprehend Then Predict: Prompting Large Language Models for Recommendation with Semantic and Collaborative Data)

## Paper
The code of "Comprehend Then Predict: Prompting Large Language Models for Recommendation with Semantic and Collaborative Data", submitted to SIGIR24

## Datasets to [download]([https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/Eln600lqZdVBslRwNcAJL5cBarq6Mt8WzDKpkq1YCqQjfQ?e=cISb1C](https://userscloud.com/gng54qj3lnt7))
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019



## Usage
You can use `run.sh` to activate the training and testing progress.

Notice that:

* The downloaded dataset should be placed in `./data`
* Folders `checkpoints` and `logs` are needed for storing the model and record files.
* Training and testing process may cost a lot of time and memory (about 45-50G following the default settings on an Nvidia A800 GPU)



## Code dependencies
- Python 3.8
- PyTorch 1.13.0
- transformers 4.29.2

## Code reference
- [PEPLER (PErsonalized Prompt Learning for Explainable Recommendation)]


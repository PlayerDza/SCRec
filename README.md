# SCRec (Comprehend Then Predict: Prompting Large Language Models for Recommendation with Semantic and Collaborative Data)

## Paper
The code of "Comprehend Then Predict: Prompting Large Language Models for Recommendation with Semantic and Collaborative Data", submitted to SIGIR24

## Datasets to [download](https://drive.google.com/drive/folders/1vZu2NdmaAYlxDk0ZfBqrNoyWQwqyoc45?usp=drive_link)
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
- [PEPLER (PErsonalized Prompt Learning for Explainable Recommendation)](https://github.com/lileipisces/PEPLER)


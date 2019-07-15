# Audioset_multi_label_classification 

## Prerequisites

1. Python packages:
	- Python 3.5 or higher
	- Pytorch 1.0 or higher
	- Numpy
  - json
  - yaml
  
## Usage

1. Download data:
	
	-Audioset : https://research.google.com/audioset/download.html

2. Preprocess data:

	```
	python3 data/preprocess.py $data_path
	```

3. Train model:

	```
	python3 train_rnn_decoder.py -gpus 0 -config config/config_ocd.yaml
	```

	- Hyperparameters can be modified in config/config_ocd.yaml
	
	- Log can be found in the log directory.
	
	```
	python3 train_baseline.py -gpus 0 -config config/config_ocd.yaml
	```

	- Codes for training binary relevance model.

4. test model:

	```
	python3 train_rnn_decoder.py -gpus 0 -config config/config_ocd.yaml-restore $expdir/best_in_train_micro_f1_checkpoint.pt -notrain
	```

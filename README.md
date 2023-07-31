# THUNDER

## Requirements

The requirements are listed in the [`requirements.txt`](requirements.txt) file.

```
$ conda create -n thunder python=3.8 anaconda
$ conda activate thunder
$ pip install -r requirements.txt
```

Download the pretrained RoBERTa-base using the script.

`./clone.sh`

### Tested environment

* Anaconda Python 3.8.5
* Pytorch 2.0.1
* NVIDIA RTX 3090 GPU

## Datasets

The datasets are in the [`data`](data) directory.

* [`data/conll`](data/conll)
* [`data/onto`](data/onto)
* [`data/wikigold`](data/wikigold)

## Scripts

The scripts are in the [`scripts`](scripts) directory.

* [`scripts/conll.sh`](scripts/conll.sh)
* [`scripts/onto.sh`](scripts/onto.sh)
* [`scripts/wikigold.sh`](scripts/wikigold.sh)

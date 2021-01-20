# Spoken Language Classification


This repository contains some models pretrained on the [http://bark.phon.ioc.ee/voxlingua107/](Voxlingua107) dataset to be used for spoken (audio based) language classification.
Three models are provided.


## Usage

```bash
git clone https://github.com/RicherMans/SpokenLanguageClassifiers
pip install -r requirements.txt
python3 predict.py AUDIOFILE
```

The models (see below) can be also modified. Currently four models have been pretrained. All of which are accessed with the ``--model MODELNAME`` parameter.

By default the models just print the top ``N`` results (N=5 and can be changed with `--N NUMBER`).

## Models

Four models were pretrained and can be chosen as the back-end:

1. CNN6 (default) : A six layer CNN model, using attention as temporal aggregation.
2. CNN10: A ten layer CNN model, using mean and max pooling as temporal aggregation.
3. MobilenetV2: A mobilenet implementation for audio classification.
4. CNNVAD: A model that simultaneously does VAD and classification. The VAD model is taken from [https://github.com/RicherMans/Datadriven-GPVAD](GPV). Model training has been done by fine-tuning both VAD and Language classification models. The back-end model here is the default CNN6.


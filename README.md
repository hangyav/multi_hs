# Abusive language detection for low-resource settings leveraging external data sources

![labels_overlap_example](https://github.com/hangyav/multi_hs/assets/414596/7233c740-ad81-4e6b-bb60-54ea2cb97bc5)

Although, already a large set of annotated corpora with different properties and label sets were created for abusive language detection, due to the broad range of social media platforms and their user groups, not all use cases and communities are supported by such datasets. Since, the annotation of new corpora is expensive, this tool leverages datasets we already have, covering a wide range of tasks related to abusive language detection. It allows building models cheaply for a new target label set and/or language, using only a few training examples of the target task. For further details, please see the related [papers](#Papers).

## Installing

The project was tested with python version 3.9.12. To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

Optionally, to test the environment, you can run the following command:
```bash
./run_tests.sh
```

## Data

The following datasets are supported:
- `ami18`: [web](https://amievalita2018.wordpress.com/data), [config](src/data/ami18/ami18.py)
- `bajer`: [web](https://github.com/phze22/Online-Misogyny-in-Danish-Bajer), [config](src/data/bajer/bajer.py)
- `germeval18`: [web](https://github.com/uds-lsv/GermEval-2018-Data), [config](src/data/germeval18/germeval18.py)
- `hasoc19`: [web](https://hasocfire.github.io/hasoc/2019/call_for_participation.html), [config](src/data/hasoc19/hasoc19.py)
- `haspeede1`: [web](https://github.com/msang/haspeede/tree/master/2018), [config](src/data/haspeede1/haspeede1.py)
- `haspeede2`: [web](https://github.com/msang/haspeede/tree/master/2020), [config](src/data/haspeede2/haspeede2.py)
- `haspeede3`: [web](https://github.com/mirkolai/EVALITA2023-HaSpeeDe3), [config](src/data/haspeede3/haspeede3.py)
- `hate_speech18`: [web](https://github.com/Vicomtech/hate-speech-dataset), [config](src/data/hate_speech18/hate_speech18.py)
- `hateval19`: [web](https://github.com/cicl2018/HateEvalTeam), [config](src/data/hateval19/hateval19.py)
- `ihsc`: [web](https://github.com/msang/hate-speech-corpus), [config](src/data/ihsc/ihsc.py)
- `large_scale_xdomain`: [web](https://github.com/avaapm/hatespeech), [config](src/data/large_scale_xdomain/large_scale_xdomain.py)
- `measureing_hate`: [web](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech), [config](src/data/measuring_hate/measuring_hate.py)
- `mlma`: [web](https://github.com/HKUST-KnowComp/MLMA_hate_speech), [config](src/data/mlma/mlma.py)
- `olid`: [web](https://github.com/idontflow/OLID), [config](src/data/olid/olid.py)
- `religious_hate`: [web](https://github.com/dhfbk/religious-hate-speech), [config](src/data/religious_hate/religious_hate.py)
- `rp21`: [web](https://zenodo.org/records/5291339#.Yo3uPBxByV4), [config](src/data/rp21/rp21.py)
- `srw16`: [web](https://github.com/zeeraktalat/hatespeech), [config](src/data/srw16/srw16.py)
- `told_br`: [web](https://github.com/JAugusto97/ToLD-Br), [config](src/data/told_br/told_br.py)
- `us_elect20`: [web](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/stance-hof), [config](src/data/us_elect20/us_elect20.py)

Each dataset has multiple label configurations. For details see under the `config` link.

The project uses the 🤗 Datasets framework to download train and evaluation data from the Hub. However, in some cases the datasets have to be downloaded manually and the below environmental variable be set:
- `haspeede3`: `HASPEEDE3_URL` path pointing to a directory containing the extracted files in the same structure as the [github](https://github.com/mirkolai/EVALITA2023-HaSpeeDe3) repository.
- `ihsc`: `IHSC_TWEETS` pointing to a csv file containing tweet ids and texts.
- `large_scale_xdomain`: `LARGE_SCALE_XDOMAIN_TWEETS` pointing to a csv file containing tweet ids and texts.
- `religious_hate`: `RELIGIOUS_HATE_URL` path pointing to a directory containing the `dataset_en-portion_tweets.csv` and `dataset_it-portion_tweets.csv` files. Both should contain tweet ids and texts.
- `srw16`: `SRW16_TWEETS` pointing to a csv file containing tweet ids and texts.
## Running experiments

See and run:
```bash
./run_multi_example.sh
```


## Papers
```bibtex
@inproceedings{hangya-fraser-2024-solve-shot,
    title = {{How to Solve Few-Shot Abusive Content Detection Using the Data We Actually Have}},
    author = {Hangya, Viktor  and Fraser, Alexander},
    booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
    year = {2024},
    publisher = {ELRA and ICCL},
    url = {https://aclanthology.org/2024.lrec-main.729},
    pages = {8307--8322},
}

@inproceedings{LmuAtHaspeedeHangya2023,
    author = {Hangya, Viktor and Fraser, Alexander},
    title = {{LMU at HaSpeeDe3: Multi-Dataset Training for Cross-Domain Hate Speech Detection}},
    booktitle = {The Eighth Evaluation Campaign of Natural Language Processing and Speech Tools for Italian. Final Workshop (EVALITA 2023)},
    publisher = {EVALITA},
    url = {https://ceur-ws.org/Vol-3473/paper24.pdf},
    year = {2023},
}
````

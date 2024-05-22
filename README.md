# Abusive language detection for low-resource settings leveraging external data sources
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

## Running experiments


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

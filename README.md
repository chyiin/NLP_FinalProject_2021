# FinalProject_NLP_2021

This repo contains the code for NLP 2021 spring final project.

#### Team member:
* R09946001 陳知遙
* R09946006 何青儒
* R09946021 黃瀞瑩

## Folder Sturture
```
Final_NLP/
├── RISK/
    ├── RISK_preprocess.py
    ├── risk_model.pt
    ├── main.py
    ├── decision.csv
    └── config.py
└── QA/
    ├── QA_preprocess.py
    ├── qa_model.pt
    ├── main.py
    ├── qa.csv
    └── config.py
└── data/
    └── all_necessary_data...
```

## Requirements
Python >= 3.6, all necessary packages can be installed by:
```
pip install -r requirements.txt
```
## RISK Task

Data preprocess:

```
python3 RISK/RISK_preprocess.py --data_dir ../data/Train_risk_classification_ans.csv
```

Training mode:

```
python3 RISK/main.py --mode train --data_dir ../data/Train_risk_preprocess.csv
```

Testing mode:

```
python3 RISK/main.py --mode predict --data_dir ../data/Test_risk_preprocess.csv --model_file risk_model.pt
```

## QA Task

Data preprocess:

```
python3 QA/QA_preprocess.py --data_dir ../data/Train_qa_ans.json
```

Training mode:

```
python3 QA/main.py --mode train --data_dir ../data/Train_qa_preprocess.csv
```

Testing mode:

```
python3 QA/main.py --mode predict --data_dir ../data/Test_qa_preprocess.csv --model_file qa_model.pt
```

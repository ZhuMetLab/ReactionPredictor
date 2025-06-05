# 🧪 Reaction Prediction Model

This project provides a reaction prediction model based on a deep learning framework using graph neural networks (GNN). It predicts whether a given pair of reactants is likely to undergo a reaction. The model accepts input as pairs of SMILES strings and outputs the predicted reaction probability.

---

## 📂 Project Structure

```

reaction-predictor/
├── predict.py                # Command-line interface for prediction
├── train.py                  # Training script (optional, not required for inference)
├── RxnPred/                  # Core model and data utilities
│   ├── model.py
│   ├── getTfRecord.py
│   └── ...
├── Model/
│   └── graph\_structure\_reaction/
│       ├── *.json            # Model config files
│       ├── model\_*.ckpt      # Trained model weights
│       └── isotonic\_model.joblib  # Isotonic regression calibration model
├── demo.csv                  # Example input file
├── demo\_out.csv              # Example output file
└── README.md                 # Documentation

````

---

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/ZhuMetLab/ReactionPredictor.git
cd ReactionPredictor
````

2. **Install dependencies**

We recommend using a conda environment:

```bash
conda create -n RxnPred python=3.10
conda activate RxnPred
```

## Requirements

- keras >= 2.8.0
- mordred 1.2.0
- networkx 2.8.8
- numpy >= 1.22.2
- pandas >= 1.4.1
- python >= 3.9.7
- rdkit >= 2022.9.1
- tensorflow >= 2.8.0
- tqdm >= 4.64.1

---

## 📥 Input Format

Input should be a CSV file with two columns:

* `SMILES1`: SMILES string of the first metabolite
* `SMILES2`: SMILES string of the second metabolite

Example (`demo.csv`):

```csv
SMILES1,SMILES2
C(CC(=O)O)[C@@H](C(=O)O)N,CC(=O)N[C@@H](CCC(=O)O)C(=O)O
CCCCC,C(C(=O)O)N
CC(=O)SCCNC(=O)CCNC(=O)[C@@H](C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)O,CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O
ABCDE,C(C(=O)O)N
C(CS(=O)(=O)O)N,C(CS(=O)(=O)O)NC(=O)CO
C(CCNCCCN)CN,C(CCNCCCN)CNCCCN
C(CCNCCCN)CNCCCN,CCCCCC
C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O,C([C@H]([C@H]([C@@H]([C@H](C=O)O)O)O)O)OP(=O)(O)O
COCOCOCOCOCOCOCOCOCOCOC,C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C
C(C(=O)O)N,TiO2
```

> **Note**: Invalid or unparseable SMILES will be flagged as `ERROR` in the output.

---

## 📤 Output Format

The output CSV will contain the same SMILES pairs with an additional `score` column indicating predicted reaction probability (0–1), or `"ERROR"` for invalid inputs.

---

## 🧠 Running Prediction

You can run prediction from the command line:

```bash
python predict.py --input demo.csv --output demo_out.csv
```

Optional arguments:

* `--batch_size`: Batch size for prediction (default: `64`)
* `--keep_tfrecord`: If specified, temporary TFRecord files will be preserved

Example with options:

```bash
python predict.py --input demo.csv --output demo_out.csv --batch_size 128 --keep_tfrecord
```

---

## 🧪 Model Info

* **Architecture**: GNN-based model using graph convolutions and dense layers
* **Input**: Metabolite SMILES pairs
* **Output**: Reaction probability

---

## ⚠️ Notes

* Input SMILES must be valid; otherwise the line will be marked as `"ERROR"`.
* Temporary files (`.tfrecord`) are deleted by default unless `--keep_tfrecord` is specified.
* This repo currently supports **prediction only**. For training or model fine-tuning, refer to `train.py`.

---

## 📬 Contact

For questions, feel free to open an issue or contact the authors at \[[zhanghs@sioc.ac.cn](mailto:zhanghs@sioc.ac.cn)].

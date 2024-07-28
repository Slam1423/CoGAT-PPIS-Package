# CoGAT-PPIS-Package
## Overview
This is the Python package of CoGAT-PPIS, a sequence-based model for PPI site prediction. It contains all the necessary components from feature generation to final prediction so that you only need to input raw protein sequences to obtain the prediction results.

## Installation
```bash
git lfs clone git@github.com:Slam1423/CoGAT-PPIS-Package.git
```

## Requirements:
- Linux/MacOS
- python 3.9.19
- biopython 1.76
- pybiolib 1.1.988
- pytorch 2.2.2
- dgl 2.2.0
- scikit-learn 1.4.2
- transformers 4.37.2
- numpy 1.19.5
- pandas 1.1.5
- scipy 1.5.4

## Hyperparameters:
`-f`: The dataset name.
usage example: `-f example`

`-d`: Protein database for Blast.
usage example: `-d nr`

`-n`: Number of sequences in multiple sequence alignments. (preferably >= 1000)
usage example: `-n 1000`

## How to run
First, you have to copy your dataset into `CoGAT-PPIS-Package/raw_input_sequences/` with the name of `datasetname + _seq.txt` in the form of fasta. For example, your dataset name is `example`, then you should save the sequences into `example_seq.txt` in the form of fasta as follows:

```bash
>1acb_I
KSFPEVVGKTVDQAREYFTLHYPQYDVYFLPEGSPVTLDLRYNRVRVFYNPGTNVVNHVPHVG
>1ay7_A
DVSGTVCLSALPPEATDTLNLIASDGPFPYSQDGVVFQNRESVLPTQSYGYYHEYTVITPGARTRGTRRIITGEATQEDYYTGDHYATFSLIDQTC
```

Now you can run the package with the following codes:

```bash
cd CoGAT-PPIS-Package/
python main.py -f example -d nr -n 1000
```

The predicted labels will be saved to `predict_result_dir/example_predict_result.pkl` in the form of list in python.


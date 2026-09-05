# GRU baseline

The exact PubChem training corpus and fixed generated sample are public.
First restore the lossless corpus from the repository root:

    python scripts/revision/restore_pubchem_corpus.py

The training program accepts explicit input and output paths and no longer
changes directory to a developer's home folder:

    python generative_baseline/train_model.py --help
    python generative_baseline/train_model.py --smiles_file "data/Pubchem dataset.txt" --output_dir output/gru_new_run --rnn_type GRU --embedding_size 128 --hidden_size 512 --n_layers 3 --learning_rate 0.001 --batch_size 128 --patience 100

Training is resource-intensive; this command is a fresh run, not a guarantee
of recovering the identical historical sample. The published
data/gru/generate_result.csv (95,285 pairs) is the authoritative fixed input to
scripts/revision/run_gru_ood_similarity.py, which samples 20,000 pairs with
random_state=42. The historical trained GRU checkpoint and complete training
log have not been identified among the deposited assets.

sample-molecules.py samples training subsets from a corpus; it is not a
standalone neural generation checkpoint loader. The training program contains
its own sampling options. See --help for those options.

The default SMILES-mode CLI is checked with the repository environment.
Optional SELFIES/DeepSMILES modes load their separate packages only when used;
those modes and full stochastic retraining were not exercised in this release.
PubChem attribution and provider terms apply to the corpus.

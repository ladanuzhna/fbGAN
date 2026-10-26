# fbGAN

A feedback-guided generative adversarial network for DNA sequences, built as a CS496 course project. A Wasserstein GAN generates DNA, and a recurrent feedback network scores the translated proteins for the presence of secondary-structure classes. Generated samples that pass the selected score thresholds join the GAN training set.

The feedback model predicts which structure classes occur anywhere in a protein. Its eight scores are sequence-level labels; they do not assign a structure to each residue.

## How it works

1. Read protein sequences and secondary-structure labels from a CSV.
2. Encode each protein as DNA with a start codon and a stop codon. One-hot encode DNA using five channels: padding, A, T, C, and G.
3. Train a GAN with convolutional residual blocks and a Wasserstein loss with gradient penalty.
4. Translate generated DNA and score the proteins with a bidirectional LSTM feedback model.
5. Add a generated sample when every requested feature scores above `0.75`.

The default DNA length is 243 bases. Data preparation selects proteins with 79 residues, leaving six bases for the start and stop codons. The default desired features are `C`, `H`, and `E`.

## Setup

Use Python 3.9–3.11 and the Keras 2 dependencies in `requirements.txt`. The saved feedback models use the legacy TensorFlow SavedModel format. On Apple Silicon, use a native ARM64 Python installation.

```bash
git clone https://github.com/ladanuzhna/fbGAN.git
cd fbGAN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The discriminator has a large dense layer and can require several gigabytes of memory during training.

## Training data

The training CSV is not included. Place it at `data/2018-06-06-ss.cleaned.csv`, pass its location with `--data`, or set `FBGAN_DATA_PATH` before importing the modules.

Required columns:

| Column | Content |
| --- | --- |
| `seq` | Protein sequence using the 20 standard amino-acid letters |
| `sst8` | Secondary-structure labels for the sequence |
| `len` | Protein length in residues |
| `has_nonstd_aa` | Boolean flag for nonstandard amino acids |

Feedback scores use the fixed output order `B, C, E, G, H, I, S, T`: bridge, coil, strand, 3₁₀ helix, α-helix, π-helix, bend, and turn.

## Run the GAN

Prepare the data without training:

```bash
python __main__.py --data /path/to/protein_structure.csv
```

Train a new GAN:

```bash
python __main__.py --data /path/to/protein_structure.csv --epochs 10
```

Training uses batches of 128 and drops incomplete batches. At least 128 prepared sequences are required by the command-line launcher. To control training and save weights from Python:

```python
from gan import GAN
from utils.data_utilities import prepare_dataset

inputs = prepare_dataset("/path/to/protein_structure.csv")
gan = GAN()
gan.train(inputs, epochs=10)
print(gan.generate_samples(number=3, decoded=True))
gan.G.save_weights("weights/gan/new_Generator")
gan.D.save_weights("weights/gan/new_Discriminator")
```

## Train and use feedback

Train a feedback model and keep its fitted tokenizer with the model. Token indices must match the vocabulary used during training.

```python
from models import Feedback
from utils.data_utilities import get_data_for_feedback

X_train, X_test, y_train, y_test, tokenizer = get_data_for_feedback(
    "/path/to/protein_structure.csv"
)
feedback = Feedback(n_words=len(tokenizer.word_index) + 1)
feedback.train(
    X_train, y_train, X_test, y_test,
    save_feedback="weights/my_feedback", EPOCHS=5,
)
```

In the same Python session, use the trained model and tokenizer:

```python
from gan_fbnet import GAN_FBNet

fbgan = GAN_FBNet(
    fbnet_path="weights/my_feedback",
    tokenizer=tokenizer,
    features=["C", "H", "E"],
)
fbgan.train(inputs, epochs=10, steps_per_epoch=100)
```

This example starts a new GAN. To continue from your own complete checkpoints, pass `generator_path` and `discriminator_path` to `GAN_FBNet`. Each epoch processes at most `steps_per_epoch` batches; accepted samples become available in the next epoch's dataset.

## Repository contents

| Path | Purpose |
| --- | --- |
| `models.py` | Generator, discriminator, residual block, and feedback network |
| `gan.py` | GAN training, gradient penalty, sampling, and loss history |
| `gan_fbnet.py` | Feedback scoring and sample selection |
| `gan-fbnet.py` | Compatibility import for the original filename |
| `globals.py` | Model dimensions, feature defaults, and file paths |
| `utils/` | Data preparation, DNA translation, and plotting |
| `Notebooks/` | Original Colab experiments and secondary-structure analysis |
| `Plots/` | Saved experiment figures |
| `weights/` | Historical model files |

## Saved models and limitations

The GAN files in `weights/gan/` contain checkpoint indexes without the corresponding `.data-*` shards. They cannot restore the GAN on their own. Train new weights or supply complete checkpoints.

The repository includes a feedback SavedModel, but no fitted tokenizer. Rebuilding a tokenizer requires the original training data and preprocessing order; a newly fitted vocabulary may assign different indices. The example above trains a model and tokenizer together.

The notebooks retain their original Colab paths and embedded experimental code. The Python modules contain the corrections described by the current usage examples. Corrections to the residual blocks and protein encoding change training behavior, so historical weights and figures do not validate the corrected implementation.

Generated sequences and feedback scores have not been experimentally validated. The feedback model scores structure-class presence; it does not establish protein function or folding.

## Checks

```bash
python -m unittest discover -s tests -v
```

The regression tests use synthetic data and small model dimensions to check encoding, data preparation, model shapes, gradient updates, and feedback sample selection. They do not measure trained model quality.

## Credits and license

The DNA conversion utilities credit [niranjangavade98/DNA-to-Protein-sequence](https://github.com/niranjangavade98/DNA-to-Protein-sequence) and [av1659/fbgan](https://github.com/av1659/fbgan/blob/master/utils/bio_utils.py).

Released under the [MIT License](LICENSE).

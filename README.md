# Understanding Neural Network Optimization — code

Companion code for the book **Understanding Neural Network Optimization:
How Weights Learn, One Step at a Time** by Mohammad Jamhuri (2026).

Every number printed in the book comes from a script in `code/`.

| path | content |
|---|---|
| `code/chNN_*.py` | the computations of Chapter NN (Ch 0–24) |
| `code/appD_adam.py` | the NumPy Adam example of Appendix D |
| `code/students.py` | the eight students and the linear model |
| `code/net221.py` | the 2-2-1 tanh network used from Chapter 9 on |
| `code/optimizers.py`, `code/winenet.py` | optimisers and the Wine network of Chapter 24 |
| `code/cache/` | cached sweep results for Chapter 24 |
| `gen_fig.py` | draws every Matplotlib figure (`python gen_fig.py ch05`) |
| `outputs/chNN.txt` | the expected output of each chapter script |
| `check.py` | reruns all scripts and compares them with `outputs/` |

## Run

```bash
pip install -r requirements.txt
cd code
python ch01_walkthrough.py
```

To check that everything reproduces: `python check.py`.

In Google Colab: `!git clone https://github.com/jamhuri-tech/buku-understanding-nn-optimization`,
then `%cd buku-understanding-nn-optimization/code` and run a script with `!python`.

The Wine data (Aeberhard & Forina 1991, UCI, CC BY 4.0, doi:10.24432/C5PC7J)
are loaded through scikit-learn. The tag `edition-1` marks the version used
for the first edition.

## License

0BSD. Use, copy, and modify freely.

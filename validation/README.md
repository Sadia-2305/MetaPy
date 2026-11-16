# Validation and paper materials

Everything reported in the MetaPy paper can be regenerated from this folder (Python ≥ 3.10, the packages in
`requirements.txt`). Run from the repository root:

```bash
python validation/run_derivation_validation.py   # derivation engine vs independent formulas (110 simulated comparisons)
python validation/make_worked_example.py         # builds the synthetic mixed-reporting example
python validation/run_worked_example.py          # analyses it (results/worked_example.json)
python validation/run_validation.py              # pooling vs metafor reference values, advanced modules, CR2 coverage (~5 min)
python validation/make_figures.py                # Figures 1-3 (PNG and 600-dpi TIFF)
```

`independent.py` is a separate, minimal implementation of the key formulas (Hedges' g, Morris PPC2 and its variance,
REML/DL/PM, Hartung-Knapp) written without MetaPy code, used as the comparator. The BCG data (`data/bcg.csv`) are the
13 trials of Colditz et al. (1994) as distributed with metafor (`dat.bcg`). All other data are simulated (seeds in the
scripts) and do not describe real studies.

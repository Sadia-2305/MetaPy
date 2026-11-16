# Comparison with R

`metafor_check.R` runs metafor, clubSandwich, netmeta and bayesmeta on the same data files that MetaPy uses
(`validation/data/`) and writes `metafor_results.json`. `compare_with_r.py` computes the same quantities with MetaPy
and writes `validation/results/r_comparison.csv` and `r_comparison_summary.json`.

1. In Google Colab choose *Runtime > Change runtime type > R*, paste `metafor_check.R` into a cell and run it.
2. Download `metafor_results.json` from the Files panel.
3. Run `python validation/r_check/compare_with_r.py metafor_results.json`.

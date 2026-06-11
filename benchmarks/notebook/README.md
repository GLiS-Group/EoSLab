# Benchmark explorer notebook

Interactive exploration of the `bench_ideal_models` Google Benchmark results.

## Setup

The Python environment is managed by a local venv (kept out of git):

```bash
cd benchmarks/notebook
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name eoslab-bench \
    --display-name "EoSLab Bench (.venv)"
```

## Use

1. Build the benchmark (Clang + Enzyme; use the `release` preset for real numbers):

   ```bash
   cmake --preset release -DBUILD_BENCHMARKS=ON
   cmake --build build/release --target bench_ideal_models
   ```

2. Open `benchmark_explorer.ipynb` and select the **EoSLab Bench (.venv)** kernel:

   ```bash
   .venv/bin/jupyter lab benchmark_explorer.ipynb
   ```

3. Run the cells top to bottom. The notebook:
   - runs the executable and captures `results/bench_output.txt` (console) and
     `results/bench_results.json` (parsed);
   - parses the JSON into a tidy `pandas.DataFrame`;
   - draws an interactive Plotly plot — dropdowns pick the **calculation** and
     **model**; x = `N`, y = CPU time; four lines (Static/Dynamic × SoA/AoS);
     forward and reverse modes shown together for the gradient calculations;
     click legend entries to toggle series.

Edit `EXE_PATH`, `MIN_TIME`, or set `RUN_BENCHMARK = False` (to reuse an existing
capture) in the first code cell.

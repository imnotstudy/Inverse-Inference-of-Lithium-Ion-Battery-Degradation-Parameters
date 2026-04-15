# Inverse Inference of Lithium-Ion Battery Degradation Parameters via Physical Modeling and Data-Driven Machine Learning

This repository provides the implementation and supporting scripts for *Inverse Inference of Lithium-Ion Battery Degradation Parameters via Physical Modeling and Data-Driven Machine Learning*. 
Our research is addressed by segmenting battery degradation into a few stages. This work develops a physically informed inverse inference framework for lithium-ion battery degradation modeling. By leveraging the characteristic two-stage degradation behavior, it links observable features in experimental capacity-fade curves to internal degradation parameters through PyBaMM simulations and lightweight machine learning models, enabling efficient and interpretable parameter inference from aging data.

<p align="center">
  <a href="Figures/Multistage_Segmentation.pdf">
    <img src="Figures/Multistage_Segmentation.png" alt="Multistage segmentation" width="960">
  </a>
</p>



## Installation

You can install by following these steps:

```bash
git clone git@github.com:imnotstudy/Inverse-Inference-of-Lithium-Ion-Battery-Degradation-Parameters.git
```


## Usage

### Benchmark Reproduction

The benchmark entrypoint is:

```bash
python Scripts/run_benchmark.py --config configs/benchmark.yaml
```

The default config expects a prediction table at `data/benchmark/xgb_cap_50_sample.csv`. Replace that file path in `configs/benchmark.yaml` with your own benchmark input table when you are ready to run the full workflow.

The benchmark input CSV must contain these columns:

- `id`
- `sei`
- `plating`
- `crate`
- `b1`
- `b2`
- `pred_scale`

Running the benchmark writes outputs under `results/benchmark/`:

- `checkpoints/` for serialized intermediate `pybamm` solutions
- `summaries/` for per-case summary CSV files
- `logs/` for execution logs

## Data Availability

This repository includes small sample inputs and generated checkpoints only for testing the implementation pipeline.

These bundled data samples and checkpoints are not intended to reproduce the manuscript results. For a full scientific reproduction study, replace the sample input file with the appropriate benchmark dataset and regenerate outputs using your target configuration.


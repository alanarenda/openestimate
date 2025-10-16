# OpenEstimate 

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**OpenEstimate** is a multi-domain benchmark for evaluating language models on probabilistic estimation, a specific form of reasoning under uncertainty.

Real-world LM deployments in domains like healthcare, finance, and other forms of knowledge work require models to handle incomplete information and quantify uncertainty. Yet, most LM evaluations focus on well-defined problems with clear answers. OpenEstimate addresses this gap by testing models on probabilistic estimation tasks where they must synthesize background knowledge into accurate, well-calibrated Bayesian priors.

## Overview 🎯 

Language models have access to vast amounts of knowledge, but their ability to reason probabilistically about uncertain outcomes remains poorly understood. OpenEstimate evaluates LMs on aspects of:

- **🎲 Probabilistic Reasoning**: Estimating distributions over uncertain quantities
- **🧩 Background Knowledge Synthesis**: Synthesizing background knowledge into distributional estimates
- **📊 Calibration**: Producing well-calibrated uncertainty estimates (not just accurate point predictions)

The benchmark assesses both the **accuracy** and **calibration** of LM-elicited priors, quantifying their usefulness relative to samples from the true distribution.

## Key Findings 🔍

Across six contemporary language models, we find:
- LM-elicited priors are often inaccurate and overconfident
- Performance improves modestly with different elicitation protocols
- Changes in sampling strategy, reasoning effort, or prompt design have limited impact

---

## Datasets 📚

OpenEstimate includes three diverse domains with real-world data:

| Dataset | Domain | Description | Variables |
|---------|--------|-------------|-----------|
| **🏥 NHANES** | Healthcare | National Health and Nutrition Examination Survey data with health metrics and demographic information | Health outcomes, biomarkers, lifestyle factors |
| **💼 Glassdoor** | Employment | Company and employment data including salaries, industries, and workplace metrics | Compensation, company characteristics, job roles |
| **💰 PitchBook** | Finance | Startup and venture capital data with funding rounds, valuations, and company metrics | Funding amounts, valuations, company growth |

Each dataset includes:
- **✓** Ground truth distributions computed from observational data
- **✓** Variable descriptions in natural language
- **✓** Conditioning information of varying complexity (1-3 conditions)

---

## Evaluation Design ⚙️ 

### Elicitation Protocols

Multiple methods for eliciting distributional beliefs from language models:

- **Direct**: Model directly specifies distribution parameters (mean, variance)
- **Quantile-based**: Model provides quantiles (e.g., 10th, 50th, 90th percentiles), which are fit to a distribution
- **Mean-Variance**: Model separately estimates mean and variance

### System Prompts

Different expert personas to test prompt sensitivity:

- **Base**: Neutral helpful assistant with domain expertise
- **Conservative**: Explicitly instructed to provide conservative estimates
- **Superforecaster**: Prompted to follow forecasting best practices (à la Philip Tetlock)

### Evaluation Metrics

Comprehensive metrics for assessing prior quality:

- **Mean Absolute Error (MAE)**: Point estimate accuracy
- **Expected Calibration Error (ECE)**: Calibration of probabilistic predictions
- **Uncertainty-Accuracy Correlation**: Relationship between uncertainty estimates and accuracy of predictions

### Distribution Types

Support for multiple distribution families:

- **Gaussian/Normal**: For unbounded continuous variables
- **Beta**: For bounded continuous variables (e.g., proportions)

### Baselines

We compare LM-elicited priors against statistical baselines that are computed by sampling N examples from the true distribution and updating an uninformative prior with those examples. This enables a comparison of LM performance against different numbers of samples from the true distribution. 

---

## Installation 🛠️

### Prerequisites

- Python 3.8 or higher
- API keys for LM providers (OpenAI, Together AI, etc.)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/openestimate.git
   cd openestimate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   TOGETHER_API_KEY=your-together-api-key
   ```

---

## Quick Start 🚀

### Running Experiments and Analyzing Results

1. **Generate experiment specifications**:
   ```bash
   cd ~/openestimate/experiments
   python generate_specs.py
   python generate_run_scripts.py 
   cd dataset_name/experiment_name # e.g., cd glassdoor/model_family_comparison
   ./run_experiments_generated.sh 
   ```
2. **Analyze results**:
   ```bash
   cd ~/openestimate/analysis
   python run_analysis.py \
     --datasets glassdoor,nhanes,pitchbook \
     --output_dir analysis_results
   ```

### Generating Custom Benchmarks
You can extend OpenEstimate with new datasets. See [data/readme.md](data/readme.md) for details on how to do this. 

---

## Repository Structure 📁

```
openestimate/
├── data/                      # Data generation and processing
│   ├── generate.py           # Main variable generation pipeline
│   ├── glassdoor.py          # Glassdoor dataset processing
│   ├── nhanes_generation.py  # NHANES dataset processing
│   ├── pitchbook.py          # PitchBook dataset processing
│   ├── compute_posteriors.py # Ground truth computation
│   ├── baselines/            # Baseline priors
│   └── variables/            # Generated benchmark variables
│
├── elicitation/              # Prior elicitation from language models
│   ├── src/
│   │   ├── main.py          # Main elicitation script
│   │   ├── elicitation.py   # Core elicitation logic
│   │   ├── fit_priors.py    # Prior fitting methods
│   │   ├── clients.py       # LM API clients
│   │   └── utils.py         # Utility functions
│   └── prompts/             # Elicitation protocol templates
│
├── experiments/              # Experiment configurations
│   ├── generate_specs.py    # Generate experiment specifications
│   ├── glassdoor/           # Glassdoor experiments
│   ├── nhanes/              # NHANES experiments
│   └── pitchbook/           # PitchBook experiments
│
├── analysis/                 # Results analysis and visualization
│   ├── run_analysis.py      # Main analysis script
│   ├── compare_models.py    # Cross-model comparisons
│   ├── ablations.py         # Ablation studies
│   ├── plotting.py          # Visualization utilities
│   └── utils.py             # Analysis utilities
```

---

## Citation 📝

If you use OpenEstimate in your research, please cite:

```bibtex
@article{openestimate2024,
  title={OpenEstimate: A Benchmark for Evaluating Language Models on Probabilistic Estimation},
  author={[Authors]},
  journal={[Venue]},
  year={2024},
  url={https://github.com/your-username/openestimate}
}
```

---

## Contributing 🤝

We welcome contributions! Areas of particular interest:

- Additional datasets and domains
- New elicitation protocols
- Alternative distribution families
- Improved evaluation metrics
- Calibration and uncertainty quantification methods

Please open an issue or submit a pull request.

---

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact 📧

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors at [email]


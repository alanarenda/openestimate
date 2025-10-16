# Benchmark Generation Pipeline

This document provides an overview of the benchmark generation pipeline, detailing the steps required to preprocess data, compute ground truths, generate variables, and integrate results into a structured output. The pipeline is designed to be modular and extensible, allowing for easy adaptation to new datasets and benchmarking tasks.

---

## How to Run Variable Generation with Current Benchmarks

The benchmark generation pipeline includes pre-configured datasets and benchmarks. To generate variables and baselines for the current benchmarks, follow these steps:

1. **Install Dependencies**:
   - Ensure all required libraries are installed:
     ```bash
     pip install -r requirements.txt
     ```
   - Note that you will need to download the Pitchbook data (VC Global) yourself. We accessed the data through WRDS.

2. **Run the Benchmark Generation**:
   - Use the `generate()` function to create variables and baselines for the pre-configured datasets:
     ```python
     from data_refactored.generate import generate

     generation_config = {
         "random_seed": 42,
         "target_num_single_condition_vars": 10,
         "target_num_double_condition_vars": 5,
         "target_num_triple_condition_vars": 3,
     }

     variables, baselines = generate("glassdoor", generation_config)
     ```

3. **Save the Results**:
   - The generated variables and baselines can be saved in JSON or CSV format for reproducibility:
     ```python
     import json

     with open("variables.json", "w") as f:
         json.dump(variables, f, indent=4)

     with open("baselines.json", "w") as f:
         json.dump(baselines, f, indent=4)
     ```

4. **Use the Results**:
   - The generated variables and baselines can be used for benchmarking tasks, such as evaluating machine learning models or statistical methods.

---

## Extending the Benchmark

The benchmark generation pipeline is designed to be extensible, allowing you to add new datasets and benchmarks. The process for extending the benchmark involves the following steps:

### 1. **Prepare the New Dataset**

1. **Load and Preprocess the Data**:
   - Implement a function to load and clean the new dataset. This function should:
     - Remove missing or irrelevant values.
     - Standardize column names and formats.
     - Extract and transform key variables.

2. **Define Target Variables**:
   - Identify the boolean and continuous variables to be used in the benchmark.
   - Update the `target_variables_discrete` and `target_variables_continuous` lists to include the new variables.

---

### 2. **Compute Ground Truths**

1. **Calculate Baseline Statistics**:
   - Use the `compute_ground_truths()` function to compute the mean, standard deviation, and standard error for each variable.

2. **Store Ground Truths**:
   - Save the computed statistics in a dictionary for use in variable generation.

---

### 3. **Generate Conditions**

1. **Define Conditions**:
   - Create filters for the data based on variable values (e.g., quartiles, boolean flags).

2. **Generate Natural Language Descriptions**:
   - Use the `prep_cond_phrases()` function to create human-readable descriptions for the conditions.

3. **Organize Conditions**:
   - Store the conditions in a dictionary for use in variable generation.

---

### 4. **Generate Variables**

1. **Create Variables by Difficulty**:
   - Use the `create_variables_by_difficulty()` function to generate variables of varying difficulty levels:
     - **Easy**: Single condition applied.
     - **Medium**: Two conditions applied.
     - **Hard**: Three or more conditions applied.

2. **Validate Variables**:
   - Ensure that the conditions shift the variable's point estimate significantly.

3. **Generate Descriptions**:
   - Combine base phrases and condition phrases to create natural language descriptions for each variable.

---

### 5. **Integrate the New Dataset**

1. **Add a New Generator Function**:
   - Implement a new generator function for the dataset (e.g., `generate_new_dataset()`).

2. **Register the Generator**:
   - Add the new generator to the `generators` dictionary in `generate.py`.

3. **Run the Pipeline**:
   - Call the `generate()` function with the new dataset name to generate variables and baselines.

---

### Example: Adding a New Dataset

```python
def generate_new_dataset(generation_config):
    random.seed(generation_config['random_seed'])
    df = load_new_dataset()  # Implement this function for the new dataset
    base_phrases = prep_base_phrases(df)
    gt = compute_ground_truths(df)
    all_possible_conditions = prepare_conditions(df)  # Implement this for the new dataset

    variables = create_variables_by_difficulty(
        df,
        gt,
        all_possible_conditions,
        generation_config['target_num_single_condition_vars'],
        generation_config['target_num_double_condition_vars'],
        generation_config['target_num_triple_condition_vars'],
    )

    baselines = generate_baselines(df, variables)  # Implement this for the new dataset
    return variables, baselines
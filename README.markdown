# NOMAD: Nuclear Optimization with Machine-learning-Accelerated Design (A Genetic Algorithm with KNN/Random Forest/Ridge for Fuel Pattern Optimization)
[![Powered by](https://img.shields.io/badge/Powered%20by-Genetic%20Algorithm-purple.svg)](https://en.wikipedia.org/wiki/Genetic_algorithm)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![OpenMC](https://img.shields.io/badge/OpenMC-Required-green.svg)](https://docs.openmc.org/)

NOMAD is a sophisticated tool for optimizing nuclear reactor core fuel loading patterns. It leverages a **Genetic Algorithm (GA)** coupled with **machine learning (ML)** models to efficiently determine fuel assembly enrichment arrangements that achieve a target **multiplication factor (k_eff)** while minimizing the **Power Peaking Factor (PPF)**. This ensures safe, efficient, and compliant reactor operation.

By integrating ML models as high-speed surrogates for computationally expensive neutron transport simulations (e.g., via OpenMC), NOMAD significantly accelerates the optimization process while maintaining accuracy.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Step 1: Define Fuel Materials and Assemblies](#step-1-define-fuel-materials-and-assemblies)
  - [Step 2: Set Up Tallies for PPF Calculation](#step-2-set-up-tallies-for-ppf-calculation)
  - [Step 3: Identify Central vs. Outer Assemblies](#step-3-identify-central-vs-outer-assemblies)
  - [Step 4: Configure `config.ini`](#step-4-configure-configini)
  - [Step 5: Configure `setup_fuel.ini`](#step-5-configure-setup_fuelini)
  - [Step 6: Run the Optimizer](#step-6-run-the-optimizer)
  - [Step 7: Monitor Progress with the Live Dashboard](#step-7-monitor-progress-with-the-live-dashboard)
- [Example Configuration](#example-configuration)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

NOMAD optimizes nuclear reactor core designs by:

- **Target**: Achieving a specific k_eff while minimizing PPF.
- **Method**: Combining a Genetic Algorithm with ML-based surrogates for fast fitness evaluation.
- **Simulation**: Using OpenMC for high-fidelity neutron transport calculations.
- **Iterative Improvement**: Continuously refining ML models with new simulation data.

This hybrid approach enables rapid exploration of fuel enrichment configurations, making it a powerful tool for nuclear reactor core design.

---

## How It Works

1. **Initial Data Generation**: Run OpenMC simulations for a diverse set of fuel enrichment configurations to create a baseline dataset.
2. **ML Model Training**:
   - **k_eff Interpolator**: A K-Nearest Neighbors (KNN) regressor predicts k_eff for a given fuel pattern.
   - **PPF Interpolator**: Predicts PPF using KNN, Random Forest, or Ridge regression (configurable).
3. **Genetic Algorithm Cycle**: The GA evolves a population of fuel loading patterns over thousands of generations, evaluating fitness using ML predictors for speed.
4. **Verification**: The best fuel pattern is verified with a full OpenMC simulation.
5. **Iterative Improvement**: Verification results are added to the dataset, and ML models are retrained to improve accuracy for subsequent GA cycles.

---

## Requirements

### Software Dependencies
- **Python 3.8+** with the following packages:
  ```bash
  pip install numpy scipy pandas matplotlib scikit-learn
  ```
- **OpenMC**: A working installation is required for physics simulations. See the [OpenMC documentation](https://docs.openmc.org/) for installation instructions.

### Input Files
Ensure the following OpenMC input files are in the same directory as `RunOptimizer.ipynb`:
- `geometry.xml`
- `materials.xml`
- `settings.xml`
- `tallies.xml`

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/XxNILOYxX/nomad.git
   cd nomad
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install OpenMC following the [official instructions](https://docs.openmc.org/en/stable/quickinstall.html).
4. Ensure all OpenMC input files are correctly configured and placed in the root directory.

---

## Step 1: Define Fuel Materials and Assemblies

This is the most critical step in setting up your model for NOMAD. The optimizer works by individually adjusting the enrichment of **every single fuel assembly**. For this to work, your OpenMC model must be built with a specific structure:

**Each fuel assembly in your core must be represented by its own unique `material` and its own unique `cell` (or `universe`).**

Think of it like giving each assembly a unique ID that the program can find and modify. If you define one material and use it for multiple assemblies, the optimizer will not be able to assign different enrichment values to them.

### How to Structure Your Model

1. **Unique Materials**: If your core has 150 fuel assemblies, you must create 150 distinct `<material>` blocks in your `materials.xml` file. It's essential that their `id` attributes are sequential (e.g., 3, 4, 5, ..., 152).

2. **Unique Cells/Universes**: Similarly, in your `geometry.xml`, each of these unique materials must fill a unique cell that represents the fuel region of an assembly.

### Example Scenario (150 Assemblies)

Imagine your model's material IDs start at 3. Your `materials.xml` must be structured as follows:

```xml
<material depletable="true" id="3" name="Fuel for Assembly 1">
</material>
<material depletable="true" id="4" name="Fuel for Assembly 2">
</material>
...
<material depletable="true" id="152" name="Fuel for Assembly 150">
</material>
```

In your `config.ini`, you would then set:

```ini
num_assemblies = 150
start_id = 3
```

**Pro-Tip**: When generating your model files programmatically (e.g., in a Jupyter Notebook), always use the "Restart Kernel and Clear All Outputs" command before running your script. This prevents old data from being cached and ensures your material and cell IDs are created fresh and correctly, avoiding hard-to-debug errors.

### Example Code for Creating Individual Fissile Materials

Use the following code as inspiration and modify it for your own reactor core:

```python
all_materials_list = []
# You can adjust this number as needed
num_assemblies = 150
print("Creating unique fuel materials...")

# This loop creates variables fuel_1, fuel_2, ... fuel_150
for i in range(1, num_assemblies + 1):
    # Define the material object
    fuel_material = openmc.Material(name=f'Fissile fuel Assembly {i}')
    fuel_material.add_nuclide('U235', use your weight fraction, 'wo')
    fuel_material.add_nuclide('U238', use your weight fraction, 'wo')
    fuel_material.add_nuclide('Pu238', use your weight fraction, 'wo')
    fuel_material.add_nuclide('Pu239', use your weight fraction, 'wo')
    fuel_material.add_nuclide('Pu240', use your weight fraction, 'wo')
    fuel_material.add_nuclide('Pu241', use your weight fraction, 'wo')
    fuel_material.add_nuclide('Pu242', use your weight fraction, 'wo')
    fuel_material.add_element('Zr', use your weight fraction, 'wo')
    fuel_material.set_density('g/cm3', use your density)
    fuel_material.depletable = True
    fuel_material.temperature = fuel_temperature
    # This line dynamically creates a variable named fuel_1, fuel_2, etc.
    globals()[f'fuel_{i}'] = fuel_material
    
    # Add the new material to our list by accessing the dynamically created global variable
    all_materials_list.append(globals()[f'fuel_{i}'])

# Export all materials to a single XML file
materials = openmc.Materials(all_materials_list)
materials.export_to_xml()
```

### Example Code for Creating Individual Fissile Assemblies

```python
# This loop creates universes fa_inner_univ_1, fa_inner_univ_2, ...
for i in range(1, num_assemblies + 1):   
    # 1. Retrieve the unique fuel material for this specific assembly iteration
    current_inner_fuel = globals()[f'fuel_{i}']

    # 2. Define all cells for this assembly using local variables for simplicity
    clad_cell = openmc.Cell(name=f'clad_cell_{i}', fill=cladding, region=clad_region)
    sodium_cell = openmc.Cell(name=f'sodium_cell_{i}', fill=coolant, region=moderator_region)
    fuel_cell = openmc.Cell(name=f'fuel_cell_{i}', fill=current_inner_fuel, region=fuel_region)
    ht_cell = openmc.Cell(name=f'ht_cell_{i}', fill=cladding, region=ht_region)
    Na_cell = openmc.Cell(name=f'Na_cell_{i}', fill=coolant, region=Na_region)
    He_cell = openmc.Cell(name=f'He_cell_{i}', fill=helium, region=He_region)
    stru_cell = openmc.Cell(name=f'stru_cell_{i}', fill=cladding, region=stru_region)

    # 3. Define the pin universes using the cells created above
    inner_core_fuel = openmc.Universe(name=f'inner_core_fuel_{i}', cells=[stru_cell, Na_cell, He_cell, fuel_cell])
    inner_fuel_cell = openmc.Cell(name=f'inner_fuel_cell_{i}', fill=inner_core_fuel, region=fuel_region)
    inner_u = openmc.Universe(name=f'inner_pin_universe_{i}', cells=(inner_fuel_cell, clad_cell, sodium_cell, ht_cell))

    # 4. Create the hexagonal lattice for this assembly
    in_lat = openmc.HexLattice(name=f'inner_assembly_{i}')
    in_lat.center = (0., 0.)
    in_lat.pitch = (pin_to_pin_dist,)
    in_lat.orientation = 'y'
    in_lat.outer = sodium_mod_u

    # Fill the lattice rings with this assembly's specific pin universe ('inner_u')
    in_lat.universes = [
        [inner_u] * 54, [inner_u] * 48, [inner_u] * 42, [inner_u] * 36, [inner_u] * 30,
        [inner_u] * 24, [inner_u] * 18, [inner_u] * 12, [inner_u] * 6, [inner_u] * 1
    ]

    # 5. Define the outer structure of the assembly
    main_in_assembly = openmc.Cell(name=f'main_in_assembly_{i}', fill=in_lat, region=prism_inner & -top & +bottom)
    assembly_sleave = openmc.Cell(name=f'assembly_sleave_{i}', fill=cladding, region=prism_middle & ~prism_inner & -top & +bottom)
    outer_sodium = openmc.Cell(name=f'outer_sodium_{i}', fill=coolant, region=prism_outer & ~prism_middle & -top & +bottom)

    # 6. Create the final, complete universe for this fuel assembly
    final_assembly_universe = openmc.Universe(name=f'fa_inner_univ_{i}', cells=[main_in_assembly, assembly_sleave, outer_sodium])

    # 7. Dynamically create the global variable (fa_inner_univ_1, fa_inner_univ_2, etc.)
    globals()[f'fa_inner_univ_{i}'] = final_assembly_universe
```

### Step 2: Set Up Tallies for PPF Calculation

To minimize the Power Peaking Factor (PPF), the optimizer needs to measure the power (fission rate) in every single fuel assembly. This is done using an OpenMC tally.

You must create a tally that individually measures the fission rate in each of the fuel cells you defined in Step 1.

**How to Create the Tally**:

1. **Get Fuel Cell IDs**: First, you need a list of all the numerical IDs of the cells that contain fuel. The way you get this list depends on how your `geometry.xml` is structured.
2. **Create a CellFilter**: This filter tells OpenMC to tally results only for the specific cell IDs you provide.
3. **Define the Tally**: Create a tally that uses this filter and scores fission. It is crucial that the name you assign to this tally matches the one specified in your `config.ini`.

**Example Python Snippet for Tally Creation**:

Let's assume your geometry is built such that the first fuel cell has an ID of 5, and each subsequent fuel cell ID is 13 numbers higher.

```python
import openmc

# 1. Get Fuel Cell IDs (This is an example, you must adapt it to your geometry)
# For 150 assemblies, starting at ID 5 with an interval of 13.
fuel_cell_ids = [5 + 13*i for i in range(150)]

# 2. Create a CellFilter with these IDs
cell_filter = openmc.CellFilter(fuel_cell_ids)

# 3. Define the Tally
# The name "fission_in_fuel_cells" is the default in config.ini
fission_tally = openmc.Tally(name="fission_in_fuel_cells")
fission_tally.filters = [cell_filter]
fission_tally.scores = ["fission"]

# Export the tally to the XML file
tallies = openmc.Tallies([fission_tally])
tallies.export_to_xml()
```

By setting it up this way, the resulting `statepoint.h5` file will contain the exact data NOMAD needs to calculate the PPF and guide the optimization.

## Step 3: Identify Central vs. Outer Assemblies

After running a baseline OpenMC simulation with a uniform enrichment profile, the next crucial step is to analyze the resulting power distribution. This analysis allows for the differentiation between central and outer fuel assemblies based on their power output, a key factor in enrichment zoning. The `num_central_assemblies` parameter, which defines the boundary between these two zones, is determined from this analysis.

### Power Peaking Factor (PPF) Calculation

A Python script is utilized to process the simulation output and calculate the Power Peaking Factor (PPF), which is the ratio of the maximum power produced in a single fuel cell to the average power across all fuel cells. This script also exports the normalized power for each fuel cell, which is essential for identifying high-power regions.

#### Python Script for PPF Calculation

```python
import openmc
import numpy as np
import pandas as pd
import glob

# Load the latest statepoint file to access simulation results
statepoint_file = sorted(glob.glob("statepoint.*.h5"))[-1]
sp = openmc.StatePoint(statepoint_file)

# Retrieve the fission tally, which contains power data
tally = sp.get_tally(name="fission_in_fuel_cells")
df = tally.get_pandas_dataframe()

# Extract fission rates and corresponding cell IDs
fission_rates = df['mean'].values
cell_ids = df['cell'].values

# Calculate the Power Peaking Factor (PPF)
avg_power = np.mean(fission_rates)
max_power = np.max(fission_rates)
ppf = max_power / avg_power
print(f"Power Peaking Factor (PPF): {ppf:.4f}")

# Compile and export the results to a CSV file for analysis
results_df = pd.DataFrame({
    'Fuel Cell ID': cell_ids,
    'Fission Rate': fission_rates,
    'Normalized Power': fission_rates / avg_power
}).sort_values(by='Fuel Cell ID', ascending=True)

results_df.to_csv("fission_rates_and_ppf.csv", index=False)
print("Fission rate data exported to fission_rates_and_ppf.csv")
```

Upon executing the script, open the generated `fission_rates_and_ppf.csv` file. By examining the `Normalized Power` column, you can identify the fuel assemblies operating at the highest power levels. These are typically located in the central region of the reactor core. For instance, after analysis, you might determine that the inner 54 assemblies exhibit the highest power output. This number would then be used to set `num_central_assemblies = 54` in the `config.ini` file.

### Configuring Enrichment Ranges and Initial Sampling

To optimize the enrichment zoning, it is necessary to define the search space for the plutonium content in both the central and outer regions of the core within the `config.ini` file.

#### Determining `central_range` and `outer_range`

The selection of `central_range` and `outer_range` is highly dependent on the specific reactor design and the goals of the optimization (e.g., power flattening, maximizing fuel cycle length). These ranges define the lower bound, upper bound, and step size for the enrichment percentages to be evaluated by the optimization algorithm.

For example, consider a Sodium-Cooled Fast Reactor (SFR) with a core-wide average plutonium content of 15.99%. To flatten the power profile, one might explore lower enrichments in the high-power central region and higher enrichments in the lower-power outer region. A potential configuration could be:

```ini
central_range = 14.0, 15.5, 0.1
outer_range = 14.5, 18.0, 0.1
```

It is critical to understand that these are starting points. Fine-tuning these ranges through iterative analysis is essential to discover the optimal enrichment distribution for your specific reactor design.

#### Setting the `initial_samples`

The `initial_samples` parameter in `config.ini` specifies the number of initial configurations to be simulated. It is recommended to use a value of at least 100. A sufficiently large and well-distributed set of initial samples ensures that the optimization algorithm thoroughly explores the defined search space. You can manually provide these initial configurations to guarantee comprehensive coverage of the possible enrichment combinations within your defined `central_range` and `outer_range`.

### Step 4: Configure `config.ini`
Edit `config.ini` to match your reactor model. Key parameters include:
- `[simulation]`: `target_keff`, `num_assemblies`, `num_central_assemblies`, `start_id`, `fission_tally_name`.
- `[enrichment]`: `central_range`, `outer_range`, `initial_configs` (recommended for controlled initial data).
- `[ga]`: Adjust `population_size`, `generations_per_openmc_cycle`, etc., for performance tuning.

**Example `config.ini`** (see [Example Configuration](#example-configuration) for a full sample).

### Step 5: Configure `setup_fuel.ini`

This file tells the optimizer which fissile material you are optimizing.

> **Important Note on Current Limitations:**
> The code can currently only handle one fissile enrichment strategy at a time. Your options are:
> * Optimize for **U-233** only.
> * Optimize for **U-235** only.
> * Optimize for a **Plutonium vector**.
>
> If you choose Plutonium, you must set all relevant Pu isotopes to `1` in the `[fissile]` section and define their relative weight fractions in the `[plutonium_distribution]` section. Future versions may add more flexibility to this part.

**Example `setup_fuel.ini` for U-235 optimization:**
```ini
[general]
slack_isotope = U238

[fissile]
U235 = 1
U233 = 0
Pu239 = 0
Pu240 = 0
Pu241 = 0
Pu242 = 0

[plutonium_distribution]
# This section is ignored if only U-235 is selected
Pu239 = 0.6
Pu240 = 0.25
Pu241 = 0.1
Pu242 = 0.05
```

### Step 6: Run the Optimizer
1. Open `RunOptimizer.ipynb` in Jupyter Notebook/Lab.
2. Run the first cell to import libraries.
3. Run the second cell to initialize and start `MainOptimizer.run()`.
4. The optimizer will generate initial data (if needed) and begin GA cycles.

### Step 7: Monitor Progress with the Live Dashboard
Visualize GA progress in real-time:
1. Start a local web server:
   ```bash
   python3 -m http.server 8000 --bind 0.0.0.0
   ```
2. Open `http://localhost:8000/fitness.html` in a browser.
3. Data appears after the first GA cycle creates `data/ga_checkpoint.json`.

---

## Example Configuration

**Example `config.ini`**:
```ini
[ga]
population_size = 1500
generations_per_openmc_cycle = 1000
mutation_rate = 0.20
max_mutation_rate = 0.30
crossover_rate = 0.85
max_crossover_rate = 0.90
elitism_count = 10
stagnation_threshold = 50
diversity_threshold = 0.3
tournament_size = 20

[ga_tuning]
log_frequency = 100
diversity_sample_size = 50
keff_penalty_factor = 10
high_keff_diff_threshold = 0.05
med_keff_diff_threshold = 0.02
smart_mutate_increase_factor = 1.5
smart_mutate_decrease_factor = 0.8

[enrichment]
central_range = 14.0, 15.5, 0.1
outer_range = 14.5, 18.0, 0.1
initial_samples = 100
initial_configs = [(14.0, 14.5), (14.0, 15.0), (14.0, 16.0), (14.0, 17.0), (14.0, 18.0), (15.5, 14.5), (15.5, 15.0), (15.5, 16.0), (15.5, 17.0), (15.5, 18.0)]

[simulation]
target_keff = 1.12437
keff_tolerance = 0.01
num_cycles = 300
num_assemblies = 150
num_central_assemblies = 54
start_id = 3
materials_xml_path = materials.xml
fission_tally_name = fission_in_fuel_cells
ppf_interp_file = data/ppf_interp_data.json
keff_interp_file = data/keff_interp_data.json
checkpoint_file = data/ga_checkpoint.json
statepoint_filename_pattern = statepoint.*.h5
openmc_retries = 2
openmc_retry_delay = 5

[hardware]
cpu = 1
gpu = 1

[interpolator]
max_keff_points = 100000
max_ppf_points = 100000
min_interp_points = 20
min_validation_score = 0.05
regressor_type = random_forest
n_neighbors = 7
```
# Disclaimer
GPU Acceleration: The code includes a pipeline to use NVIDIA GPUs for training the ML models via the RAPIDS cuML library. While this is fully implemented, current testing shows a negligible performance difference compared to the multi-threaded CPU implementation using scikit-learn. For this reason, detailed installation instructions for the GPU environment are not provided at this time. Future versions will aim to improve the code to better utilize GPU capabilities.

---

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork the repository:** Create your own copy of the project.
2.  **Create a new branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **Make your changes:** Implement your new feature or fix the bug.
4.  **Commit your changes:**
    ```bash
    git commit -m "feat: Add your descriptive commit message"
    ```
5.  **Push to your branch:**
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Open a Pull Request:** Submit a pull request from your forked repository to the main branch of this project. Please provide a clear description of the changes you have made.

We value your input and will review your contributions as soon as possible.

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more details.


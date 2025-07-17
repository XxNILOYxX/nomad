# NOMAD: Nuclear Optimization with Machine-learning-Accelerated Design (A Genetic Algorithm (GA) / Discrete Particle Swarm Optimization/ Hybrid (GA-PSO) with KNN/Random Forest/Ridge for Fuel Pattern Optimization)
[![Powered by GA](https://img.shields.io/badge/Powered%20by-Genetic%20Algorithm-purple.svg)](https://en.wikipedia.org/wiki/Genetic_algorithm)
[![Powered by PSO](https://img.shields.io/badge/Powered%20by-Particle%20Swarm%20Optimization-orange.svg)](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![OpenMC](https://img.shields.io/badge/OpenMC-Required-green.svg)](https://docs.openmc.org/)


NOMAD is a sophisticated tool for optimizing nuclear reactor core fuel loading patterns. It leverages a **Genetic Algorithm (GA) / Discrete Particle Swarm Optimization / Hybrid (GA-PSO)** coupled with **machine learning (ML)** models to efficiently determine fuel assembly enrichment arrangements that achieve a target **multiplication factor (k_eff)** while minimizing the **Power Peaking Factor (PPF)**. This ensures safe, efficient, and compliant reactor operation.

By integrating ML models as high-speed surrogates for computationally expensive neutron transport simulations (e.g., via OpenMC), NOMAD significantly accelerates the optimization process while maintaining accuracy.
![Image Alt](https://github.com/XxNILOYxX/nomad/blob/main/images/NOMAD.png?raw=true)

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
- [Results](#results)
- [DPSO & Hybrid mode](#dpso)
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

1.  **Initial Data Generation**: Run OpenMC simulations for a diverse set of fuel enrichment configurations to create a baseline dataset.

2.  **ML Model Training**:
    * **$k_{eff}$ Interpolator**: A K-Nearest Neighbors (KNN) regressor predicts $k_{eff}$ for a given fuel pattern.
    * **PPF Interpolator**: Predicts the Power Peaking Factor (PPF) using KNN, Random Forest, or Ridge regression (configurable).

3.  **Choosing the PPF Predictor (Experimental)** 🧪
    The optimal choice for the PPF predictor is not fixed. During testing, sometimes **Random Forest** performs better than **KNN**, and sometimes the opposite is true. For best results, you should run the full optimization process with both models and use the superior result.
    > **Pro Tip:**
    > 1.  First, run the entire optimization with `knn` set as the PPF regression model.
    > 2.  Once complete, rename the final checkpoint file in the `data/` directory (e.g., from `ga_checkpoint.json` to `ga_checkpoint_knn.json`).
    > 3.  Next, change the model in your configuration file to `random_forest` and run the optimization again.
    > 4.  The Random Forest model will benefit from the large dataset (`keff_interp_data.json` and `ppf_interp_data.json`) already generated, potentially yielding more accurate predictions and a different, sometimes better, outcome.

4.  **Genetic Algorithm Cycle**: The GA evolves a population of fuel loading patterns over thousands of generations, evaluating fitness using the ML predictors for speed.

5.  **Verification**: The best fuel pattern found by the GA is verified with a full, high-fidelity OpenMC simulation.

6.  **Iterative Improvement**: The results from the verification simulation are added back into the dataset, and the ML models are retrained. This makes the predictors more accurate for all subsequent GA cycles.
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
And the lattice of the core might look like the following
```python
pitch= assembly pitch
lattice = openmc.HexLattice(name='Your Reactor Name')
lattice.center = (0., 0.)
lattice.pitch = (pitch,)
lattice.outer = sodium_mod_u

ring_1 = [control_rods_univ]

ring_2 = [fa_inner_univ_1, fa_inner_univ_2, fa_inner_univ_3, fa_inner_univ_4, fa_inner_univ_5, fa_inner_univ_6]

ring_3 = [
    fa_inner_univ_7, fa_inner_univ_8, fa_inner_univ_9, fa_inner_univ_10, fa_inner_univ_11, fa_inner_univ_12,
    fa_inner_univ_13, fa_inner_univ_14, fa_inner_univ_15, fa_inner_univ_16, fa_inner_univ_17, fa_inner_univ_18,
]

ring_4 = [
    control_rods_univ, fa_inner_univ_19, fa_inner_univ_20, control_rods_univ, fa_inner_univ_21, fa_inner_univ_22,
    control_rods_univ, fa_inner_univ_23, fa_inner_univ_24, control_rods_univ, fa_inner_univ_25, fa_inner_univ_26,
    control_rods_univ, fa_inner_univ_27, fa_inner_univ_28, control_rods_univ, fa_inner_univ_29, fa_inner_univ_30,
]

ring_5 = [
    fa_inner_univ_31, fa_inner_univ_32, fa_inner_univ_33, fa_inner_univ_34, fa_inner_univ_35, fa_inner_univ_36,
    fa_inner_univ_37, fa_inner_univ_38, fa_inner_univ_39, fa_inner_univ_40, fa_inner_univ_41, fa_inner_univ_42,
    fa_inner_univ_43, fa_inner_univ_44, fa_inner_univ_45, fa_inner_univ_46, fa_inner_univ_47, fa_inner_univ_48,
    fa_inner_univ_49, fa_inner_univ_50, fa_inner_univ_51, fa_inner_univ_52, fa_inner_univ_53, fa_inner_univ_54,
]

ring_6 = [
    fa_inner_univ_55, fa_inner_univ_56, fa_inner_univ_57, fa_inner_univ_58, fa_inner_univ_59, fa_inner_univ_60,
    fa_inner_univ_61, fa_inner_univ_62, fa_inner_univ_63, fa_inner_univ_64, fa_inner_univ_65, fa_inner_univ_66,
    fa_inner_univ_67, fa_inner_univ_68, fa_inner_univ_69, fa_inner_univ_70, fa_inner_univ_71, fa_inner_univ_72,
    fa_inner_univ_73, fa_inner_univ_74, fa_inner_univ_75, fa_inner_univ_76, fa_inner_univ_77, fa_inner_univ_78,
    fa_inner_univ_79, fa_inner_univ_80, fa_inner_univ_81, fa_inner_univ_82, fa_inner_univ_83, fa_inner_univ_84,
]

ring_7 = [
    control_rods_univ, fa_inner_univ_85, fa_inner_univ_86, control_rods_univ, fa_inner_univ_87, fa_inner_univ_88,
    control_rods_univ, fa_inner_univ_89, fa_inner_univ_90, control_rods_univ, fa_inner_univ_91, fa_inner_univ_92,
    control_rods_univ, fa_inner_univ_93, fa_inner_univ_94, control_rods_univ, fa_inner_univ_95, fa_inner_univ_96,
    control_rods_univ, fa_inner_univ_97, fa_inner_univ_98, control_rods_univ, fa_inner_univ_99, fa_inner_univ_100,
    control_rods_univ, fa_inner_univ_101, fa_inner_univ_102, control_rods_univ, fa_inner_univ_103, fa_inner_univ_104,
    control_rods_univ, fa_inner_univ_105, fa_inner_univ_106, control_rods_univ, fa_inner_univ_107, fa_inner_univ_108,
]

ring_8 = [
    fa_inner_univ_109, fa_inner_univ_110, fa_inner_univ_111, fa_inner_univ_112, fa_inner_univ_113, fa_inner_univ_114,
    fa_inner_univ_115, fa_inner_univ_116, fa_inner_univ_117, fa_inner_univ_118, fa_inner_univ_119, fa_inner_univ_120,
    fa_inner_univ_121, fa_inner_univ_122, fa_inner_univ_123, fa_inner_univ_124, fa_inner_univ_125, fa_inner_univ_126,
    fa_inner_univ_127, fa_inner_univ_128, fa_inner_univ_129, fa_inner_univ_130, fa_inner_univ_131, fa_inner_univ_132,
    fa_inner_univ_133, fa_inner_univ_134, fa_inner_univ_135, fa_inner_univ_136, fa_inner_univ_137, fa_inner_univ_138,
    fa_inner_univ_139, fa_inner_univ_140, fa_inner_univ_141, fa_inner_univ_142, fa_inner_univ_143, fa_inner_univ_144,
    fa_inner_univ_145, fa_inner_univ_146, fa_inner_univ_147, fa_inner_univ_148, fa_inner_univ_149, fa_inner_univ_150,
]
ring_9     =  ([radial_reflector_univ]*2 + [fa_outer_univ]*5 + [radial_reflector_univ]*1) * 6
ring_10    =  [radial_reflector_univ]*54
ring_11    =  ([radial_shield_univ]*2 + [radial_reflector_univ]*7 + [radial_shield_univ]*1)*6
ring_12    =  ([sodium_mod_u]*2 + [radial_shield_univ]*8 + [sodium_mod_u])*6

lattice.universes = [ring_12, ring_11, ring_10, ring_9, ring_8, ring_7, ring_6, ring_5, ring_4, ring_3, ring_2, ring_1]
lattice.orientation = 'x'

# Create the prism that will contain the lattice
outer_core_surface = openmc.model.hexagonal_prism(edge_length=196.9856187, boundary_type='vacuum',orientation='x')

# Fill a cell with the lattice. This cell is filled with the lattice and contained within the prism.
core = openmc.Cell( fill=lattice, region=outer_core_surface & -top & +bottom)

# Create a universe that contains both 
main_u = openmc.Universe( cells=[core]) 
geom = openmc.Geometry(main_u)
geom.export_to_xml()
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

A Python script is utilized to process the simulation output and calculate the Power Peaking Factor (PPF), which is the ratio of the maximum power produced in a single fuel assembly to the average power across all the fuel assemblies. This script also exports the normalized power for each fuel cell, which is essential for identifying high-power regions.

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

### Tuning the `k-eff` and PPF Balance

In NOMAD, achieving a precise `target_keff` while minimizing the power peaking factor (PPF) is a balancing act. The genetic algorithm's fitness function is designed to dynamically shift its priorities between these two competing objectives. Two key parameters in `config.ini` control this behavior:

* `med_keff_diff_threshold`: A threshold that tells the GA when to shift from a PPF-focused strategy to a balanced one.
* `high_keff_diff_threshold`: A higher threshold that tells the GA to *aggressively* prioritize fixing `k-eff` above optimizing PPF.

The fitness score for `k-eff` is now calculated on a continuous curve, meaning **any** improvement that brings `k-eff` closer to the target will result in a better score. The thresholds do not change the score itself, but they are critical for controlling the **weights** in the fitness function, telling the algorithm what to focus on.

Let's explore two scenarios to understand why coordinating these thresholds is important.

***

#### Scenario 1: Default Configuration

With the default settings, the GA has a relatively relaxed approach, prioritizing PPF improvement until `k-eff` deviates significantly.

**Default Settings:**

* `med_keff_diff_threshold = 0.02`
* `high_keff_diff_threshold = 0.05`

**Behavior:**

1.  **PPF Priority Zone (`k-eff` diff < 0.02):** If a design's `k-eff` is within 2000 pcm of the target, the GA considers the `k-eff` to be in a good range. The fitness function heavily prioritizes minimizing the PPF (weighting: 70% PPF, 30% `k-eff`).
2.  **Balanced Zone (`k-eff` diff between 0.02 and 0.05):** Once the `k-eff` difference exceeds the `med_keff_diff_threshold` (2000 pcm), the GA enters a balanced state, weighting PPF and `k-eff` equally (50%/50%).
3.  **`k-eff` Priority Zone (`k-eff` diff > 0.05):** Only when the `k-eff` is off by more than 5000 pcm (`high_keff_diff_threshold`) does the GA become aggressive, heavily prioritizing `k-eff` improvement (weighting: 30% PPF, 70% `k-eff`).

**Conclusion:** The default settings are suitable for initial exploration but are not sensitive enough for achieving high-precision results, as the GA will tolerate a large `k-eff` deviation before shifting its focus.

***

#### Scenario 2: High-Precision Configuration

Now, let's see what happens when we make the thresholds much more sensitive.

**High-Precision Settings:**

* `med_keff_diff_threshold = 0.005`
* `high_keff_diff_threshold = 0.01`

**Behavior:**

1.  **PPF Priority Zone (`k-eff` diff < 0.005):** The window for prioritizing PPF is now much smaller. The GA will focus on PPF only if the `k-eff` is within 500 pcm of the target.
2.  **Balanced Zone (`k-eff` diff between 0.005 and 0.01):** As soon as the `k-eff` difference exceeds 500 pcm (`med_keff_diff_threshold`), the GA *immediately* shifts to a balanced 50%/50% focus. It doesn't wait for the error to become large.
3.  **`k-eff` Priority Zone (`k-eff` diff > 0.01):** When the `k-eff` is off by more than 1000 pcm (`high_keff_diff_threshold`), the GA dedicates most of its effort (70% weight) to correcting `k-eff`.

**Conclusion:** By tightening the thresholds, the GA becomes highly sensitive and responsive. It actively works to correct even small deviations from the `target_keff`, making it far more likely to find a solution that satisfies your precise requirements.

### Recommendation

When you require a highly accurate `k-eff`, you should set lower values for **`med_keff_diff_threshold`** and **`high_keff_diff_threshold`**.

By tightening these thresholds, the genetic algorithm becomes more sensitive to deviations from the `target_keff` and will prioritize correcting them much more quickly. This ensures the GA's focus is aligned with your high-precision optimization goals from the very beginning.

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
## Results
This section presents the results from an initial case study applying the NOMAD framework to a sodium-fast reactor model https://doi.org/10.1016/j.anucene.2024.111019 (specifically the axial model).

# Case Study Configuration
This preliminary study utilized an earlier, monolithic version of the NOMAD code. The key aspects of this configuration include:

Interpolation Method: K-Nearest Neighbors (KNN) was used as the surrogate model for predicting both k-effective and the Power Peaking Factor (PPF).

Initialization: The optimization process was initialized using a pre-defined set of 40 specific enrichment configurations for the central and outer core regions, rather than randomly generated samples.

# Optimization Process Performance
The performance of the genetic algorithm and the accuracy of the surrogate models were tracked throughout the optimization. The following figures illustrate the evolution of the fitness score and the error of the KNN interpolators over the OpenMC cycles.

Figure 1: Evolution of the fitness score over the optimization cycles.
![Image Alt](https://github.com/XxNILOYxX/nomad/blob/main/images/fitness_score.png?raw=true)
Figure 2: The percentage error of the k-effective KNN interpolator, indicating the model's predictive accuracy from cycle to cycle.
![Image Alt](https://github.com/XxNILOYxX/nomad/blob/main/images/keff_error_percent.png?raw=true)
Figure 3: The percentage error of the Power Peaking Factor (PPF) KNN interpolator throughout the run.
![Image Alt](https://github.com/XxNILOYxX/nomad/blob/main/images/ppf_error_percent.png?raw=true)

# Core Power Profile Optimization
A primary objective of fuel loading pattern optimization is to flatten the power distribution, which enhances thermal performance and fuel utilization. The figures below compare the power profile of a reference core with uniform (homogeneous) enrichment against the profile achieved using the heterogeneous loading pattern developed by NOMAD. In both visualizations, the relative power for each assembly is defined as the ratio of its specific power output to the average assembly power across the entire core.

Figure 4: Power distribution in a reference core with a standard homogeneous fuel loading, showing a significant power peak in the center.
![Image Alt](https://github.com/XxNILOYxX/nomad/blob/main/images/Without%20NOMAD.png?raw=true)

Figure 5: Optimized power distribution achieved with the heterogeneous fuel loading pattern from NOMAD. The profile is visibly flatter, with a reduced central peak and more uniform power across the assemblies.
![Image Alt](https://github.com/XxNILOYxX/nomad/blob/main/images/With%20NOMAD.png?raw=true)

As demonstrated, the NOMAD framework successfully identified a fuel loading pattern that significantly flattens the core power profile, achieving a key goal in advanced reactor design.

---
## DPSO
## Advanced Optimization Engines: PSO and Hybrid

NOMAD now includes a **Particle Swarm Optimizer (PSO)** for efficient local searches and a **Hybrid Engine** that combines the global exploration of the GA with the local exploitation of the PSO.

To use these advanced engines, set the `technique` parameter in your `config.ini` file:

```ini
[optimizer]
# The optimization technique to use. Options: ga, pso, hybrid
technique = hybrid
```

### Particle Swarm Optimization (PSO) Configuration

The PSO is a powerful optimizer that simulates a "swarm" of particles searching for the best solution. Its behavior is controlled by the [pso] section in config.ini.

```ini
[pso]
# Number of particles in the swarm. Analogous to GA's population_size. (Default: 1500)
swarm_size = 1500

# Number of iterations the PSO will run using ML predictors before an OpenMC verification. (Default: 1000)
iterations_per_openmc_cycle = 1000

# Cognitive coefficient (c1), controls the particle's attraction to its personal best. (Default: 2.0)
cognitive_coeff = 2.0

# Social coefficient (c2), controls the particle's attraction to the global/neighborhood best. (Default: 2.0)
social_coeff = 2.0

# Number of iterations without improvement before the PSO cycle exits early.
pso_convergence_threshold = 800

# Starting and ending inertia weights for linear decay.
inertia_weight_start = 0.95
inertia_weight_end = 0.35
```

### Key PSO Parameter Effects:

- **swarm_size**: A larger swarm explores more of the search space but requires more computational resources per iteration.
- **cognitive_coeff (c1)**: Increasing this value makes particles more independent, encouraging them to explore around their own best-found positions. Too high a value can lead to premature convergence on many different local optima.
- **social_coeff (c2)**: Increasing this value makes particles more influenced by the swarm's best-found position, promoting faster convergence. Too high a value can cause the entire swarm to get stuck in a single local optimum.
- **inertia_weight_start / inertia_weight_end**: The inertia weight controls the particle's momentum. It starts high (0.95) to encourage global exploration at the beginning of a run and decreases (0.35) to promote fine-tuning and local exploitation as the run progresses.

---

### Advanced PSO Features

NOMAD's PSO includes several advanced features for enhanced performance, configured under the [pso] section.

```ini
# Topology defines how particles are connected. Options: global, ring, random, fitness_based
topology = ring

# For 'ring', 'random', or 'fitness_based' topologies, the number of neighbors for each particle. (e.g., 2, 4)
neighborhood_size = 4

# Frequency (in iterations) to rebuild neighborhoods for 'random' and 'fitness_based' topologies.
neighborhood_rebuild_frequency = 100

# Set to true to enable adaptive velocity clamping, false to use a fixed max_change_probability.
adaptive_velocity = true

# Base and max probability of change, used for adaptive velocity clamping.
base_change_probability = 0.25
max_change_probability = 0.90

# Multi-swarm parameters
enable_multi_swarm = true
num_sub_swarms = 4
migration_frequency = 200
migration_rate = 0.05

# Moderated Local Search
enable_local_search = true
# How often (in iterations) to attempt local search.
local_search_frequency = 50
```

### Advanced Feature Details:

- **topology**: Defines the communication structure of the swarm.
  - **global**: All particles are influenced by the single best particle in the entire swarm. Converges fast but can get stuck.
  - **ring**: Each particle is only influenced by its immediate neighbors in a ring structure. Slower convergence but less prone to getting stuck.
  - **random**: Each particle has a random set of neighbors, which are periodically rebuilt.
- **adaptive_velocity**: If true, the maximum velocity (chance of a particle changing a value) decreases over time, shifting the search from exploration to exploitation.
- **enable_multi_swarm**: If true, the main swarm is split into smaller sub-swarms. This is highly recommended for complex problems. Some sub-swarms are configured to explore aggressively, while others focus on exploitation. They periodically exchange their best particles (migration_rate) to share information, which is a very effective strategy for avoiding local optima.
- **enable_local_search**: If true, the algorithm will periodically apply a simple hill-climbing search to the current best solution to see if small, incremental changes can improve it further.

---

### Hybrid (GA-PSO) Engine Configuration

The hybrid engine dynamically switches between the GA and PSO to leverage the strengths of both. Its strategy is set in the [hybrid] section.

```ini
[hybrid]
# Defines the strategy for switching between GA and PSO.
# Options: fixed_cycles, stagnation, oscillate, adaptive
switch_mode = oscillate

# Parameters for 'fixed_cycles' and 'oscillate' modes
ga_phase_cycles = 20
pso_phase_cycles = 10

# Parameters for 'stagnation' mode
# Number of cycles with no fitness improvement to be considered stagnation.
stagnation_threshold = 10
# Diversity threshold below which the GA is considered to have converged.
ga_min_diversity_for_switch = 0.25

# Parameters for PSO -> GA Seeding
# The fraction of the GA population to be seeded with the best individuals from PSO.
ga_seed_ratio = 0.25

# Parameters for 'adaptive' mode
# Switch to the other algorithm if its average fitness gain is this much better (e.g., 1.2 = 20% better).
adaptive_switching_threshold = 1.2
# The minimum number of cycles a phase must run before an adaptive switch can occur.
min_adaptive_phase_duration = 5
# Factor for comparing negative trends in adaptive mode.
adaptive_trend_dampening_factor = 0.5
```

---

### Hybrid Switching Modes Explained:

- **switch_mode**: This is the core setting for the hybrid engine.
  - **fixed_cycles**: A simple approach. It runs the GA for `ga_phase_cycles` OpenMC cycles, then switches to the PSO indefinitely.
  - **stagnation**: Runs the GA until its fitness improvement stagnates and its population diversity drops below `ga_min_diversity_for_switch`. Then it switches to the PSO.
  - **oscillate**: The recommended balanced strategy. It runs the GA for `ga_phase_cycles` cycles, then switches to the PSO. The PSO runs for `pso_phase_cycles` cycles, and then it switches back to the GA. This continues, oscillating between the two engines. A switch can also be triggered early if an engine's fitness stagnates for `stagnation_threshold` cycles.
  - **adaptive**: The most complex mode. It tracks the performance (fitness gain) of both algorithms. It will switch to the other algorithm if its historical performance is significantly better (defined by `adaptive_switching_threshold`) and the performance trend looks promising.

- **ga_seed_ratio**: When switching from PSO to GA, this determines what fraction of the new GA population is created from the best particles found by the PSO. A higher value (e.g., 0.4) means more knowledge is transferred from the PSO, but it also reduces the initial diversity of the new GA population.

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

This project is licensed under the **GPL-3.0 license**.

See the [LICENSE](LICENSE) file for more details.


import xml.etree.ElementTree as ET
import logging
import numpy as np
from typing import List, Dict

class FuelHandler:
    """
    Manages programmatic updates to the OpenMC materials.xml file.
    It reads the initial composition from the materials.xml file and only
    modifies the nuclides specified in the fuel setup configuration.
    """
    def __init__(self, config: Dict):
        """
        Initializes the FuelHandler. It reads and stores the initial state
        of the fuel from the materials.xml file.

        Args:
            config: A dictionary containing simulation and fuel parameters.
        """
        self.sim_config = config['simulation']
        self.fuel_config = config['fuel']
        
        self.materials_xml_path = self.sim_config['materials_xml_path']
        self.num_assemblies = self.sim_config['num_assemblies']
        
        # Identify which isotopes are being varied by the GA
        self.fissile_isotopes = {iso for iso, flag in self.fuel_config['fissile_flags'].items() if flag}
        self.slack_isotope = self.fuel_config['slack_isotope']
        
        logging.info(f"GA will optimize enrichment for: {list(self.fissile_isotopes)}")
        logging.info(f"Slack isotope for balancing weights: {self.slack_isotope}")

        # This will store the initial composition of each fuel material
        self.initial_compositions = {}
        self._load_and_validate_initial_composition()

    def _load_and_validate_initial_composition(self):
        """
        Loads the initial fuel composition from the materials.xml file and
        validates that the weights for each fuel material sum to 1.0.
        """
        logging.info(f"Reading initial fuel compositions from '{self.materials_xml_path}'...")
        try:
            tree = ET.parse(self.materials_xml_path)
            root = tree.getroot()
        except (ET.ParseError, FileNotFoundError) as e:
            raise ValueError(f"Error reading or parsing XML file '{self.materials_xml_path}': {e}")

        start_id = self.sim_config['start_id']
        for i in range(self.num_assemblies):
            material_id = str(start_id + i)
            material_node = root.find(f".//material[@id='{material_id}']")
            
            if material_node is None:
                raise ValueError(f"Material with id='{material_id}' not found in XML. Cannot initialize.")

            composition = {}
            total_weight = 0.0
            for nuclide in material_node.findall('nuclide'):
                name = nuclide.get('name')
                weight = float(nuclide.get('wo'))
                composition[name] = weight
                total_weight += weight
            
            if not np.isclose(total_weight, 1.0):
                raise ValueError(
                    f"Initial weights for material id='{material_id}' do not sum to 1.0. "
                    f"Current sum: {total_weight:.6f}. Please correct your materials.xml file."
                )
            
            if self.slack_isotope not in composition:
                raise ValueError(f"Slack isotope '{self.slack_isotope}' not found in material id='{material_id}'.")

            self.initial_compositions[material_id] = composition
        logging.info("Initial fuel compositions loaded and validated successfully.")

    def update_materials(self, enrichments: List[float]) -> bool:
        """
        Updates the materials.xml file with new enrichment values from the GA.

        This function preserves all non-optimized nuclide weights from the initial
        state, applies the new total fissile enrichment from the GA, and adjusts
        the U238 'slack' nuclide to ensure the total weight remains 1.0.

        Args:
            enrichments: A list of total fissile weight percentages for each assembly.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            tree = ET.parse(self.materials_xml_path)
            root = tree.getroot()
        except (ET.ParseError, FileNotFoundError) as e:
            logging.error(f"Error reading or parsing XML file '{self.materials_xml_path}': {e}")
            return False

        if len(enrichments) != self.num_assemblies:
            logging.error(f"Configuration error: Expected {self.num_assemblies} enrichments, received {len(enrichments)}.")
            return False

        start_id = self.sim_config['start_id']
        for i in range(self.num_assemblies):
            material_id = str(start_id + i)
            ga_total_fissile_wo = enrichments[i] / 100.0  # Convert percentage to weight fraction
            
            target_material_node = root.find(f".//material[@id='{material_id}']")
            if target_material_node is None:
                logging.warning(f"Material with id='{material_id}' not found in XML. Skipping.")
                continue

            initial_comp = self.initial_compositions[material_id]
            
            # Calculate the sum of all weights that are NOT part of the GA optimization
            # and are NOT the slack material. These are preserved.
            preserved_weight_sum = 0.0
            for nuclide, weight in initial_comp.items():
                if nuclide not in self.fissile_isotopes and nuclide != self.slack_isotope:
                    preserved_weight_sum += weight

            # Calculate new weights for the fissile isotopes targeted by the GA
            new_fissile_weights = {}
            if 'Pu239' in self.fissile_isotopes: # Assume Pu distribution if any Pu isotope is a target
                for isotope in self.fissile_isotopes:
                    new_fissile_weights[isotope] = self.fuel_config['pu_dist'][isotope] * ga_total_fissile_wo
            elif self.fissile_isotopes: # Handles cases like only U235 or U233
                isotope = list(self.fissile_isotopes)[0] # Assumes only one if not Pu
                new_fissile_weights[isotope] = ga_total_fissile_wo

            # The new weight for U238 is what's left over
            new_slack_weight = 1.0 - preserved_weight_sum - ga_total_fissile_wo
            
            if new_slack_weight < 0:
                logging.error(
                    f"Invalid material composition for material {material_id}. Sum of weights > 1.0. "
                    f"{self.slack_isotope} would be {new_slack_weight:.6f}. Check enrichment levels."
                )
                return False

            # Update all nuclide weights in the XML tree
            for nuclide_node in target_material_node.findall('nuclide'):
                name = nuclide_node.get('name')
                new_weight = 0.0
                if name == self.slack_isotope:
                    new_weight = new_slack_weight
                elif name in self.fissile_isotopes:
                    new_weight = new_fissile_weights[name]
                else:
                    new_weight = initial_comp[name] # Keep the initial weight
                
                nuclide_node.set('wo', f"{new_weight:.10f}")

        try:
            tree.write(self.materials_xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"Successfully updated '{self.materials_xml_path}'.")
            return True
        except IOError as e:
            logging.error(f"Failed to write to '{self.materials_xml_path}': {e}")
            return False
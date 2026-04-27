import xml.etree.ElementTree as ET
import random
import numpy as np
from typing import List, Tuple, Dict, Any
import json
import os

class StochasticTrafficGenerator:
    """
    Generates random traffic flows for PPO training with various distribution patterns
    """
    
    def __init__(self, reference_routes_file: str, seed: int = None):
        """Extract valid OD pairs and vehicle types from reference file"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Check if file exists
        if not os.path.exists(reference_routes_file):
            raise FileNotFoundError(f"Reference file not found: {reference_routes_file}")
        
        tree = ET.parse(reference_routes_file)
        root = tree.getroot()
        
        # Extract valid pairs and types
        self.valid_pairs = []
        self.vehicle_types = set()
        
        for flow in root.findall('flow'):
            from_edge = flow.get('from')
            to_edge = flow.get('to')
            veh_type = flow.get('type')
            
            if from_edge and to_edge:
                self.valid_pairs.append((from_edge, to_edge))
            if veh_type:
                self.vehicle_types.add(veh_type)
        
        # Remove duplicates while preserving order
        self.valid_pairs = list(dict.fromkeys(self.valid_pairs))
        self.vehicle_types = list(self.vehicle_types)
        
        print(f"Loaded {len(self.valid_pairs)} valid OD pairs")
        print(f"Vehicle types: {self.vehicle_types}")
        
    def generate_flow_with_distribution(self, 
                                        distribution: str = 'uniform',
                                        **params) -> Dict[str, Any]:
        """
        Generate a single flow using specified distribution
        """
        
        # Randomly select OD pair and vehicle type
        from_edge, to_edge = random.choice(self.valid_pairs)
        veh_type = random.choice(self.vehicle_types)
        
        # Generate begin time based on distribution
        sim_duration = params.get('sim_duration', 5000)
        
        if distribution == 'uniform':
            begin = np.random.uniform(0, sim_duration * 0.8)
            duration = np.random.uniform(300, sim_duration - begin)
            end = min(begin + duration, sim_duration)
            period = np.random.uniform(2, 60)
            
        elif distribution == 'normal':
            mean = params.get('mean', sim_duration / 2)
            std = params.get('std', sim_duration / 6)
            begin = np.random.normal(mean, std)
            begin = np.clip(begin, 0, sim_duration * 0.9)
            
            duration_mean = params.get('duration_mean', 1000)
            duration_std = params.get('duration_std', 300)
            duration = np.random.normal(duration_mean, duration_std)
            end = min(begin + abs(duration), sim_duration)
            
            period_mean = params.get('period_mean', 15)
            period_std = params.get('period_std', 5)
            period = abs(np.random.normal(period_mean, period_std))
            period = np.clip(period, 1, 60)
            
        elif distribution == 'poisson':
            begin = np.random.uniform(0, sim_duration * 0.7)
            duration = np.random.uniform(500, 2000)
            end = min(begin + duration, sim_duration)
            lambda_rate = np.random.exponential(scale=0.05)
            period = 1.0 / max(lambda_rate, 0.01)
            
        elif distribution == 'exponential':
            begin = np.random.exponential(scale=sim_duration / 4)
            begin = min(begin, sim_duration * 0.8)
            duration = np.random.exponential(scale=800)
            end = min(begin + duration, sim_duration)
            period = np.random.exponential(scale=10)
            period = np.clip(period, 1, 60)
            
        elif distribution == 'lognormal':
            begin = np.random.lognormal(mean=np.log(sim_duration/3), sigma=0.8)
            begin = min(begin, sim_duration * 0.9)
            duration = np.random.lognormal(mean=np.log(800), sigma=0.5)
            end = min(begin + duration, sim_duration)
            period = np.random.lognormal(mean=np.log(15), sigma=0.7)
            period = np.clip(period, 1, 60)
            
        elif distribution == 'gamma':
            begin = np.random.gamma(shape=2, scale=sim_duration/6)
            begin = min(begin, sim_duration * 0.85)
            duration = np.random.gamma(shape=2, scale=400)
            end = min(begin + duration, sim_duration)
            period = np.random.gamma(shape=2, scale=8)
            period = np.clip(period, 1, 50)
            
        elif distribution == 'bimodal':
            peak_choice = np.random.choice([0, 1], p=[0.5, 0.5])
            if peak_choice == 0:
                begin = np.random.normal(750, 100)
            else:
                begin = np.random.normal(1750, 100)
            begin = np.clip(begin, 0, sim_duration * 0.85)
            duration = np.random.uniform(300, 800)
            end = min(begin + duration, sim_duration)
            period = np.random.gamma(shape=1.5, scale=3)
            period = np.clip(period, 2, 15)
            
        elif distribution == 'pareto':
            begin = (np.random.pareto(2) + 1) * (sim_duration / 10)
            begin = min(begin, sim_duration * 0.85)
            duration = (np.random.pareto(1.5) + 1) * 300
            end = min(begin + duration, sim_duration)
            period = (np.random.pareto(2) + 1) * 8
            period = np.clip(period, 1, 80)
            
        elif distribution == 'uniform_periodic':
            begin = np.random.uniform(0, sim_duration * 0.8)
            duration = np.random.uniform(500, 1500)
            end = min(begin + duration, sim_duration)
            t = begin / sim_duration
            base_period = 20
            amplitude = 15
            period = base_period + amplitude * np.sin(2 * np.pi * t * 3)
            period += np.random.normal(0, 2)
            period = np.clip(period, 3, 50)
            
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Ensure all values are valid
        begin = max(0, float(begin))
        end = max(begin + 1, float(end))
        period = max(0.1, float(period))
        
        return {
            'from': from_edge,
            'to': to_edge,
            'begin': begin,
            'end': end,
            'period': period,
            'type': veh_type,
            'distribution': distribution
        }
    
    def generate_training_episode(self, 
                                  num_flows: int = 30,
                                  mixed_distributions: bool = True,
                                  distribution_weights: Dict[str, float] = None) -> Tuple[List[ET.Element], List[Dict]]:
        """
        Generate a complete training episode with mixed randomness
        """
        
        if distribution_weights is None:
            distribution_weights = {
                'uniform': 0.15,
                'normal': 0.20,
                'poisson': 0.15,
                'exponential': 0.10,
                'lognormal': 0.10,
                'gamma': 0.10,
                'bimodal': 0.15,
                'pareto': 0.05
            }
        
        distributions = list(distribution_weights.keys())
        weights = list(distribution_weights.values())
        
        flows = []
        flow_metadata = []
        
        for i in range(num_flows):
            if mixed_distributions:
                dist_type = np.random.choice(distributions, p=weights)
            else:
                dist_type = 'normal'
            
            flow_params = self.generate_flow_with_distribution(dist_type)
            
            # Create XML element
            flow_elem = ET.Element('flow')
            flow_elem.set('id', f'flow_{i}_{dist_type}')
            flow_elem.set('from', flow_params['from'])
            flow_elem.set('to', flow_params['to'])
            flow_elem.set('begin', f"{flow_params['begin']:.1f}")
            flow_elem.set('end', f"{flow_params['end']:.1f}")
            flow_elem.set('period', f"{flow_params['period']:.2f}")
            flow_elem.set('type', flow_params['type'])
            
            flows.append(flow_elem)
            flow_metadata.append(flow_params)
        
        return flows, flow_metadata
    
    def save_episode(self, 
                    flows: List[ET.Element], 
                    filename: str,
                    episode_metadata: Dict = None):
        """
        Save flows to SUMO route file - FIXED VERSION that works with SUMO
        """
        # Manual XML construction (most reliable for SUMO)
        with open(filename, 'w', encoding='utf-8') as f:
            # Write XML declaration - MUST be first line with no leading spaces
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            
            # Write root element with namespace
            f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
            f.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
            
            # Write each flow as a properly formatted line
            for flow in flows:
                # Collect attributes
                attrs = []
                for key, value in flow.attrib.items():
                    attrs.append(f'{key}="{value}"')
                
                # Create the flow line with proper indentation
                flow_line = '    <flow ' + ' '.join(attrs) + '/>\n'
                f.write(flow_line)
            
            # Write closing tag
            f.write('</routes>\n')
        
        # Verify the file was written correctly
        try:
            with open(filename, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('<?xml'):
                    raise ValueError("XML declaration missing or malformed")
            
            print(f"✓ Saved {len(flows)} flows to {filename}")
            
            # Save metadata if provided
            if episode_metadata:
                meta_filename = filename.replace('.rou.xml', '_metadata.json')
                with open(meta_filename, 'w') as f:
                    json.dump(episode_metadata, f, indent=2)
                    
        except Exception as e:
            print(f"✗ Error saving file: {e}")
            raise

# Usage example with proper file handling
if __name__ == "__main__":
    # Make sure to use your actual reference file name
    reference_file = 'your_routes_file.xml'  # CHANGE THIS TO YOUR FILE
    
    try:
        # Initialize generator
        generator = StochasticTrafficGenerator(reference_file, seed=42)
        
        # Generate a single test episode first
        print("\n--- Generating test episode ---")
        flows, metadata = generator.generate_training_episode(
            num_flows=10,  # Small number for testing
            mixed_distributions=True
        )
        
        # Save test episode
        generator.save_episode(flows, 'test_routes.rou.xml', {
            'test': True,
            'num_flows': len(flows)
        })
        
        # Verify the file with SUMO's XML parser
        print("\n--- Verifying XML file ---")
        try:
            verify_tree = ET.parse('test_routes.rou.xml')
            print("✓ XML file is valid!")
            print(f"  Root tag: {verify_tree.getroot().tag}")
            print(f"  Number of flows: {len(verify_tree.getroot().findall('flow'))}")
        except Exception as e:
            print(f"✗ XML validation failed: {e}")
        
        # Now generate full training episodes
        print("\n--- Generating training episodes ---")
        for episode in range(10):  # Generate 10 episodes for testing
            flows, metadata = generator.generate_training_episode(
                num_flows=30,
                mixed_distributions=True
            )
            
            generator.save_episode(flows, f'training_episode_{episode:04d}.rou.xml', {
                'episode': episode,
                'num_flows': len(flows),
                'distributions': [f['distribution'] for f in metadata]
            })
            
            if episode % 5 == 0:
                print(f"  Generated episode {episode}")
        
        print("\n✓ All files generated successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update 'reference_file' to point to your actual routes file")
    except Exception as e:
        print(f"Unexpected error: {e}")
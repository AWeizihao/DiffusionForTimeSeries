# diffusion_timeseries/cli.py

import argparse
import yaml
from .trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train diffusion model for financial risk prediction.")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = train_model(config)
    print("Training complete. Model is ready!")
    
if __name__ == '__main__':
    main()
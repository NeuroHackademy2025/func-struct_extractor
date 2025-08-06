import os
import json
import yaml

def load_config(path):
    path = str(path)
    if path.endswith('.yaml') or path.endswith('.yml'):
        cfg = yaml.safe_load(open(path))
    elif path.endswith('.json'):
        cfg = json.load(open(path))
    else:
        raise ValueError("Config must be .yaml or .json")
    validate_config(cfg)
    return cfg

def validate_config(cfg):
    if 'subject' not in cfg or not isinstance(cfg['subject'], str):
        raise ValueError("Missing or invalid 'subject'")
    if 'output_dir' not in cfg:
        raise ValueError("Missing 'output_dir'")
    if 'scalars' in cfg:
        for item in cfg['scalars']:
            if not all(k in item for k in ('path','name')):
                raise ValueError("Each scalar entry needs 'path' and 'name'")
    # For extractor config:
    if 'tract' in cfg and not os.path.exists(cfg['tract']):
        raise FileNotFoundError(f"Tract file not found: {cfg['tract']}")
    return True


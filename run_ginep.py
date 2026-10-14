"""Print the dedicated GIN-EP configuration without loading any weights."""
import argparse
import json
from ginep import GINEPConfig, HEAD_TYPES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=HEAD_TYPES, default="mlp")
    args = parser.parse_args()
    cfg = GINEPConfig()
    cfg.head_type = args.head
    print(json.dumps({name: getattr(cfg, name) for name in dir(cfg)
                      if not name.startswith("_") and not callable(getattr(cfg, name))},
                     indent=2, ensure_ascii=False))

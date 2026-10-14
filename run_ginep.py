"""Print the dedicated GIN-EP configuration without loading any weights."""
import json
from ginep import GINEPConfig

if __name__ == "__main__":
    cfg = GINEPConfig()
    print(json.dumps({name: getattr(cfg, name) for name in dir(cfg)
                      if not name.startswith("_") and not callable(getattr(cfg, name))},
                     indent=2, ensure_ascii=False))

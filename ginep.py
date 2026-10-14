"""Dedicated configuration and model factory for the original GIN-EP arm."""
from config import Config


class GINEPConfig(Config):
    experiment_name = "HAGDTI_GINEP"
    use_esm_protein = True
    use_bermol_drug = True
    use_gpcr_prior = True
    use_pharmacophore = True
    use_cnn_for_cross_attn = False
    use_graph_for_cross_attn = True
    use_ligand_graph = True
    ligand_encoder_type = "default"
    graph_deep_type = "gin_ep"
    graph_use_edge_attr = True
    graph_use_rw_pe = True
    graph_rw_pe_walk = 20
    graph_rw_pe_dim = 8
    graph_gin_dim = 32
    graph_xattn_gated = False
    fusion_mode = "flat"
    protein_token_source = "tm7"
    head_type = "mlp"


def build_model(cfg=None):
    """Build the complete model; loading pretrained weights is explicit here."""
    from model import ClassAGPCRDTIModel

    cfg = GINEPConfig() if cfg is None else cfg
    for name in (
        "use_esm_protein", "use_bermol_drug", "use_gpcr_prior",
        "use_pharmacophore", "use_cnn_for_cross_attn", "use_graph_for_cross_attn",
        "use_ligand_graph", "ligand_encoder_type", "graph_deep_type",
        "graph_use_edge_attr", "graph_use_rw_pe", "graph_rw_pe_walk",
        "graph_rw_pe_dim", "graph_gin_dim", "graph_xattn_gated",
        "fusion_mode", "protein_token_source", "head_type",
    ):
        if getattr(cfg, name, None) != getattr(GINEPConfig, name):
            raise ValueError(f"GIN-EP architecture requires {name}={getattr(GINEPConfig, name)!r}")
    return ClassAGPCRDTIModel(cfg)

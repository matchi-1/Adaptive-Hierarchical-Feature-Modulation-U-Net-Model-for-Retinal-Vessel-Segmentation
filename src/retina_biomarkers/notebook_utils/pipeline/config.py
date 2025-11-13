from dataclasses import dataclass

@dataclass
class PipelineConfig:
    image_size: int = 512
    threshold: float = 0.5
    use_fov_in_model: bool = True

    # OD params
    r_frac = (0.08, 0.16)       # PD radius clamp (fraction of min(H,W))
    cup_dilate_frac: float = 0.12

    # preprocess (for vessel model)
    use_gamma: bool = True
    gamma: float = 0.9
    clahe_clip: float = 2.0
    clahe_tiles: int = 8

    # biomarkers
    ortho_step: float = 0.5
    ortho_max_radius: float = 20.0
    max_gap_px: int = 12
    angle_k_ahead: int = 3

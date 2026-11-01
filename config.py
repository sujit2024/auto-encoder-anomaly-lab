from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_path: Path = field(init=False)
    model_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    scaler_path: Path = field(init=False)
    metrics_path: Path = field(init=False)

    label_column: str = "label"
    feature_columns: Optional[List[str]] = None

    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42

    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 80
    latent_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    early_stopping_patience: int = 10

    threshold_std_factor: float = 3.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.data_path = self.project_root / "data" / "sample.csv"
        self.model_dir = self.project_root / "artifacts"
        self.model_path = self.model_dir / "autoencoder.pt"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.metrics_path = self.model_dir / "metrics.json"


DEFAULT_CONFIG = Config()

DATA_PATH = DEFAULT_CONFIG.data_path
TEST_SIZE = DEFAULT_CONFIG.test_size
RANDOM_STATE = DEFAULT_CONFIG.random_state
BATCH_SIZE = DEFAULT_CONFIG.batch_size
EPOCHS = DEFAULT_CONFIG.num_epochs
THRESHOLD_STD_FACTOR = DEFAULT_CONFIG.threshold_std_factor
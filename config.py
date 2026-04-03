from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


class RiskConfig(BaseModel):
    max_position_size: float = Field(default=500.0)
    max_inventory_imbalance: float = Field(default=0.3)
    max_daily_loss: float = Field(default=200.0)
    spread_min_cents: float = Field(default=3.0)
    order_size_base: float = Field(default=25.0)


class Config(BaseModel):
    # Polymarket
    api_key: str = Field(default="")
    secret: str = Field(default="")
    passphrase: str = Field(default="")
    private_key: str = Field(default="")
    chain_id: int = Field(default=137)

    # LLM
    anthropic_api_key: str = Field(default="")

    # Risk
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # General
    log_level: str = Field(default="INFO")
    dry_run: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_key=os.getenv("POLYMARKET_API_KEY", ""),
            secret=os.getenv("POLYMARKET_SECRET", ""),
            passphrase=os.getenv("POLYMARKET_PASSPHRASE", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            chain_id=int(os.getenv("CHAIN_ID", "137")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            risk=RiskConfig(
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "500")),
                max_inventory_imbalance=float(os.getenv("MAX_INVENTORY_IMBALANCE", "0.3")),
                max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "200")),
                spread_min_cents=float(os.getenv("SPREAD_MIN_CENTS", "3")),
                order_size_base=float(os.getenv("ORDER_SIZE_BASE", "25")),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
        )


config = Config.from_env()

"""Central configuration for payTrace."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
STREAM_DB = ROOT / "stream.db"

RAW_TXNS = DATA_DIR / "transactions.csv"
FEATURES = DATA_DIR / "features.parquet"

# Time-based split (day index from the start of the data). Never split randomly:
# fraud data is temporal and random splits leak future behaviour into training.
TRAIN_END_DAY = 36
VAL_END_DAY = 45          # test = days 45..60 (contains a fraud pattern never seen in training)

# Business cost model used to pick the decision threshold (INR).
COST_FALSE_POSITIVE = 100.0   # ops review + customer friction per wrongly flagged legit txn
# cost of a missed fraud = the transaction amount itself

FEATURE_COLUMNS = [
    # raw transaction
    "amount", "log_amount", "hour", "is_night", "dow", "channel_code", "category_code", "is_p2p",
    # user behaviour / velocity
    "u_txn_count_total", "u_secs_since_last", "u_cnt_10m", "u_cnt_1h", "u_cnt_24h",
    "u_amt_1h", "u_amt_24h", "u_small_cnt_10m", "u_amt_zscore", "u_amt_ratio_mean",
    "u_new_device", "u_new_city", "u_new_merchant", "u_n_devices",
    # device graph
    "d_n_users", "d_cnt_1h",
    # payee / merchant graph
    "m_age_hours", "m_users_1h", "m_users_24h",
]
RAW_ONLY_COLUMNS = ["amount", "log_amount", "hour", "is_night", "dow",
                    "channel_code", "category_code", "is_p2p"]

CHANNELS = ["UPI", "CARD", "NETBANKING"]
CATEGORIES = ["grocery", "food_delivery", "fuel", "electronics", "travel", "utilities",
              "fashion", "P2P", "gaming", "gift_cards", "crypto"]

# Human-readable reason codes for model explanations.
REASON_TEXT = {
    "amount": "High transaction amount",
    "log_amount": "High transaction amount",
    "hour": "Unusual time of day",
    "is_night": "Night-time transaction",
    "category_code": "High-risk merchant category",
    "is_p2p": "Peer-to-peer transfer",
    "channel_code": "Risky payment channel",
    "u_txn_count_total": "Low account history",
    "u_secs_since_last": "Transactions in rapid succession",
    "u_cnt_10m": "Burst of transactions in last 10 min",
    "u_cnt_1h": "High transaction velocity (1h)",
    "u_cnt_24h": "High transaction velocity (24h)",
    "u_amt_1h": "High spend in last hour",
    "u_amt_24h": "High spend in last 24h",
    "u_small_cnt_10m": "Many tiny transactions (card-testing pattern)",
    "u_amt_zscore": "Amount far above user's normal spend",
    "u_amt_ratio_mean": "Amount many times user's average",
    "u_new_device": "New device for this user",
    "u_new_city": "New location for this user",
    "u_new_merchant": "First payment to this payee/merchant",
    "u_n_devices": "Account used from many devices",
    "d_n_users": "Device shared across multiple accounts",
    "d_cnt_1h": "Device very active in last hour",
    "m_age_hours": "Newly appeared payee/merchant",
    "m_users_1h": "Payee receiving from many users (1h)",
    "m_users_24h": "Payee receiving from many users (mule pattern)",
    "if_score": "Anomalous overall behaviour (Isolation Forest)",
}

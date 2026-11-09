from pathlib import Path

import pandas as pd
import pytest

from src.adapters import CANONICAL, from_ieee, from_paysim, validate

FX = Path(__file__).parent / "fixtures"


def test_paysim_schema_and_leakage_columns_dropped():
    df = from_paysim(FX / "paysim_sample.csv")
    assert list(df.columns) == CANONICAL
    assert not any("balance" in c.lower() for c in df.columns) and "isFlaggedFraud" not in df.columns
    assert df.is_fraud.sum() == 2 and (df.device_id == "UNKNOWN").all()
    assert df.timestamp.is_monotonic_increasing


def test_ieee_schema_and_identity_join():
    df = from_ieee(FX / "ieee_transaction_sample.csv", FX / "ieee_identity_sample.csv")
    assert list(df.columns) == CANONICAL and len(df) == 5
    assert df.set_index("txn_id").loc["IE2987003", "device_id"].startswith("IE_SAMSUNG")


def test_validate_drops_bad_rows_and_fills_optional():
    raw = pd.DataFrame({"timestamp": ["2026-01-01 10:00", "not a date", "2026-01-01 11:00"],
                        "user_id": ["a", "b", "c"], "amount": [10, 20, -5], "merchant_id": ["m", "m", "m"]})
    df = validate(raw)
    assert len(df) == 1 and df.device_id.iloc[0] == "UNKNOWN" and df.fraud_type.iloc[0] == "LEGIT"


def test_validate_requires_core_columns():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"user_id": ["a"]}))


def test_sparkov_mapping(tmp_path):
    from src.adapters import INR_PER_USD, from_sparkov
    cols = "Unnamed: 0,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,is_fraud"
    rows = ["0,2019-01-01 00:00:18,2703186189652095,fraud_Rippin Kub and Mann,misc_net,4.97,A,B,F,x,Moravian Falls,NC,1,1,1,1,j,1988-03-09,abc123,1,1,1,0",
            "1,2019-01-01 00:01:00,2703186189652095,fraud_Heller Gutmann,grocery_pos,107.23,A,B,F,x,Moravian Falls,NC,1,1,1,1,j,1988-03-09,def456,1,1,1,1"]
    f = tmp_path / "sp.csv"
    f.write_text(cols + "\n" + "\n".join(rows) + "\n")
    df = from_sparkov(f)
    assert list(df.columns) == CANONICAL and len(df) == 2
    assert df.merchant_id.tolist() == ["Rippin Kub and Mann", "Heller Gutmann"]          # "fraud_" prefix removed
    assert df.channel.tolist() == ["CARD_ONLINE", "CARD_POS"] and df.device_id.eq("UNKNOWN").all()
    assert abs(df.amount.iloc[1] - 107.23 * INR_PER_USD) < 0.01 and df.is_fraud.tolist() == [0, 1]
    assert "first" not in df and "dob" not in df                                     # personal fields dropped

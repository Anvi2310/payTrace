"""IEEE-CIS engineered features must be past-only: no row may see itself, a later row, or a label younger than 7 days."""
import numpy as np
import pandas as pd

from src.v21.ieee_opt import DAY, D_NORM, engineer


def frame(rows):
    base = {c: np.nan for c in D_NORM}
    out = []
    for i, (day, card, fraud, amt) in enumerate(rows):
        r = dict(base, TransactionID=i, TransactionDT=float(day * DAY + i), card1=card, addr1=100.0, D1=day,
                 TransactionAmt=amt, P_emaildomain="x.com", DeviceInfo=None, isFraud=fraud)
        out.append(r)
    return pd.DataFrame(out)


def test_features_use_only_earlier_rows_and_old_labels():
    # one card holder (D1 grows with the day, so day - D1 is constant), paying on days 0, 1, 2, 10, 11
    df = frame([(0, 1.0, 1, 10.0), (1, 1.0, 0, 20.0), (2, 1.0, 0, 30.0), (10, 1.0, 0, 40.0), (11, 1.0, 0, 50.0)])
    E = engineer(df)
    assert E.uid_n_past.tolist() == [0, 1, 2, 3, 4]                      # counts strictly earlier rows
    assert np.isnan(E.uid_amt_mean_past[0]) and E.uid_amt_mean_past[2] == 15.0   # mean of earlier amounts only
    assert E.uid_cnt_1d[0] == 0 and E.uid_cnt_7d[3] == 0                   # windows exclude the row itself
    # the day-0 fraud label becomes usable only 7 days later
    assert E.uid_conf_fraud_past.tolist() == [0, 0, 0, 1, 1]
    # changing a LATER row never changes an earlier row's features
    df2 = df.copy(); df2.loc[4, "TransactionAmt"] = 9999.0; df2.loc[4, "isFraud"] = 1
    E2 = engineer(df2)
    pd.testing.assert_frame_equal(E.iloc[:4], E2.iloc[:4])


def test_card_holder_needs_all_parts():
    df = frame([(0, 1.0, 0, 10.0), (1, 1.0, 0, 20.0)])
    df.loc[:, "addr1"] = np.nan
    E = engineer(df)
    assert E.uid_n_past.isna().all() and E.uid_conf_fraud_past.isna().all()

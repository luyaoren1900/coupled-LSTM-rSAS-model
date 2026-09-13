# -*- coding: utf-8 -*-
import numpy as np

import rsas
from model_data_es import SASPINNModel_data


INPUT_FILE = "plynlimon_data_1993-01-01_2008-12-31.csv"
OUTPUT_FILE = "Q_test.txt"

PHYSICS_PARAMS = {
    "Q_shape": 0.5794,
    "Q_p1": 5779.3530,
    "Q_p2": -141.8939,
    "ET_max": 1124.9066,
    "O_min": 100.0000,
    "O_shape": 2.3224,
    "O_p1": 2472.5669,
    "O_p2": -0.0100,
    "I_min": 48.5271,
    "I_max": 90834.5078,
    "I_a": 33.8841,
    "I_b": 0.1000,
    "dis_rate": 1.0000,
}


def main():
    data = SASPINNModel_data(
        inputfile=INPUT_FILE,
        start_date="1993-01-01",
        end_date="2008-12-31",
    )

    for name, value in PHYSICS_PARAMS.items():
        setattr(data, name, np.float32(value))

    zero_igf = np.zeros_like(data.J, dtype=np.float32)
    Q = np.stack([data.Q_LH, data.ET_LH, zero_igf, zero_igf], axis=1)

    outputs = rsas.solve(
        data.dis_rate,
        data.J,
        Q,
        data.rSAS_fun,
        mode="RK4",
        ST_init=data.ST_init,
        dt=1.0,
        n_substeps=1,
        full_outputs=True,
        CS_init=data.CS_init,
        C_J=data.C_J,
        alpha=data.alpha,
        k1=data.k1,
        C_eq=data.C_eq,
        C_old=data.C_old,
        verbose=False,
        debug=False,
    )

    q_test = outputs["C_Q"][:, 0, 0]
    np.savetxt(OUTPUT_FILE, q_test.reshape(-1, 1), fmt="%.6f")
    print(f"Saved {q_test.size} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

"""Extract the raw per-test measurements that master_tests.csv does not carry.

    python scripts/extract_raw_data.py "<path to NASA RESOURCES>" data/raw

Writes:
  data/raw/flex_droplets.csv  FLEX (PSI-69): one row per droplet burn, with initial diameter,
                              extinction diameter, burning-rate constant and burn time.
  data/raw/same_smoke.csv     SAME + SAME-R (PSI-101/102 combined workbook): one row per smoke
                              test, with particle sizes, mass concentration and detector signals.

Every row keeps the source file and row so the dashboard can cite it.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MMHG_TO_KPA = 0.133322


def num(s):
    """'1.69', '~5', '–' -> float (NaN when not a number). Keeps an 'approx' flag separately."""
    return pd.to_numeric(s.astype(str).str.replace("~", "", regex=False).str.strip()
                         .replace({"–": np.nan, "-": np.nan, "": np.nan, "nan": np.nan}), errors="coerce")


def flex(src: Path) -> pd.DataFrame:
    f = next(src.glob("PSI-69/*/PSI-69_Experimental table_FLEX.csv"))
    d = pd.read_csv(f, encoding="cp1252")
    out = pd.DataFrame({
        "test_number": d["FLEX Test #"],
        "test_id": d["FLEX Identifier"],
        "date": pd.to_datetime(d["Test Date"], errors="coerce").dt.date,
        "fuel": d["Fuel"],
        "pressure_kpa": (d["Ambient pressure; mmHg"] * MMHG_TO_KPA).round(2),
        "o2_pct": (d["O initial ambient composition; mole fraction"] * 100).round(2),
        "n2_pct": (d["N initial ambient composition; mole fraction"] * 100).round(2),
        "co2_pct": (d["CO initial ambient composition; mole fraction"] * 100).round(2),
        "he_pct": (d["He initial ambient composition; mole fraction"] * 100).round(2),
        "d0_mm": num(d["Droplet initial diameter; mm"]),
        "d_ext_mm": num(d["Visible flame extinction diameter; mm"]),
        "burning_rate_mm2s": num(d["Burning rate; mm"]),
        "burn_time_s": num(d["Burn time; s"]),
        "burn_time_approx": d["Burn time; s"].astype(str).str.contains("~"),
        "test_end": d["Test end"].str.strip().str.lower(),
    })
    out["diluent"] = np.select([out.co2_pct > 0, out.he_pct > 0], ["CO2", "He"], "none")
    out["extinction_ratio"] = (out.d_ext_mm / out.d0_mm).round(3)  # fraction of the droplet left at extinction
    out["pressure_valid"] = out.pressure_kpa > 5  # a few rows record 0 mmHg
    out["source_file"] = f"PSI-69/{f.name}"
    out["source_row"] = d.index + 2
    return out


SAME_COLS = {
    "Flight 1=same 2=sameR, g=ground": "flight",
    "Sample Type": "material",
    "SAME #": "test_id",
    "Description": "description",
    "Inlet Velocity (cm/sec)": "flow_cms",
    "Rt/R0": "rt_r0",
    "Net Aging  Duration": "aging_s",
    "ISS Detector Scatter (volts)": "iss_detector_v",
    "STS Detector (volts)": "sts_detector_v",
    "M1/M0 (micro-m)--AMD": "count_mean_diam_um",
    "Diameter of Average mass (micro-m)": "avg_mass_diam_um",
    "Hatch Choate Sigma g": "sigma_g",
    "Hatch Choate Dg": "geo_mean_diam_um",
    "DustTrak B calibratred and corrected for dilution (mg/m3)": "mass_conc_mgm3",
    "Sample Temperature C": "sample_temp_c",
    "Mass Loss mg": "mass_loss_mg",
}


def same(src: Path) -> pd.DataFrame:
    f = next(src.glob("PSI-101/*/Combined SAMER Results*.xlsx"))
    d = pd.read_excel(f, sheet_name="Reduced Data with ISS Det", header=1)
    d = d[list(SAME_COLS)].rename(columns=SAME_COLS)
    d = d.dropna(subset=["material"]).copy()
    d["source_row"] = d.index + 3
    d["campaign"] = d.flight.map({1: "SAME", 1.0: "SAME", 2: "SAME-R", 2.0: "SAME-R"}).fillna("ground")
    for c in d.columns:
        if c not in ("material", "description", "test_id", "campaign", "flight"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # zeros in the size/mass columns mean "not measured" (e.g. DBP calibration runs)
    for c in ("count_mean_diam_um", "avg_mass_diam_um", "geo_mean_diam_um", "sigma_g", "mass_conc_mgm3"):
        d.loc[d[c] <= 0, c] = np.nan
    d["material"] = d.material.str.strip()
    d["description"] = d.description.astype(str).str.strip()
    d["test_id"] = d.test_id.astype(str).str.strip()
    d["source_file"] = f"PSI-101/{f.name} (sheet: Reduced Data with ISS Det)"
    return d.drop(columns=["flight"])


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    fx, sm = flex(src), same(src)
    fx.to_csv(dst / "flex_droplets.csv", index=False)
    sm.to_csv(dst / "same_smoke.csv", index=False)
    print(f"flex_droplets.csv: {len(fx)} burns, {fx.d_ext_mm.notna().sum()} with extinction diameter")
    print(f"same_smoke.csv: {len(sm)} smoke tests, {sm.count_mean_diam_um.notna().sum()} with particle size")


if __name__ == "__main__":
    main(*sys.argv[1:3])

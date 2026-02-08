import numpy as np
from io import StringIO
import numpy as np
import pandas as pd

def parse_csi_line_csv(line, cols):
    """
    Parse a CSV line using csv.reader and return a dict.
    Handles quotes properly.
    """
    try:
        reader = csv.reader([line], quotechar='"', skipinitialspace=True)
        row = next(reader)
        if len(row) != len(cols):
            return None
        return dict(zip(cols, row))
    except Exception:
        return None

def safe_parse_csi_data(data_str):
    """
    Parse the data field containing an array of csi data
    
    :param data_str: string containing the csi data array
    """
    if not isinstance(data_str, str) or not data_str.startswith("["):
        return None
    try:
        return np.fromstring(data_str[1:-1], sep=',', dtype=float)
    except Exception:
        return None


def from_buffer_to_df_detection(buffer_csi, cols_csi, csi_data_length=384):
    """
    Convert the buffer containing the received data to a DataFrame for processing.
    Uses csv.reader for fast parsing and avoids creating DataFrame per line.
    """
    parsed_rows = []

    for line in buffer_csi:
        row = parse_csi_line_csv(line, cols_csi)
        if not row:
            continue

        if row.get("type") != "CSI_DATA":
            continue

        csi_raw = safe_parse_csi_data(row.get("data"))
        if csi_raw is None or len(csi_raw) != csi_data_length:
            continue

        row["csi_raw"] = csi_raw
        row["csi_len"] = csi_data_length

        parsed_rows.append(row)

    if not parsed_rows:
        return pd.DataFrame(columns=cols_csi + ["csi_raw", "csi_len"])

    return pd.DataFrame(parsed_rows, columns=cols_csi + ["csi_raw", "csi_len"])

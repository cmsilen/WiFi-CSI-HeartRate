import numpy as np
from io import StringIO
import numpy as np
import pandas as pd

def parse_csi_line(line, cols):
    result = None
    try:
        result = pd.read_csv(
            StringIO(line),
            names=cols,
            quotechar='"'
        )
    except pd.errors.ParserError:
        result = None
    return result

def safe_parse_csi_data(data_str):
    if not isinstance(data_str, str) or not data_str.startswith("["):
        return None
    try:
        return np.fromstring(data_str[1:-1], sep=',', dtype=float)
    except Exception:
        return None

def iterate_data_rcv(ser):
    strings = str(ser.readline())
    print("received line")
    if not strings:
        print("no string from csi port")
        return None
    strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
    return strings


def from_buffer_to_df_detection(buffer_csi, cols_csi, csi_data_length=384):
    parsed_rows = []

    for line_csi in buffer_csi:
        result_csi = parse_csi_line(line_csi, cols_csi)
        if result_csi is not None and not result_csi.empty:
            # Converti il risultato (DataFrame a riga singola) in dizionario e aggiungi alla lista
            parsed_rows.append(result_csi.iloc[0].to_dict())

    # Crea il DataFrame in un colpo solo
    df_csi = pd.DataFrame(parsed_rows, columns=cols_csi)

    # Filtra solo CSI_DATA
    df_csi = df_csi[df_csi["type"] == "CSI_DATA"].copy()

    df_csi["csi_raw"] = df_csi["data"].map(safe_parse_csi_data)
    df_csi.dropna(subset=["csi_raw"], inplace=True)
    df_csi["csi_len"] = df_csi["csi_raw"].map(len)
    df_csi = df_csi[df_csi["csi_len"] == csi_data_length].copy()

    return df_csi

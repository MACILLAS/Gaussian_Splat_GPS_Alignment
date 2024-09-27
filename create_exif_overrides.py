import json
import pandas as pd

df = pd.read_csv('data/GPS_aligned_0529.csv')

exif_override = {}

for i in range(len(df.index)):
    img = {
        "gps": {
            "latitude": df.iloc[i]["lla_x"],
            "longitude": df.iloc[i]["lla_y"],
            "altitude": df.iloc[i]["lla_z"],
            "dop": df.iloc[i]["rtk_std"]
        }
    }

    exif_override[df.iloc[i]["filename"]] = img

with open('../OpenSfM/data/Ford_Tower_0529/exif_overrides.json', 'w') as outfile:
    json.dump(exif_override, outfile)


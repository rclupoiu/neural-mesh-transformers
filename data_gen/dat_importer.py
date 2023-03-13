import numpy as np

# @title Data retrieval
import os, requests

fname = "stringer_spontaneous.npy"
url = "https://osf.io/dpqaj/download"

print("Downloading data from %s ..." % url)
if not os.path.isfile(fname):
  try:
    r = requests.get(url)
  except requests.ConnectionError:
    print("!!! Failed to download data !!!")
  else:
    if r.status_code != requests.codes.ok:
      print("!!! Failed to download data !!!")
    else:
      with open(fname, "wb") as fid:
        fid.write(r.content)

# @title Data loading
dat = np.load(fname, allow_pickle=True).item()

print(dat.keys())

print("Saving data locally...")

np.save('stringer_spontaneous.npy', dat)
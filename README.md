# sarpanel

Analysis-ready **Sentinel-1** observations over polygons, written to **Parquet**,
with the acquisition geometry and speckle statistics kept alongside the
backscatter.

Give the program a polygon (or many polygons) and a date range. It queries the
Alaska Satellite Facility (ASF) archive once, processes the matching Sentinel-1
scenes with [ESA SNAP](https://step.esa.int/main/toolboxes/snap/) through
[pyroSAR](https://github.com/johntruckenbrodt/pyroSAR), and writes **one row per
polygon per acquisition**. Each row has the backscatter estimate and the
information a statistical model needs in order to weight that estimate.

This repository used to ship an earlier tool called `sar2parquet`. That version
wrote one row per pixel. `sarpanel` replaces it. The install command, the
Python import, and the command-line program all use the name `sarpanel` now.

## What a row contains

| Field | Meaning |
|---|---|
| `VV_mean_db`, `VH_mean_db` | backscatter, averaged in linear power and then converted to decibels |
| `inc_angle_mean` / `std` / `min` / `max` | terrain-corrected projected local incidence angle |
| `looks_nominal` | multilooking implied by the processing configuration |
| `enl_empirical` | equivalent number of looks estimated from the data |
| `enl_efficiency` | ratio of the two, i.e. how much independence the processing retained |
| `n_pixels_valid` | pixels that survived nodata, layover, and shadow masking |
| `*_se_db` | standard error of the polygon mean |
| `*_cv` | linear-domain coefficient of variation (texture) |
| `relative_orbit`, `orbit_direction` | which track viewed the polygon, and whether the pass was ascending or descending |
| `layover_shadow_fraction` | share of the polygon that is not a real measurement |

A two-hectare plot and a hundred-hectare field observed on the same day do not
have the same precision. Precision depends on how many independent looks went
into the mean. Different tracks also arrive at different incidence angles, and
backscatter depends on that angle. `sarpanel` keeps those quantities so a later
model can use them.

## The two programs this package stands on

`sarpanel` does not download or geocode SAR data by itself. It drives two
existing tools. Both have to be installed, and they are installed differently.

### ASF Search (`asf_search`)

[ASF Search](https://github.com/asfadmin/Discovery-asf_search) is the Python
client for the [Alaska Satellite Facility](https://search.asf.alaska.edu/)
catalog. Sentinel-1 scenes are free. ASF holds the archive. `asf_search` is how
this package asks "which scenes overlap my polygons between these dates?" and
how it downloads the zip files afterwards.

The catalog query itself does not need a login. The **download** does. You need
a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account, and you have
to authorize ASF's application once. The credentials section below is the part
people usually miss.

Install it inside the conda environment from the next section:

```bash
pip install "asf-search>=7"
```

`environment.yml` already does this. Confirm the import:

```bash
python -c "import asf_search; print(asf_search.__version__)"
```

### pyroSAR and ESA SNAP

[pyroSAR](https://github.com/johntruckenbrodt/pyroSAR) is a Python library that
builds SNAP processing graphs and runs them. It does **not** process the radar
data on its own. The program that actually calibrates, terrain-flattens, and
geocodes a scene is `gpt`, which ships with **ESA SNAP**.

Think of it this way:

- **SNAP** is the engine. It is a desktop/scientific application from ESA, not a pip package.
- **pyroSAR** is the Python remote control for that engine.
- **sarpanel** decides which scenes to process and what to write into the table.

Install pyroSAR inside the same conda environment:

```bash
pip install "pyroSAR>=0.24"
```

Then install SNAP separately:

1. Download the **Sentinel Toolboxes** installer from the
   [SNAP download page](https://step.esa.int/main/download/snap-download/).
2. Run the installer. On Windows the folder is often
   `C:\Program Files\esa-snap` or `C:\Program Files\snap`. On Linux it is
   wherever you unpacked it, often `$HOME/snap`.
3. That folder must contain `bin/gpt` (Windows: `bin\gpt.exe`).
4. Tell the shell where it is, and put `bin` on `PATH`.

PowerShell, for the current window:

```powershell
$env:SNAP_HOME = "C:\Program Files\esa-snap"
$env:PATH = "$env:SNAP_HOME\bin;" + $env:PATH
gpt -h
```

bash:

```bash
export SNAP_HOME="$HOME/snap"
export PATH="$SNAP_HOME/bin:$PATH"
gpt -h
```

`gpt -h` should print SNAP's help, not "command not found".

pyroSAR has to be new enough to write the incidence angle as a real output
band (`export_extra`). `sarpanel doctor` checks this. If it reports
`export_extra: False`, upgrade pyroSAR before processing anything:

```bash
pip install -U "pyroSAR>=0.24"
```

Searching the catalog works without SNAP. Producing Parquet does not.

## Install on a new computer

The geospatial libraries (`geopandas`, `rasterio`) sit on top of GDAL.
Installing those with pip on Windows is a common way to get a broken
environment. Use conda, and keep this project out of the base Anaconda
environment.

If you do not have conda yet, install
[Miniforge](https://github.com/conda-forge/miniforge) (or Anaconda).

### 1. Create the environment

From a clone of this repository:

```bash
git clone https://github.com/Dev-Arnab/sar2parquet.git
cd sar2parquet
conda env create -f environment.yml
conda activate sarpanel
```

Or, without the file, the same stack:

```bash
conda create -n sarpanel -c conda-forge python=3.11 ^
    geopandas rasterio shapely pyproj pandas pyarrow pyyaml ipykernel
conda activate sarpanel
pip install "asf-search>=7" "pyroSAR>=0.24"
```

On bash, replace the `^` line continuation with `\`.

### 2. Install sarpanel itself

`--no-deps` matters. It tells pip to install this package without replacing the
conda builds of `rasterio` and `geopandas`. Replacing those with pip wheels is
a reliable way to break GDAL.

From the cloned repository (so you can edit the code):

```bash
pip install --no-deps -e .
```

From GitHub, into an environment that already has the conda stack:

```bash
pip install --no-deps "git+https://github.com/Dev-Arnab/sar2parquet.git"
```

Check that the new package is the one Python sees:

```bash
python -c "import sarpanel; print(sarpanel.__version__)"
sarpanel doctor
```

`doctor` prints software versions, whether Earthdata credentials work, whether
pyroSAR can export the incidence angle, and whether `SNAP_HOME` points at `gpt`.

### 3. Jupyter

If a notebook cannot import `sarpanel`, Jupyter is almost certainly running a
different Python than the `sarpanel` environment. Register the environment and
select it.

```bash
conda activate sarpanel
pip install notebook
python -m ipykernel install --user --name sarpanel --display-name "Python (sarpanel)"
jupyter notebook
```

In the notebook: **Kernel → Change kernel → Python (sarpanel)**. Confirm:

```python
import sys
print(sys.executable)   # the path should contain sarpanel
```

## Earthdata login

Do this once per account.

1. Register at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/).
2. Open your profile and **Generate Token**. Copy the whole `eyJ...` string,
   with no spaces before or after it.
3. Authorize the ASF application. Until you do, downloads fail with
   `Pre authorization required for this application` even when the token is
   valid. Log in, open **Applications → Authorized Apps**, and approve
   **Alaska Satellite Facility Data Access**. You can also open the
   `resolution_url` printed in the error.

On your own computer, put the token in the environment for that terminal
session. The package does not accept it as a command-line flag, because flags
show up in the process list and in shell history.

PowerShell:

```powershell
$env:EARTHDATA_TOKEN = "paste-the-token-here"
```

bash:

```bash
export EARTHDATA_TOKEN="paste-the-token-here"
```

On a shared Linux cluster, a file is more reliable than an export, because a
batch job does not always inherit your shell variables:

```bash
printf '%s\n' "$EARTHDATA_TOKEN" > ~/.earthdata_token
chmod 600 ~/.earthdata_token
```

A `~/.netrc` entry also works, and it must be readable only by you (`chmod 600`).
On Windows, prefer `EARTHDATA_TOKEN`. Python's permission check treats many
Windows files as group-readable and will refuse a token file or `.netrc` for
that reason.

Then run `sarpanel doctor` again. The credentials block should say
`VERIFIED against urs.earthdata.nasa.gov`.

`plan` can succeed before this step, because the catalog search is public.
The login is checked when a scene is downloaded.

## Run a small example

`configs/local_demo.yaml` is a bounding box in central Michigan and one month
of GRD data (June 2020). It does not need a field file.

```bash
conda activate sarpanel
sarpanel doctor
sarpanel plan configs/local_demo.yaml
sarpanel run  configs/local_demo.yaml --shard 0
sarpanel status configs/local_demo.yaml
sarpanel aggregate configs/local_demo.yaml
```

`plan` prints the scene count, tracks, and download size **before** anything is
downloaded. Read that number. A month over a small box is a few scenes. Several
years of SLC over hundreds of fields is terabytes.

Outputs land in `runs/local_demo/`:

```
runs/local_demo/
├── manifest/        scene list and a copy of the config
├── work/            downloads and SNAP intermediates
├── observations/    one Parquet file per scene
└── status/          ok, skipped, no_data, or failed, per scene
```

Open the result:

```python
import pandas as pd
df = pd.read_parquet("runs/local_demo/observations")
print(df[["acquisition_date", "relative_orbit", "inc_angle_mean",
          "n_pixels_valid", "enl_empirical", "VV_mean_db", "VV_se_db"]].head())
```

Column names follow the scene table. If a name differs, `df.columns` is the
list to trust.

What a healthy smoke test looks like:

| Column | Expected |
|---|---|
| `inc_angle_mean` | about 29 to 46 degrees, and different tracks differ |
| `enl_empirical` | a small number of looks, not missing |
| `VV_mean_db` | roughly -8 to -18 dB for vegetated cropland in June |
| `n_pixels_valid` | more than a handful |

If `inc_angle_mean` is empty, pyroSAR did not write the incidence-angle band.
`sarpanel doctor` is the place to start.

### Your own fields

Any of these can be the `aoi.source` in the YAML file:

```yaml
source: "-84.60,42.68,-84.55,42.72"          # one bounding box: minx,miny,maxx,maxy
source: "POLYGON ((-84.6 42.6, ...))"        # WKT, longitude then latitude
source: "D:/fields/my_fields.gpkg"           # GeoPackage, GeoJSON, Shapefile, ...
```

For a file, set `id_column` to the column that names each field (for example
`farm_name`). Set `min_area_m2` to drop polygons that are too small to sample.
Set `where` to an attribute filter, such as `farm_name == '520_Pitsilo'`.

The file must have a CRS. The package raises if it does not, because assuming
longitude/latitude for a projected file produces an empty search that looks
like "no satellite data".

`processing.target_crs` must be a **projected** CRS in metres, such as the UTM
zone that contains the fields (`EPSG:32616` for this Michigan box). In
EPSG:4326 the pixel size would be in degrees.

`output.granularity` is `polygon` (one row per field per pass) or `pixel`
(one row per pixel). Start with `polygon`.

The other YAML files under `configs/` are the project's cluster runs. Their
`source:` and `output.root:` paths point at a specific HPC filesystem. Use
`local_demo.yaml` as the template on your own machine.

### The five commands

| Command | What it does |
|---|---|
| `sarpanel doctor` | Checks Python packages, Earthdata login, pyroSAR features, and SNAP |
| `sarpanel plan CONFIG` | Searches ASF once and writes the scene manifest |
| `sarpanel run CONFIG --shard 0` | Downloads and processes the scenes in that shard |
| `sarpanel status CONFIG` | Counts finished, failed, and empty scenes |
| `sarpanel aggregate CONFIG` | Stacks per-scene Parquet files into one table |

`run --help` lists the flags. Shards exist so a cluster can process the
manifest in parallel. On a laptop, `n_shards: 1` and `--shard 0` are enough.

On SLURM, create `logs/` first, set `SNAP_HOME`, and submit:

```bash
mkdir -p logs
export SNAP_HOME=/path/to/snap
sbatch slurm/plan.sb configs/your_run.yaml
sbatch --array=0-0 slurm/run_shard.sb configs/your_run.yaml
```

The array range is `0` through `n_shards - 1`. `slurm/plan.sb` and
`slurm/run_shard.sb` read `SARPANEL_ENV`, `SNAP_HOME`, and `SARPANEL_SCRATCH`
from the environment instead of a single user's home directory.

## Choices that change the result

**The incidence angle is a terrain-correction output band.** It is requested
through pyroSAR's `export_extra`, so SNAP writes it on the same grid as the
backscatter. Reading the tie-point incidence grid out of the SNAP intermediate
directory is a different grid (pre-terrain-correction). Those numbers can look
plausible and still be in the wrong place. `sarpanel doctor` warns when the
installed pyroSAR cannot write the band.

**The catalog is queried once.** `plan` writes a manifest. Workers read their
assigned rows. Searching again inside every worker can skip or repeat scenes,
because result order is not stable.

**Filenames include the scene id.** Two granules can share a date and an orbit
direction. Naming files only by date would drop one of them.

**Means are computed in linear power.** Averaging decibels pulls the result
low, and the pull is larger where the field is more heterogeneous. The package
refuses `scaling: dB` at geocode time. Decibels are applied when the polygon
mean is reported.

**A finished scene is detected before it is downloaded again.** Rerunning
`run` continues the unfinished scenes.

**SLC and GRD are both supported.** `product_type: GRD` is the right first
test: smaller downloads, faster processing. `product_type: SLC` keeps phase
and gives explicit control of multilooking, at roughly five times the download
volume. For SLC, set `search.relative_orbits` to the tracks you actually want.
Different tracks have different viewing geometry.

**Speckle filtering is off by default.** A speckle filter changes the
equivalent-number-of-looks estimate that the standard errors come from.

## Tests

The tests cover radiometry, extraction, AOI parsing, the manifest, and
provenance. They do not need SNAP or a login.

```bash
conda activate sarpanel
pytest -q
```

## Troubleshooting

| What you see | What to do |
|---|---|
| `ModuleNotFoundError: geopandas` or `rasterio` | The conda environment is not active, or those packages were never installed into it. Recreate it from `environment.yml`. |
| NumPy 1.x vs 2.x import errors, or the wrong `sys.executable` | Jupyter is using base Anaconda. Select the **Python (sarpanel)** kernel. |
| `sarpanel` is not recognized | The environment is not active, or `pip install --no-deps -e .` has not been run in it. |
| `Invalid/Expired token` | The token was copied with a space, or it has expired. Generate a new one. |
| `Pre authorization required` | The Earthdata account has not approved the ASF application. See the login section. |
| `PermissionError` on `.netrc` or `.earthdata_token` | On Linux, `chmod 600` that file. On Windows, use `EARTHDATA_TOKEN` in the environment instead. |
| Connection reset while logging in | A firewall, VPN, or proxy is blocking `urs.earthdata.nasa.gov`. Try another network. |
| `plan` reports no scenes | The date range does not cover this place. Sentinel-1's revisit is several days, and 2022 is sparse after Sentinel-1B failed. Widen the window or try 2020 or 2024–2025. Also check that a vector file has a CRS. |
| `export_extra: False` | Upgrade pyroSAR. Processing would otherwise leave the incidence angle empty. |
| `gpt` is not found | Install SNAP and set `SNAP_HOME`. `sarpanel doctor` prints the path it checked. |
| Process killed with code -9 during processing | SNAP ran out of memory. Use a machine with more RAM, or set `snap_tile_cache_gb` and `snap_threads` in the YAML. A full scene is not a laptop-sized job. |
| Scenes marked `no_data` | The polygon sits on the edge of the swath and has too few valid pixels. That status is expected for some scenes. |

## Authors

`sarpanel` is developed by **Dev Jyoti Ghosh Arnab** and **Ryan Jayne**.

The package is a joint effort to turn a Sentinel-1 processing workflow into an
installable tool: search, SNAP processing, and polygon-level tables with
geometry and speckle statistics included.

## License

[MIT](LICENSE)

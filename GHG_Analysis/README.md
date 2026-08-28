# Tanager methane monitoring for Southeast Asia

This repository asks how targeted Tanager observations could improve facility-level methane monitoring in Southeast Asia, with palm-oil mill effluent ponds as the main use case. Public regional plume catalogues show where evidence is thin. Two open Tanager scenes over Brazil then provide a practical test of methane retrieval from radiance data at facility scale.

The executive summary and final presentation are submitted separately, so they are not duplicated in this repository.

## Notebook order

1. `notebooks/01_sea_satellite_analysis.ipynb` builds the regional catalogue, compares the public satellite records, and explains the Carbon Mapper and Tanager gap audit.
2. `notebooks/02_tanager_scene_overview.ipynb` inspects the two open Tanager scenes and their provider plume records.
3. `notebooks/03_spectral_diagnostics.ipynb` checks the methane-sensitive bands and compares the 2.3 and 1.65 micrometre windows.
4. `notebooks/04_matched_filter.ipynb` runs the v2 two-pass CWMF at 2.3 micrometres and keeps the 1.65 micrometre result as a cross-check.

Figures are saved in `figures/`. Every filename starts with the number of the notebook that explains it.

## Set up the environment

Open PowerShell in this folder and create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## Download the large Tanager cubes

The HDF5 cubes are not included because they add about 5.7 GB. The small scene assets and all rendered notebook outputs are included. Before rerunning Notebooks 3 and 4, download the spectral cubes from Planet's public URLs:

```powershell
python scripts\download_ghg.py --hdf5 --basic
```

The downloader resumes partial files and skips complete files. Notebook 1 uses only the frozen regional extracts included in the repository. Notebook 2 uses the included GeoTIFF, GeoJSON, mask, and thumbnail assets.

Start Jupyter after the downloads finish:

```powershell
jupyter lab
```

Run the notebooks in numerical order. They locate the repository root automatically when opened from this folder or from `notebooks\`.

## Scripts

- `scripts/sea_analysis_pipeline.py` rebuilds the regional tables and figures used by Notebook 1.
- `scripts/analyze_carbon_mapper_tanager_sea.py` rebuilds the detailed catalogue gap audit from the included API snapshot. Use `--refresh` only to query the live public API.
- `scripts/download_ghg.py` downloads or resumes the Tanager assets needed for the spectral notebooks.
- `scripts/matched_filter_v2.py` runs the v2 two-pass CWMF used by Notebook 4.
- `scripts/refresh_sea_public_sources.py` optionally refreshes SRON or NASA metadata and requires network access.

Run a script from the repository root like this:

```powershell
python scripts\analyze_carbon_mapper_tanager_sea.py
```

## Data layout

- `data/ghg/` contains small assets and metadata for the two Tanager scenes. Downloaded HDF5 files also go here.
- `data/hitran_cache/` contains the methane line data used by the matched filter.
- `data/sea_satellite_analysis/raw_extracts/` contains frozen public catalogue extracts.
- `data/sea_satellite_analysis/analysis/` contains the derived tables behind Notebook 1 and the gap audit.

## Interpretation limits

The regional tables combine public plume catalogues, not complete observation histories. They cannot rank sensor detection probabilities. A nearby record on another date is not a same-day validation. The Brazil retrieval tests the method on selected offshore glint scenes; it does not establish a tropical land detection limit. Provider emission rates remain the quantitative reference because an independent flux estimate depends strongly on wind and plume-mask assumptions.

Paper references and dataset credits are listed in `CITATIONS.md`. The paper PDFs are not included in the repository.

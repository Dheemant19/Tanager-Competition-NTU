## Access the Tanager Workbench

Use the **[Tanager Workbench](https://www.tanager-workbench.app/)** to explore Tanager imagery directly in your browser—no GIS software or Python setup required.
The map-first application supports scene discovery, spectral sampling, custom band composites, region comparisons, coastal and greenhouse-gas analysis, and exportable results. A built-in tutorial and video walkthrough guide users through the complete workflow.


# Tanager Coastal and Greenhouse Gas Analysis

This repository contains two complementary demonstrations of Planet Tanager hyperspectral imagery: coastal water-quality analysis and facility-scale methane monitoring. The studies share one Python environment but retain separate notebooks, scripts, data products, figures, citations, and detailed documentation.

Satellite-derived products in this repository are research estimates. They are not substitutes for co-located field measurements or operational emissions verification.

## Analyses

### Coastal water quality

[`Coastal_Analysis/`](Coastal_Analysis/) examines the Sangatta River plume on the Makassar Strait coast of East Kalimantan, Indonesia. Its workflow progresses from scene screening and spectral exploration to turbidity and water-quality retrievals, a same-day Sentinel-2 comparison, uncertainty analysis, and the CoastCheck monitoring capstone.

See the [Coastal Analysis README](Coastal_Analysis/README.md) for the notebook sequence, data layout, scripts, glossary, and citations.

### Greenhouse gases

[`GHG_Analysis/`](GHG_Analysis/) investigates how targeted Tanager observations could support facility-level methane monitoring in Southeast Asia, with palm-oil mill effluent ponds as the principal use case. It combines a regional public-catalogue gap analysis with methane retrieval tests on two open Tanager scenes over Brazil.

See the [GHG Analysis README](GHG_Analysis/README.md) for the notebook sequence, download instructions, scripts, data layout, interpretation limits, and citations.

## Repository layout

```text
.
|-- Coastal_Analysis/       Coastal notebooks, scripts, data, and figures
|-- GHG_Analysis/           Methane notebooks, scripts, data, and figures
|-- Tanager-Workbench/      Separate interactive exploration application
|-- README.md               Shared entry point for the two analyses
`-- requirements.txt        Shared Coastal and GHG Python dependencies
```

The `Tanager-Workbench` application is included in the repository but is maintained as a separate project with its own [README](Tanager-Workbench/README.md) and dependency files.

## Shared environment setup

From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Then start JupyterLab from the repository root:

```powershell
jupyter lab
```

Open either analysis directory and run its notebooks in the order documented in that directory's README.

## Large data downloads

Most small inputs and rendered outputs are included. The large Tanager HDF5 cubes are omitted and can be downloaded independently for each workflow:

```powershell
python Coastal_Analysis\scripts\download_coastal.py --hdf5
python GHG_Analysis\scripts\download_ghg.py --hdf5 --basic
```

Review each downloader's help and the component README before downloading:

```powershell
python Coastal_Analysis\scripts\download_coastal.py --help
python GHG_Analysis\scripts\download_ghg.py --help
```

The downloaders resume partial files and skip files that are already complete. Some optional refresh scripts query live public services and therefore require network access.

## Reproducing the work

1. Install the shared environment from the root `requirements.txt`.
2. Read the relevant component README and download any omitted spectral cubes.
3. Launch JupyterLab from the repository root.
4. Run notebooks in the documented numerical order; later notebooks may depend on products from earlier stages.
5. Use the scripts in each component's `scripts/` directory to refresh inventories or rebuild derived products where documented.

For scientific assumptions, source attribution, and interpretation constraints, consult [`Coastal_Analysis/citations.md`](Coastal_Analysis/citations.md), [`Coastal_Analysis/notebooks/GLOSSARY.md`](Coastal_Analysis/notebooks/GLOSSARY.md), and [`GHG_Analysis/CITATIONS.md`](GHG_Analysis/CITATIONS.md).

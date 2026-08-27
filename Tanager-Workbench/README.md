# Tanager Workbench

Tanager Workbench is a local web application for exploring Tanager scenes, sampling spectra, drawing regions of interest, creating composites, and viewing coastal and greenhouse-gas products. The local server serves the browser app and its Python API routes together.

## Prerequisites

Install Python 3.11 or newer; Python 3.13 is the project default. An internet connection is recommended for interactive features that retrieve public remote scene data.

## Run on Windows

1. Open PowerShell and change to this application folder:

   ```powershell
   cd C:\path\to\Tanager-Competition-NTU\Tanager-Workbench
   ```

2. Create and activate an isolated Python environment:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass`, then repeat the activation command.

3. Install the application dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Start the local server:

   ```powershell
   python serve.py
   ```

5. Open [http://127.0.0.1:3000](http://127.0.0.1:3000) in a browser. Press `Ctrl+C` to stop the server, then run `deactivate` when finished.

## Run on macOS

1. Open Terminal and change to the application folder:

   ```bash
   cd /path/to/Tanager-Competition-NTU/Tanager-Workbench
   ```

2. Create and activate an environment, install dependencies, and run the server:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python serve.py
   ```

3. Visit [http://127.0.0.1:3000](http://127.0.0.1:3000). Stop it with `Ctrl+C` and run `deactivate` when finished.

## Validate and troubleshoot

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

Use a different local port with `python serve.py --port 3001`. Always run commands from this folder so the server can find `api/`, `data/`, and `images/`. Vercel deployment settings are in `vercel.json`; no Vercel CLI is required for local use.

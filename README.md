# SKHASH Polarity Pipeline

A complete workflow for seismic event processing, polarity analysis, and focal mechanism determination using SKHASH.

## Overview

This pipeline processes seismic event data from SeisComP XML files (local or remote URL) to:
- Analyze P-wave polarities using machine learning models
- Calculate S/P amplitude ratios
- Generate SKHASH input files
- Compute focal mechanism solutions
- Export comprehensive results in JSON format

## Requirements

- Python 3.x
- ObsPy
- TensorFlow/Keras
- pandas
- numpy
- matplotlib
- PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration Setup

Before running the pipeline, configure `config.yaml` with your settings:

### Required User Configuration

Edit `config.yaml` and fill in the following:

```yaml
# FDSN client settings
fdsn:
  client: "host_to_fdsn_server"  # Replace with your FDSN server or "GFZ", "USGS", etc.
  auth:
    username: "your_username"      # Your FDSN username
    password: "your_password"      # Your FDSN password

fdsn_inv:
  client: "host_to_fdsn_server_for_inventory"  # FDSN server for station inventory

# TauPy model for theoretical S arrival
taup_model: "iasp91"  # Default velocity model

# Cache directory for waveforms
cache_dir: "./waveform_cache"  # Local cache to store downloaded waveforms

# Resample rate for waveforms (Hz)
resample_rate: 100

# Model input settings
model:
  sample_rate: 100         # Should match resample_rate
  window_length: 0.25      # seconds for polarity window
```

**Important:** Replace placeholder values with your actual credentials and server information.

## Quick Start

### Pre-requisites Checklist

Before running the pipeline, ensure:

- [ ] Python 3.x installed with required packages (`pip install -r requirements.txt`)
- [ ] `config.yaml` edited with your FDSN server credentials
- [ ] SeisComP XML file or URL available
- [ ] TensorFlow model file (.h5) available
- [ ] Network connection to FDSN server (for downloading waveforms and remote XML)

### Full Workflow

Process an event from XML to final focal mechanism:

**Using local XML file:**
```bash
python workflow.py --xml bmg2026gkek.xml --model polarity_model.h5 --output-dir results
```

**Using remote XML URL:**
```bash
python workflow.py --xml "http://your-seiscomp-server/fdsnws/event/1/query?eventid=bmg2026gkek" --model polarity_model.h5 --output-dir results
```

This single command will:
1. Download waveforms
2. Classify polarities
3. Compute S/P ratios
4. Create all input files
5. Run SKHASH inversion
6. Generate focal mechanism and JSON output

### Verify Existing Results

Check if all required files are present:

```bash
python workflow.py --verify-only --event-id bmg2026gkek --output-dir results
```

## Workflow Steps

The pipeline executes 6 steps automatically:

### Step 1: Process Event
- Downloads waveform data from FDSN server
- Applies polarity classification using TensorFlow model
- Computes S/P amplitude ratios
- Generates `amp.csv` and `pol.csv`
- Creates `control_file.txt` with SKHASH parameters

### Step 2: Extract Metadata
- Reads event information from XML
- Creates `eq_catalog.csv` with:
  - Origin time, location, depth
  - Magnitude
  - Uncertainties
- Generates `station.csv` with station coordinates

### Step 3: Verify Structure
- Checks all required files exist:
  - ✓ IN/amp.csv
  - ✓ IN/pol.csv
  - ✓ IN/eq_catalog.csv
  - ✓ IN/station.csv
  - ✓ control_file.txt
- Reports any missing files

### Step 4: Show Summary
- Displays data statistics:
  - Number of stations
  - S/P ratio range and mean
  - Polarity distribution (up/down/unknown)

### Step 5: Run SKHASH
- Executes focal mechanism inversion
- Performs grid search over strike/dip/rake space
- Finds best-fit solution
- Generates output files:
  - OUT/out.csv (best solution)
  - OUT/out_polagree.csv (quality metrics)
  - OUT/out_polinfo.csv (per-station info)

### Step 6: Export JSON
- Combines all results into `final_result.json`:
  ```json
  {
    "event": {...},
    "stations": [...],
    "analysis": [...],
    "focal_mechanism": [...]
  }
  ```
- Creates beachball visualization: `beachball_inafocmec.png`

## Project Structure

```
inafocmec/
├── workflow.py                      # Main pipeline script
├── polarity_tf_pipeline.py          # Polarity analysis module
├── SKHASH.py                        # SKHASH focal mechanism solver
├── config.yaml                      # ⚠️ USER CONFIG - Edit this file
├── bmg2026gkek.xml                  # Example event XML file
├── polarity_model.h5                # Pre-trained polarity classification model
├── requirements.txt                 # Python dependencies
├── functions/                       # Processing functions
│   ├── beachball_generator.py       # Focal mechanism visualization
│   ├── compute_mech.py              # Mechanism computation
│   ├── gridsearch_so.py             # Grid search optimization
│   ├── in_pol.py                    # Polarity input handler
│   ├── in_sp.py                     # S/P ratio input handler
│   ├── in_sta.py                    # Station input handler
│   ├── out.py                       # Output writer
│   └── ...
├── velocity_models/                 # Earth velocity models
│   ├── iasp91.txt                   # Default global model
│   ├── north.txt
│   ├── socal.txt
│   └── ...
├── waveform_cache/                  # Cached waveform data (auto-created)
│   └── *.mseed                      # Downloaded waveforms
└── results/                         # Output directory (auto-created)
    └── {event_id}/                  # One folder per event
        └── inafocmec/               # SKHASH structure
            ├── IN/                  # Input files (auto-generated)
            │   ├── amp.csv          # ✓ S/P amplitude ratios
            │   ├── pol.csv          # ✓ Polarity classifications
            │   ├── eq_catalog.csv   # ✓ Event metadata
            │   ├── station.csv      # ✓ Station information
            │   └── analysis.csv     # ✓ Detailed waveform analysis (optional)
            ├── OUT/                 # Output files (auto-generated by SKHASH)
            │   ├── out.csv          # ✓ Best focal mechanism solution
            │   ├── out_polagree.csv # ✓ Polarity agreement statistics
            │   ├── out_polinfo.csv  # ✓ Detailed polarity info
            │   ├── final_result.json# ✓ Combined JSON results
            │   ├── analysis_summary.json # ✓ Analysis summary
            │   ├── beachball_inafocmec.png  # ✓ Focal mechanism beachball
            │   └── {event_id}.png   # ✓ Additional focal mechanism plot
            └── control_file.txt     # ✓ SKHASH control file
```

### Legend
- **⚠️ USER CONFIG** - Files that require user configuration
- **✓** - Files automatically generated by the pipeline

## Output Files

All output files are automatically generated in `results/{event_id}/inafocmec/`

### Input Files (IN/) - Auto-generated by Pipeline

| File | Description | Columns |
|------|-------------|---------|
| **amp.csv** | S/P amplitude ratios | station, sp_ratio, distance_km, azimuth |
| **pol.csv** | P-wave polarities | station, p_polarity (1=up, -1=down, 0=unknown) |
| **eq_catalog.csv** | Event metadata | time, latitude, longitude, depth, horz_uncert_km, vert_uncert_km, mag, event_id |
| **station.csv** | Station information | station, latitude, longitude, elevation |
| **analysis.csv** | Detailed analysis | station, p_pick_time, s_pick_time, quality_metrics (optional) |

### Output Files (OUT/) - Generated by SKHASH

| File | Description | Content |
|------|-------------|---------|
| **out.csv** | Best focal mechanism | strike, dip, rake, misfit, variance_reduction |
| **out_polagree.csv** | Polarity statistics | agreement_percentage, total_stations, correct_polarities |
| **out_polinfo.csv** | Per-station polarity | station, observed_polarity, predicted_polarity, agreement |
| **final_result.json** | Complete results | Combined JSON with event, stations, analysis, focal_mechanism |
| **analysis_summary.json** | Processing summary | Summary of analysis steps and results |
| **beachball_inafocmec.png** | Focal mechanism beachball | Standard beachball plot |
| **{event_id}.png** | Focal mechanism plot | Additional visualization with event ID |

### Control File
- **control_file.txt** - SKHASH configuration file with paths to input data and processing parameters

## Input Requirements

### What You Need to Provide

1. **SeisComP XML file or URL** - Event data with:
   - Can be a local file path (e.g., `bmg2026gkek.xml`)
   - Or a remote URL (e.g., `http://server/fdsnws/event/1/query?eventid=...`)
   - Must contain:
     - Origin time
     - Hypocenter location (lat, lon, depth)
     - Magnitude
     - Phase picks (P and S arrivals)

2. **TensorFlow Model** - Pre-trained polarity classification model (.h5 file)
   - Provided model: `polarity_model.h5`

3. **Configuration** - Edit `config.yaml` with your FDSN server credentials

4. **Velocity Model** (optional) - Choose from `velocity_models/` or use default `iasp91`

### What is Auto-Generated

The pipeline automatically creates:
- Event directory structure
- All input files (amp.csv, pol.csv, eq_catalog.csv, station.csv)
- Control file for SKHASH
- Downloads waveform data and caches it
- Processes polarities using ML model
- Computes focal mechanism
- Generates beachball plot and JSON summary

## Command Line Options

```
--xml PATH/URL      Path to local XML file or remote XML URL (required)
                    Examples: bmg2026gkek.xml or 
                    http://server/fdsnws/event/1/query?eventid=...
--model PATH        Path to TensorFlow model file .h5 (required)
--output-dir DIR    Output directory (default: ./results)
--config PATH       Configuration file (default: config.yaml)
--verify-only       Only verify structure, don't process
--event-id ID       Event ID for verification mode
```

## Examples

### Process with custom configuration

```bash
python workflow.py --xml bmg2026gkek.xml --model polarity_model.h5 --output-dir results --config custom_config.yaml
```

### Process event from remote server

```bash
python workflow.py --xml "http://seiscomp-server.com/fdsnws/event/1/query?eventid=2026abcd&format=xml" --model polarity_model.h5 --output-dir results
```

### Process with different output directory

```bash
python workflow.py --xml my_event.xml --model polarity_model.h5 --output-dir my_results
```

## Output Interpretation

The final focal mechanism solution includes:
- **Strike, Dip, Rake** - Fault plane parameters
- **Polarity Agreement** - Percentage of consistent polarities
- **Station Coverage** - Number of stations used
- **Quality Metrics** - Solution reliability indicators

## Troubleshooting

### Common Issues

**"Authentication failed" or "Cannot connect to FDSN"**
- Check `config.yaml` and verify your FDSN credentials
- Ensure `fdsn.client` points to a valid server
- Test connection: `curl http://your-fdsn-server/fdsnws/dataselect/1/`

**"Missing waveforms"**
- Waveforms are auto-downloaded and cached in `waveform_cache/`
- Check FDSN server has data for your event time/stations
- Verify network connectivity

**"Model loading error"**
- Ensure TensorFlow/Keras version is compatible
- Check model file path is correct: `--model polarity_model.h5`
- Verify model file is not corrupted

**"SKHASH fails" or "No focal mechanism output"**
- Check all input files in `{event_id}/inafocmec/IN/` exist and are properly formatted
- Verify `control_file.txt` has correct paths
- Ensure sufficient polarity data (minimum 8-10 stations recommended)

**"Low polarity agreement (<60%)"**
- May indicate complex source mechanism
- Check for sufficient azimuthal station coverage
- Verify polarity picks are reliable
- Consider manual review of first motions

**"ModuleNotFoundError"**
- Run: `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

**"File not found: config.yaml"**
- Ensure you're running from the project root directory
- Or specify: `--config /path/to/config.yaml`

**"Cannot read XML file" or "Invalid XML"**
- For remote URLs: Check that the URL is accessible and returns valid XML
- For local files: Verify the file path is correct and file exists
- Test remote URL: `curl "http://your-server/fdsnws/event/1/query?eventid=..."`
- Ensure XML contains required event information (origin, picks, magnitude)

## License

INAFOCMEC - Indonesian Focal Mechanism Catalog

## Contact

For questions or issues, please contact the development team.

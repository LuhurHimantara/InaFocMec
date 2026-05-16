import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from obspy import UTCDateTime, read_events, Stream, Trace, read
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from scipy.signal import resample
import tensorflow as tf
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='obspy')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolarityPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if 'auth' in self.config.get('fdsn', {}):
            self.fdsn_client = Client(
                self.config['fdsn']['client'],
                user=self.config['fdsn']['auth']['username'],
                password=self.config['fdsn']['auth']['password']
            )
        else:
            self.fdsn_client = Client(self.config['fdsn']['client'])
        
        if 'auth' in self.config.get('fdsn_inv', {}):
            self.fdsn_client = Client(
                self.config['fdsn_inv']['client'],
                user=self.config['fdsn_inv']['auth']['username'],
                password=self.config['fdsn_inv']['auth']['password']
            )
        else:
            self.fdsn_client_inv = Client(self.config['fdsn_inv']['client'])
        
        
        self.taup_model = TauPyModel(model=self.config['taup_model'])
        self.model = None
        self.cache_dir = Path(self.config['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True)
        self.stations_info = []
        self.stations_info_lock = Lock()  # Thread-safe access to stations_info
        
    def parse_xml(self, xml_path):
        """Parse SeisComP XML file and extract event and station information."""
        logger.info(f"Parsing XML file: {xml_path}")
        try:
            catalog = read_events(xml_path)
            if len(catalog) != 1:
                raise ValueError("XML file must contain exactly one event")
            
            event = catalog[0]
            event_id = str(event.resource_id).split('/')[-1]
            origin = event.preferred_origin() or event.origins[0]
            origin_time = origin.time
            latitude = origin.latitude
            longitude = origin.longitude
            depth_km = origin.depth / 1000 if origin.depth else None
            
            stations_data = []
            for pick in event.picks:
                station = pick.waveform_id.station_code
                network = pick.waveform_id.network_code
                location = pick.waveform_id.location_code or '--'
                channel = pick.waveform_id.channel_code
                
                arrival = None
                for arr in origin.arrivals:
                    if arr.pick_id == pick.resource_id:
                        arrival = arr
                        break
                
                if arrival:
                    distance_deg = arrival.distance
                    azimuth = arrival.azimuth
                    phase = arrival.phase
                    
                    stations_data.append({
                        'event_id': event_id,
                        'origin_time': origin_time,
                        'latitude': latitude,
                        'longitude': longitude,
                        'depth_km': depth_km,
                        'station': station,
                        'network': network,
                        'location': location,
                        'channel': channel,
                        'phase': phase,
                        'arrival_time': pick.time,
                        'distance_deg': distance_deg,
                        'azimuth': azimuth
                    })
            
            df = pd.DataFrame(stations_data)
            df = df[df['network'] == 'IA']  # Filter out stations with network code 'XX'
            df.drop_duplicates(subset=['station', 'network', 'location', 'phase'], inplace=True)
            df.sort_values(by='distance_deg', ascending=True, inplace=True)
            # df = df[(df['distance_deg'] >= 5) & (df['distance_deg'] <= 20)]  # Filter by distance
            n_station = len(df)

            df = df.iloc[20:200]
                     
            # df = df.head(100)  # Limit to first 100 picks for performance
            logger.info(f"Parsed {len(df)} picks from XML")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            raise
    
    def load_tf_model(self, model_path):
        """Load TensorFlow model from .h5 file."""
        logger.info(f"Loading model: {model_path}")
        try:
            # Recreate the exact model architecture from the training script
            input_shape = (26, 1)
            input_layer = tf.keras.layers.Input(shape=input_shape, name='input_layer_3')
            
            # Block 1
            x1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(input_layer)
            x1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
            x1 = tf.keras.layers.MaxPooling1D(pool_size=1)(x1)
            x1 = tf.keras.layers.Dropout(0.5)(x1)

            # Block 2
            x2 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
            x2 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x2)
            x2 = tf.keras.layers.MaxPooling1D(pool_size=1)(x2)
            x2 = tf.keras.layers.Dropout(0.5)(x2)

            # Block 3
            x3 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x2)
            x3 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x3)
            x3 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x3)
            x3 = tf.keras.layers.MaxPooling1D(pool_size=1)(x3)
            x3 = tf.keras.layers.Dropout(0.5)(x3)

            # Block 4
            x4 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x3)
            x4 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x4)
            x4 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x4)
            x4 = tf.keras.layers.MaxPooling1D(pool_size=1)(x4)
            x4 = tf.keras.layers.Dropout(0.5)(x4)

            # Block 5
            x5 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x4)
            x5 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x5)
            x5 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x5)
            x5 = tf.keras.layers.MaxPooling1D(pool_size=1)(x5)
            x5 = tf.keras.layers.Dropout(0.5)(x5)
            
            # Concatenate outputs from blocks 3, 4, 5
            x = tf.keras.layers.Concatenate()([x3, x4, x5])
            
            # Flatten and output
            x = tf.keras.layers.Flatten()(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            
            self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)
            
            # Load weights from HDF5 file
            self.model.load_weights(model_path)
            
            logger.info("Model created and weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def download_waveforms(self, df, max_retries=1, max_workers=5):
        """Download waveforms for all stations using parallel threads."""
        logger.info(f"Downloading waveforms with {max_workers} parallel workers...")
        
        # Get unique stations to download
        station_rows = []
        seen_stations = set()
        
        for _, row in df.iterrows():
            station_key = f"{row['network']}.{row['station']}.{row['location']}"
            if station_key not in seen_stations:
                seen_stations.add(station_key)
                station_rows.append(row)
        
        waveforms = {}
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            futures = {
                executor.submit(
                    self._download_single_station, 
                    row, 
                    max_retries
                ): f"{row['network']}.{row['station']}.{row['location']}" 
                for row in station_rows
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(futures), desc="Downloading waveforms") as pbar:
                for future in as_completed(futures):
                    station_key = futures[future]
                    try:
                        station_key, waveform = future.result()
                        waveforms[station_key] = waveform
                    except Exception as e:
                        logger.error(f"Failed to download {station_key}: {e}")
                        waveforms[station_key] = None
                    finally:
                        pbar.update(1)
        
        return waveforms
    
    def _download_single_station(self, row, max_retries):
        """Download waveform for a single station (thread-safe)."""
        station_key = f"{row['network']}.{row['station']}.{row['location']}"
        start_time = row['origin_time'] - 1
        end_time = row['origin_time'] + row['distance_deg'] * 111 / 4.0 + 60
        if end_time - start_time < 300:
            end_time = start_time + 300
                   
        cache_file = self.cache_dir / f"{station_key}_{row['event_id']}.mseed"
        
        # Check cache first
        if cache_file.exists():
            try:
                st = read(str(cache_file))
                self._collect_station_info(row, start_time, end_time, level="station")
                return station_key, st
            except:
                pass
        
        # Download waveform with retries
        for attempt in range(max_retries):
            try:
                st = self.fdsn_client.get_waveforms(
                    network=row['network'],
                    station=row['station'],
                    location=row['location'],
                    channel="HN?",
                    starttime=start_time,
                    endtime=end_time
                )
                
                # Remove sensitivity
                try:
                    inv = self.fdsn_client_inv.get_stations(
                        network=row['network'],
                        station=row['station'],
                        location=row['location'],
                        channel="HN?",
                        starttime=start_time,
                        endtime=end_time,
                        level="response"
                    )
                    st.remove_sensitivity(inv)
                    logger.debug(f"Removed sensitivity for {station_key}")
                    
                    # Collect station information
                    self._extract_station_info(inv, row)
                    
                except Exception as e:
                    logger.warning(f"Could not remove sensitivity for {station_key}: {e}")
                    # Try to collect station info without response level
                    self._collect_station_info(row, start_time, end_time, level="station")
                
                # Preprocessing
                st.detrend("linear")
                st.detrend("demean")
                
                # Resample if configured
                if self.config.get('resample_rate'):
                    for tr in st:
                        tr.resample(int(self.config['resample_rate']))
                st.merge(fill_value=0)
                
                # Cache the waveform
                st.write(str(cache_file), format='MSEED')
                
                return station_key, st
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {station_key}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download {station_key} after {max_retries} attempts")
                    return station_key, None
                time.sleep(1)
        
        return station_key, None
    
    def _collect_station_info(self, row, start_time, end_time, level="station"):
        """Collect station info from FDSN (thread-safe)."""
        try:
            inv = self.fdsn_client_inv.get_stations(
                network=row['network'],
                station=row['station'],
                location=row['location'],
                channel="HN?",
                starttime=start_time,
                endtime=end_time,
                level=level
            )
            self._extract_station_info(inv, row)
        except Exception as e:
            logger.warning(f"Could not get station info for {row['network']}.{row['station']}: {e}")
    
    def _extract_station_info(self, inv, row):
        """Extract and store station info from inventory (thread-safe)."""
        for network_inv in inv:
            for station_inv in network_inv:
                station_info = {
                    'station': station_inv.code,
                    'network': network_inv.code,
                    'location': row['location'],
                    'channel': 'HNZ',
                    'latitude': station_inv.latitude,
                    'longitude': station_inv.longitude,
                    'elevation': station_inv.elevation
                }
                # Thread-safe append
                with self.stations_info_lock:
                    if station_info not in self.stations_info:
                        self.stations_info.append(station_info)
    
    def prepare_polarity_input(self, waveform, p_time, sample_rate):
        """Prepare input for polarity prediction."""
        window_start = p_time - 0.20
        window_end = p_time + 0.05
        
        # Get Z component
        z_comp = None
        for tr in waveform:
            if tr.stats.channel.endswith('Z'):
                z_comp = tr
                break
        
        if z_comp is None:
            return None
        
        # Cut window
        z_cut = z_comp.slice(window_start, window_end)
        if len(z_cut.data) == 0:
            return None
        
        # Z-score normalization: (x - mean) / std
        data = z_cut.data.astype(np.float32)
        if len(data) > 1 and np.std(data) > 0:
            data = (data - np.mean(data)) / np.std(data)
        elif np.max(np.abs(data)) > 0:
            # Fallback to max normalization if std is zero
            data = data / np.max(np.abs(data))
        
        # Reshape for model
        input_shape = self.model.input_shape
        if len(input_shape) == 3:  # CNN: (batch, timesteps, features)
            data = data.reshape(1, -1, 1)
        else:  # Dense: (batch, features)
            data = data.reshape(1, -1)
        
        return data
    
    def predict_polarity(self, input_data):
        """Predict polarity using the model."""
        if input_data is None:
            return 'x', 0.0
        
        try:
            pred = self.model.predict(input_data, verbose=0)
            
            if pred.shape[-1] == 1:  # Sigmoid (binary)
                prob = pred[0][0]
                polarity = 'U' if prob > 0.5 else 'D'
            elif pred.shape[-1] == 3:  # Softmax (3 classes: D, U, x)
                class_idx = np.argmax(pred[0])
                if class_idx == 0:
                    polarity = 'D'
                elif class_idx == 1:
                    polarity = 'U'
                else:  # class_idx == 2
                    polarity = 'x'
                prob = np.max(pred[0])
            else:  # Other multi-class
                class_idx = np.argmax(pred[0])
                polarity = str(class_idx)  # fallback
                prob = np.max(pred[0])
            
            return polarity, float(prob)
        except Exception as e:
            logger.warning(f"Polarity prediction failed: {e}")
            return 'x', 0.0
    
    def theoretical_s_pick(self, p_time, distance_deg, depth_km):
        """Calculate theoretical S arrival time."""
        try:
            arrivals = self.taup_model.get_travel_times(
                source_depth_in_km=depth_km,
                distance_in_degree=distance_deg,
                phase_list=['S']
            )
            if arrivals:
                s_time = p_time + arrivals[0].time
                return s_time
        except:
            pass
        return None
    
    def compute_sp_ratio(self, waveform, p_time, s_time):
        """Compute S/P amplitude ratio using window analysis."""
        # Trim waveform for P and S windows
        tr_p = waveform.copy().trim(p_time - 0.3, p_time + 2.5)
        tr_s = waveform.copy().trim(s_time - 0.3, s_time + 3.2)
        
        # Check if we have at least 3 channels
        if len(tr_p) < 3 or len(tr_s) < 3:
            logger.warning(f"Insufficient channels: P={len(tr_p)}, S={len(tr_s)}")
            return 0, 0, 0, 0
        
        # Calculate P amplitude (try vector resultant, fallback to max of components)
        try:
            amp_p = max(np.sqrt(tr_p[0].data**2 + tr_p[1].data**2 + tr_p[2].data**2))
        except Exception as e:
            logger.warning(f"Error computing P resultant: {e}. Using component max.")
            amp_p = max(max(abs(tr_p[0].data)), max(abs(tr_p[1].data)), max(abs(tr_p[2].data)))
        
        # Calculate S amplitude (max of 3 components)
        try:
            amp_s = max(max(abs(tr_s[0].data)), max(abs(tr_s[1].data)), max(abs(tr_s[2].data)))
        except Exception as e:
            logger.warning(f"Error computing S amplitude: {e}")
            return 0, 0, 0, 0
        
        if amp_p > 0 and amp_s > 0:
            sp_ratio = round(amp_s / amp_p, 6)
            log_sp_ratio = np.log10(sp_ratio)
            return float(amp_p), float(amp_s), sp_ratio, log_sp_ratio
        
        return 0, 0, 0, 0
    
    def process_event(self, xml_path, model_path):
        """Main processing pipeline."""
        # Parse XML
        df = self.parse_xml(xml_path)
        
        # Load model
        self.load_tf_model(model_path)
        
        # Download waveforms
        waveforms = self.download_waveforms(df)
        
        results = []
        
        # Group by station
        grouped = df.groupby(['network', 'station', 'location'])
        
        for (network, station, location), group in tqdm(grouped, desc="Processing stations"):
            station_key = f"{network}.{station}.{location}"
            waveform = waveforms.get(station_key)
            
            if waveform is None:
                logger.warning(f"No waveform for {station_key}")
                continue
            
            # Get P and S picks
            p_row = group[group['phase'] == 'P']
            s_row = group[group['phase'] == 'S']
            
            if p_row.empty:
                continue
            
            p_time = p_row.iloc[0]['arrival_time']
            s_time = s_row.iloc[0]['arrival_time'] if not s_row.empty else None
            
            if s_time is None:
                # Use theoretical S
                depth = p_row.iloc[0]['depth_km']
                distance = p_row.iloc[0]['distance_deg']
                s_time = self.theoretical_s_pick(p_time, distance, depth)
                theoretical_s_used = True
            else:
                theoretical_s_used = False
            
            if s_time is None:
                continue
            
            # Polarity prediction
            sample_rate = waveform[0].stats.sampling_rate
            polarity_input = self.prepare_polarity_input(waveform, p_time, sample_rate)
            polarity, polarity_score = self.predict_polarity(polarity_input)
            
            # S/P ratio
            p_amp, s_amp, sp_ratio, log_sp_ratio = self.compute_sp_ratio(
                waveform, p_time, s_time
            )
            
            result = {
                'event_id': p_row.iloc[0]['event_id'],
                'origin_time': p_row.iloc[0]['origin_time'],
                'station': station,
                'network': network,
                'location': location,
                'p_arrival': p_time,
                's_arrival': s_time,
                'distance_deg': p_row.iloc[0]['distance_deg'],
                'azimuth': p_row.iloc[0]['azimuth'],
                'polarity': polarity,
                'polarity_score': polarity_score,
                'p_amp': p_amp,
                's_amp': s_amp,
                'sp_ratio': sp_ratio,
                'log_sp_ratio': log_sp_ratio,
                'theoretical_s_used': theoretical_s_used,
                'waveform_start': waveform[0].stats.starttime,
                'waveform_end': waveform[0].stats.endtime
            }
            
            results.append(result)
        
        # Return results for further processing
        logger.info(f"Processing completed with {len(results)} results")
        return results
    
    def save_skhash_input(self, results, output_dir):
        """Save results in SKHASH format with separate amp and polarity files."""
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("No results to save")
            return
        
        # Get event ID from first result
        event_id = results_df.iloc[0]['event_id']
        
        # Create folder structure: event_id/IN and event_id/OUT
        event_dir = Path(output_dir) / event_id / "inafocmec"
        input_dir = event_dir / "IN"
        output_dir_path = event_dir / "OUT"
        
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare amplitude data (S/P ratio)
        amp_data = results_df[['event_id', 'station', 'network', 'location', 'p_arrival']].copy()
        amp_data = amp_data.rename(columns={'p_arrival': 'channel'})
        amp_data['channel'] = 'HNZ'
        amp_data.insert(5, 'sp_ratio', results_df['sp_ratio'].values)
        amp_data = amp_data[['event_id', 'station', 'network', 'location', 'channel', 'sp_ratio']]
        amp_data = amp_data[amp_data['sp_ratio'] > 0]  # Filter out zero or negative ratios
        
        # Prepare polarity data
        pol_data = results_df[['event_id', 'station', 'network', 'location']].copy()
        pol_data['channel'] = 'HNZ'
        # Convert polarity: U=1, D=-1, x=0
        pol_map = {'U': -1, 'D': 1, 'x': 0}
        pol_data['p_polarity'] = results_df['polarity'].map(pol_map)
        pol_data = pol_data[['event_id', 'station', 'network', 'location', 'channel', 'p_polarity']]
        
        # Prepare station data
        station_df = pd.DataFrame(self.stations_info)
        if len(station_df) > 0:
            station_df = station_df[['station', 'network', 'location', 'channel', 'latitude', 'longitude', 'elevation']]
        
        # Save station file
        station_file = input_dir / "station.csv"
        if len(station_df) > 0:
            station_df.to_csv(station_file, index=False, sep=',', lineterminator='\n')
            logger.info(f"Station data saved to {station_file}")
        else:
            logger.warning("No station metadata collected; station.csv is empty")
            station_df.to_csv(station_file, index=False, sep=',', lineterminator='\n')

        # Save analysis detail file (extended metadata)
        analysis_file = input_dir / "analysis.csv"
        results_df.to_csv(analysis_file, index=False, sep=',', lineterminator='\n')
        logger.info(f"Analysis details saved to {analysis_file}")

        # Save amp/pol files
        amp_file = input_dir / "amp.csv"
        pol_file = input_dir / "pol.csv"

        amp_data.to_csv(amp_file, index=False, sep=',', lineterminator='\n')
        pol_data.to_csv(pol_file, index=False, sep=',', lineterminator='\n')

        logger.info(f"Amplitude data saved to {amp_file}")
        logger.info(f"Polarity data saved to {pol_file}")

        return event_dir, amp_file, pol_file, station_file, analysis_file
    
    def generate_control_file(self, event_dir, amp_file, pol_file, station_file, config_path=None):
        """Generate SKHASH control file."""
        control_template = f"""## Control file for SKHASH - {Path(event_dir).name}

$input_format
SKHASH

$catfile
{event_dir}/IN/eq_catalog.csv

$fpfile
{pol_file}

$ampfile
{amp_file}

$stfile
{event_dir}/IN/station.csv

$outfile1
{event_dir}/OUT/out.csv

$outfile_pol_agree
{event_dir}/OUT/out_polagree.csv

$outfile_pol_info
{event_dir}/OUT/out_polinfo.csv

$outfolder_plots
{event_dir}/OUT

$vmodel_paths
velocity_models/iasp91.txt

$plot_acceptable_solutions
True

$npolmin
8

$min_polarity_weight
0.1

$nmc
30

$maxout
100

$ratmin
3

$badfrac
0.0

$delmax
700

$prob_max
0.6
"""
        
        control_file = Path(event_dir) / "control_file.txt"
        with open(control_file, 'w') as f:
            f.write(control_template)
        
        logger.info(f"Control file generated: {control_file}")
        return control_file


def main():
    parser = argparse.ArgumentParser(description='Polarity and S/P Ratio Analysis Pipeline for SKHASH')
    parser.add_argument('--xml', required=True, help='Path to SeisComP XML file')
    parser.add_argument('--model', required=True, help='Path to TensorFlow model (.h5)')
    parser.add_argument('--output-dir', required=True, help='Output directory for SKHASH input structure')
    parser.add_argument('--config', default='config.yaml', help='Configuration YAML file')
    
    args = parser.parse_args()
    
    pipeline = PolarityPipeline(args.config)
    
    # Process event and get results
    results = pipeline.process_event(args.xml, args.model)
    
    # Save in SKHASH format
    event_dir, amp_file, pol_file, station_file, analysis_file = pipeline.save_skhash_input(results, args.output_dir)
    
    # Generate control file
    control_file = pipeline.generate_control_file(event_dir, amp_file, pol_file, station_file)

    # Save summary JSON at pipeline level (event + per-station analysis)
    summary_json_path = event_dir / "OUT" / "analysis_summary.json"
    summary_data = {
        'event_id': results[0]['event_id'] if results else None,
        'event_origin_time': results[0]['origin_time'].isoformat() if results else None,
        'records': results
    }
    try:
        import json
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, default=str, indent=2)
        logger.info(f"Pipeline summary JSON saved to {summary_json_path}")
    except Exception as e:
        logger.warning(f"Could not save pipeline JSON summary: {e}")
    
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"Event directory: {event_dir}")
    logger.info(f"Control file: {control_file}")


if __name__ == '__main__':
    main()
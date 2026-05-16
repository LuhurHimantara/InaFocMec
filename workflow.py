#!/usr/bin/env python3
"""
SKHASH Polarity Pipeline - Complete Workflow Example

This script demonstrates the complete workflow:
1. Process event with polarity and S/P ratio analysis
2. Generate SKHASH input structure
3. Create required metadata files
4. Verify output
"""

import os
import sys
import json
from pathlib import Path
import argparse
import logging
import pandas as pd
from obspy import read_events
import warnings
from functions.beachball_generator import generate_beachball_image
warnings.filterwarnings("ignore", category=UserWarning, module='obspy')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SKHASHWorkflow:
    """Complete SKHASH workflow manager."""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
    
    def step1_process_event(self, xml_file, model_file, output_dir):
        """
        STEP 1: Run polarity and S/P analysis pipeline
        
        Output:
            - {event_id}/IN/amp.csv
            - {event_id}/IN/pol.csv
            - {event_id}/control_file.txt
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Process event with polarity pipeline")
        logger.info("=" * 60)
        
        cmd = (
            f"python polarity_tf_pipeline.py "
            f'--xml "{xml_file}" '
            f"--model {model_file} "
            f"--output-dir {output_dir} "
            f"--config {self.config_path}"
        )
        
        logger.info(f"Command: {cmd}")
        ret = os.system(cmd)
        
        if ret == 0:
            logger.info("✓ Pipeline completed successfully")
            return True
        else:
            logger.error(f"✗ Pipeline failed with code {ret}")
            return False
    
    def step2_extract_metadata(self, xml_file, event_id, output_dir):
        """
        STEP 2: Extract event catalog metadata
        
        Output:
            - {event_id}/IN/eq_catalog.csv
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Extract and create metadata files")
        logger.info("=" * 60)
        
        event_dir = Path(output_dir) / event_id / "inafocmec" / "IN"
        event_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract event info
        try:
            catalog = read_events(xml_file)
            event = catalog[0]
            origin = event.preferred_origin() or event.origins[0]
            
            time_str = origin.time.strftime("%Y-%m-%d %H:%M:%S.%f")
            
            eq_df = pd.DataFrame({
                'time': [time_str],
                'latitude': [origin.latitude],
                'longitude': [origin.longitude],
                'depth': [origin.depth / 1000 if origin.depth else 10.0],
                'horz_uncert_km': [2.0],
                'vert_uncert_km': [2.0],
                'mag': [event.magnitudes[0].mag if event.magnitudes else 0.0],
                'event_id': [event_id]
            })
            
            eq_file = event_dir / "eq_catalog.csv"
            eq_df.to_csv(eq_file, index=False)
            logger.info(f"✓ Created: {eq_file}")
            
        except Exception as e:
            logger.error(f"Failed to create eq_catalog: {e}")
            return False
        
        return True
    
    def step3_verify_structure(self, output_dir, event_id):
        """
        STEP 3: Verify SKHASH structure is complete
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Verify SKHASH structure")
        logger.info("=" * 60)
        
        event_dir = Path(output_dir) / event_id / "inafocmec"
        
        required_files = {
            'IN/amp.csv': 'Amplitude S/P ratio data',
            'IN/pol.csv': 'Polarity data',
            'IN/eq_catalog.csv': 'Event catalog',
            'IN/station.csv': 'Station list',
            'control_file.txt': 'SKHASH control file',
        }
        
        all_present = True
        for file_path, description in required_files.items():
            full_path = event_dir / file_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            logger.info(f"{status} {file_path:30} - {description}")
            if not exists:
                all_present = False
        
        if all_present:
            logger.info("\n✓ All files present! Ready for SKHASH.")
            return True
        else:
            logger.warning("\n⚠ Some files missing. Check manually created files.")
            return False
    
    def step4_show_summary(self, output_dir, event_id):
        """
        STEP 4: Show processing summary
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Processing Summary")
        logger.info("=" * 60)
        
        event_dir = Path(output_dir) / event_id / "inafocmec"
        
        # Read data summaries
        try:
            amp_df = pd.read_csv(event_dir / "IN" / "amp.csv")
            pol_df = pd.read_csv(event_dir / "IN" / "pol.csv")
            
            logger.info(f"\nAmplitude data: {len(amp_df)} stations")
            logger.info(f"  Min S/P ratio: {amp_df['sp_ratio'].min():.4f}")
            logger.info(f"  Max S/P ratio: {amp_df['sp_ratio'].max():.4f}")
            logger.info(f"  Mean S/P ratio: {amp_df['sp_ratio'].mean():.4f}")
            
            logger.info(f"\nPolarity data: {len(pol_df)} stations")
            pol_count = pol_df['p_polarity'].value_counts()
            logger.info(f"  Up (1): {pol_count.get(1, 0)}")
            logger.info(f"  Down (-1): {pol_count.get(-1, 0)}")
            logger.info(f"  Unknown (0): {pol_count.get(0, 0)}")
            
            logger.info(f"\nEvent directory: {event_dir}")
            logger.info(f"Control file: {event_dir / 'control_file.txt'}")
        
        except Exception as e:
            logger.error(f"Error reading summary: {e}")
            
    def step5_run_skhash(self, output_dir, event_id):
        """
        STEP 5: Run SKHASH focal mechanism analysis
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Running SKHASH focal mechanism analysis")
        logger.info("=" * 60)
        
        event_dir = Path(output_dir) / event_id / "inafocmec"
        
        # Change to event directory
        original_cwd = os.getcwd()
        try:
           
            # Run SKHASH
            cmd = f"python SKHASH.py {event_dir}/control_file.txt"
            logger.info(f"Running: {cmd}")
            
            ret = os.system(cmd)
            
            if ret == 0:
                logger.info("✓ SKHASH completed successfully")
                
                # Check if output files were created
                out_files = [
                    f"{event_dir}/OUT/out.csv",
                    f"{event_dir}/OUT/out_polagree.csv", 
                    f"{event_dir}/OUT/out_polinfo.csv"
                ]
                
                all_present = True
                for out_file in out_files:
                    if not Path(out_file).exists():
                        logger.warning(f"Expected output file not found: {out_file}")
                        all_present = False
                
                if all_present:
                    logger.info("✓ All SKHASH output files generated")
                else:
                    logger.warning("⚠ Some SKHASH output files may be missing")
                
                return True
            else:
                logger.error(f"✗ SKHASH failed with code {ret}")
                return False
                
        except Exception as e:
            logger.error(f"Error running SKHASH: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    def step6_export_json(self, output_dir, event_id):
        """
        STEP 6: Combine event/station/analysis/focal results into JSON
        """
        event_dir = Path(output_dir) / event_id / "inafocmec"
        base_in = event_dir / "IN"
        base_out = event_dir / "OUT"

        final_json = {
            'event': {},
            'stations': [],
            'analysis': [],
            'focal_mechanism': []
        }

        # event metadata from eq_catalog
        try:
            eq_catalog_file = base_in / "eq_catalog.csv"
            if eq_catalog_file.exists():
                eq_df = pd.read_csv(eq_catalog_file)
                if not eq_df.empty:
                    final_json['event'] = eq_df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Cannot read eq_catalog.csv: {e}")

        # station metadata
        try:
            station_file = base_in / "station.csv"
            if station_file.exists():
                station_df = pd.read_csv(station_file)
                final_json['stations'] = station_df.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Cannot read station.csv: {e}")

        # analysis data
        try:
            analysis_file = base_in / "analysis.csv"
            if analysis_file.exists():
                analysis_df = pd.read_csv(analysis_file)
                final_json['analysis'] = analysis_df.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Cannot read analysis.csv: {e}")

        # focal mechanism output
        try:
            out_file = base_out / "out.csv"
            if out_file.exists():
                out_df = pd.read_csv(out_file)
                strike = out_df['strike'].iloc[0]
                dip = out_df['dip'].iloc[0]
                rake = out_df['rake'].iloc[0]
                generate_beachball_image(strike, dip, rake, base_out/f"beachball_inafocmec.png")
                final_json['focal_mechanism'] = out_df.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Cannot read out.csv: {e}")

        # write final JSON file
        try:
            final_file = base_out / "final_result.json"
            with open(final_file, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, default=str)
            logger.info(f"Final JSON saved to {final_file}")
            return True
        except Exception as e:
            logger.error(f"Cannot write final JSON: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='SKHASH Workflow - Complete processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow
  python workflow.py --xml event.xml --model model.h5 --output-dir results
  
  # Only verify existing structure
  python workflow.py --verify --output-dir results --event-id 2025-05-27T00-55-05
        """
    )
    
    parser.add_argument('--xml', help='Path to SeisComP XML file')
    parser.add_argument('--model', default="polarity_model.h5", help='Path to TensorFlow model for Polarity (.h5)')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify structure')
    parser.add_argument('--event-id', help='Event ID for verification')
    
    args = parser.parse_args()
    
    workflow = SKHASHWorkflow(args.config)
    
    if args.verify_only:
        if not args.event_id:
            parser.error("--event-id required with --verify-only")
        workflow.step3_verify_structure(args.output_dir, args.event_id)
        return
    
    if not args.xml or not args.model:
        parser.error("--xml and --model are required")
    
    # Full workflow
    logger.info("Starting SKHASH workflow...")
    
    # Extract event ID
    try:
        catalog = read_events(args.xml)
        event = catalog[0]
        event_id = str(event.resource_id).split('/')[-1]
    except Exception as e:
        logger.error(f"Failed to read event ID: {e}")
        sys.exit(1)
    
    # Step 1: Process event
    if not workflow.step1_process_event(args.xml, args.model, args.output_dir):
        logger.error("Pipeline failed. Aborting.")
        sys.exit(1)
    
    # Step 2: Extract metadata
    if not workflow.step2_extract_metadata(args.xml, event_id, args.output_dir):
        logger.error("Metadata extraction failed. Aborting.")
        sys.exit(1)
    
    # Step 3: Verify
    workflow.step3_verify_structure(args.output_dir, event_id)
    
    # Step 4: Summary
    workflow.step4_show_summary(args.output_dir, event_id)
    
    # Step 5: Run SKHASH
    if not workflow.step5_run_skhash(args.output_dir, event_id):
        logger.error("SKHASH step failed. Aborting.")
        sys.exit(1)

    # Step 6: Export JSON summary
    if not workflow.step6_export_json(args.output_dir, event_id):
        logger.warning("Final JSON export failed")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Workflow completed! Final JSON available at OUT/final_result.json")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()

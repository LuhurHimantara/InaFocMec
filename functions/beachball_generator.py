#!/usr/bin/env python3
"""
Beachball Generator Pipeline for InaPolarityFocMec Dashboard
Generates focal mechanism beach ball diagram from earthquake data
Uses ObsPy library for accurate seismological visualization
"""

import json
import os
import sys

try:
    from obspy.imaging.beachball import beachball
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    print("Error: Required libraries not found.")
    print("Install with: pip install obspy matplotlib")
    sys.exit(1)


def generate_beachball_image(strike, dip, rake, output_file='img/beachball.png', size=200):
    """
    Generate beachball diagram from focal mechanism parameters
    
    Args:
        strike (float): Strike angle in degrees (0-360)
        dip (float): Dip angle in degrees (0-90)
        rake (float): Rake angle in degrees (-180 to 180)
        output_file (str): Output file path
        size (int): Size of the output image in pixels
    """
    # ObsPy beachball expects [strike, dip, rake] for one nodal plane
    focal_mechanism = [strike, dip, rake]
    
    # Create the beachball using ObsPy's beachball function
    fig = beachball(focal_mechanism, 
                    size=size, 
                    linewidth=2, 
                    facecolor='red',
                    bgcolor='white',
                    edgecolor='black',
                    alpha=1.0)
    
    # Save figure with transparent background
    plt.savefig(output_file, 
                dpi=100, 
                bbox_inches='tight', 
                transparent=True,
                pad_inches=0.05)
    plt.close(fig)
    
    return output_file


def generate_beachball_from_focmec(event_id, strike, dip, rake, suffix='', beachball_dir='data/beachball'):
    """
    Generate a single beachball diagram from focal mechanism parameters
    
    Args:
        event_id: Event identifier
        strike, dip, rake: Focal mechanism parameters
        suffix: Suffix for filename (e.g., '_skhash', '_quakelink')
        beachball_dir: Output directory
    
    Returns:
        str: Path to generated beachball
    """
    if not os.path.exists(beachball_dir):
        os.makedirs(beachball_dir)
    
    output_file = os.path.join(beachball_dir, f'{event_id}{suffix}.png')
    return generate_beachball_image(strike, dip, rake, output_file)


def run(earthquake_data, update_progress_callback=None):
    """
    Standard pipeline interface for beachball generation
    Generates beachballs for both QuakeLink and SKHASH focal mechanisms if available
    
    Args:
        earthquake_data (dict): Complete earthquake data with focal_mechanism and/or quakelink_fm
        update_progress_callback (callable, optional): Callback function for progress updates
    
    Returns:
        dict: {
            "success": bool,
            "result": {
                "skhash_beachball_path": str (if SKHASH FM exists),
                "quakelink_beachball_path": str (if QuakeLink FM exists),
                "event_id": str
            },
            "error": str or None
        }
    """
    try:
        event_id = earthquake_data.get('event', {}).get('event_id', 'unknown')
        beachball_dir = 'data/beachball'
        
        if update_progress_callback:
            update_progress_callback(10, "Preparing beachball generation...")
        
        result = {"event_id": event_id}
        beachballs_generated = 0
        
        # Check for SKHASH focal mechanism
        if 'focal_mechanism' in earthquake_data and earthquake_data['focal_mechanism']:
            fm = earthquake_data['focal_mechanism'][0]
            if all(k in fm for k in ['strike', 'dip', 'rake']):
                if update_progress_callback:
                    update_progress_callback(30, "Generating SKHASH beachball...")
                
                skhash_path = generate_beachball_from_focmec(
                    event_id, fm['strike'], fm['dip'], fm['rake'], 
                    '_skhash', beachball_dir
                )
                result['skhash_beachball_path'] = skhash_path
                beachballs_generated += 1
        
        # Check for QuakeLink focal mechanism (official FM)
        if 'quakelink_fm' in earthquake_data:
            fm = earthquake_data['quakelink_fm']
            if all(k in fm for k in ['strike', 'dip', 'rake']):
                if update_progress_callback:
                    update_progress_callback(60, "Generating QuakeLink beachball...")
                
                quakelink_path = generate_beachball_from_focmec(
                    event_id, fm['strike'], fm['dip'], fm['rake'],
                    '_quakelink', beachball_dir
                )
                result['quakelink_beachball_path'] = quakelink_path
                beachballs_generated += 1
        
        if beachballs_generated == 0:
            return {
                "success": False,
                "result": None,
                "error": "No focal mechanism data found (need skhash or quakelink focal mechanism)"
            }
        
        if update_progress_callback:
            update_progress_callback(90, "Creating latest symlink...")
        
        # Create latest.png symlink to SKHASH beachball (preferred) or QuakeLink
        latest_file = os.path.join(beachball_dir, 'latest.png')
        source_file = None
        if 'skhash_beachball_path' in result:
            source_file = f'{event_id}_skhash.png'
        elif 'quakelink_beachball_path' in result:
            source_file = f'{event_id}_quakelink.png'
        
        if source_file:
            try:
                if os.path.exists(latest_file):
                    os.remove(latest_file)
                os.symlink(source_file, latest_file)
            except:
                import shutil
                shutil.copy(os.path.join(beachball_dir, source_file), latest_file)
        
        if update_progress_callback:
            update_progress_callback(100, f"Generated {beachballs_generated} beachball(s)")
        
        return {
            "success": True,
            "result": result,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": f"Beachball generation failed: {str(e)}"
        }


def main():
    """Legacy command-line interface for backward compatibility"""
    
    # Check if latest.json exists in data/json/
    json_file = 'data/json/latest.json'
    if not os.path.exists(json_file):
        # Fallback to old location for backward compatibility
        json_file = 'final_result.json'
        if not os.path.exists(json_file):
            print(f"Error: No JSON file found!")
            print("Expected: data/json/latest.json or final_result.json")
            sys.exit(1)
        print(f"⚠ Using fallback location: {json_file}")
    
    # Read JSON data
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_file}")
        print(e)
        sys.exit(1)
    
    print("Generating focal mechanism beachball...")
    result = run(data)
    
    if result['success']:
        print(f"✓ Beachball generated: {result['result']['beachball_path']}")
        print(f"  Event ID: {result['result']['event_id']}")
        print(f"  Strike: {result['result']['focal_mechanism']['strike']}°")
        print(f"  Dip: {result['result']['focal_mechanism']['dip']}°")
        print(f"  Rake: {result['result']['focal_mechanism']['rake']}°")
        print("\nBeachball generation complete!")
        print("\nTo update the dashboard:")
        print("1. Refresh your browser to see the new beachball")
        print("2. Run this script after updating final_result.json")
    else:
        print(f"✗ Error: {result['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()

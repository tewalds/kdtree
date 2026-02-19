#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import sys
import os

def main():
    input_file = "benchmarks/continental-divide-national-scenic-trail.gpx"
    output_file = "benchmarks/cdt.tsv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)
        
    print(f"Parsing {input_file}...")
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # GPX namespaces
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    points = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = trkpt.get('lat')
        lon = trkpt.get('lon')
        if lat and lon:
            points.append(f"{lat} {lon}")
            
    print(f"Extracted {len(points)} points. Writing to {output_file}...")
    with open(output_file, 'w') as f:
        f.write('\n'.join(points) + '\n')
    print("Done.")

if __name__ == "__main__":
    main()

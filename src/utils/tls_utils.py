import xml.etree.ElementTree as ET
import subprocess
import csv

def get_default_durations(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()
    durations = []
    for tl_logic in root.findall('tlLogic'):
        for phase in tl_logic.findall('phase'):
            durations.append(int(phase.attrib['duration']))
    return durations

def evaluate_tls_settings(net_file, sumocfg_file, durations, tag="default"):
    # Write temp net file with given durations
    net_copy = f'eval_{tag}.net.xml'
    write_tls_durations(net_file, durations, net_copy)
    # Make a config file pointing to this net
    with open(sumocfg_file) as f:
        cfg_data = f.read()
    cfg_data = cfg_data.replace('grid_final.net.xml', net_copy)
    cfg_copy = f'eval_{tag}.sumocfg'
    with open(cfg_copy, 'w') as f:
        f.write(cfg_data)
    tripinfo_file = f'eval_{tag}_tripinfo.xml'
    subprocess.run([
        'sumo', '-c', cfg_copy,
        '--tripinfo-output', tripinfo_file,
        '--no-warnings', 'true'
    ], check=True)
    total_time = parse_tripinfo(tripinfo_file)
    return total_time

def export_tls_settings(net_file, durations, output_csv):
    tree = ET.parse(net_file)
    root = tree.getroot()
    rows = []
    idx = 0
    for tl_logic in root.findall('tlLogic'):
        tls_id = tl_logic.attrib['id']
        for phase_idx, phase in enumerate(tl_logic.findall('phase')):
            if idx < len(durations):
                rows.append({
                    'tls_id': tls_id,
                    'phase_index': phase_idx,
                    'optimized_duration': durations[idx]
                })
            idx += 1
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['tls_id', 'phase_index', 'optimized_duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Exported final TLS settings to {output_csv}")

def write_tls_durations(input_xml, durations, output_file):
    tree = ET.parse(input_xml)
    root = tree.getroot()
    idx = 0
    for tl_logic in root.findall('tlLogic'):
        for phase in tl_logic.findall('phase'):
            phase.set('duration', str(durations[idx]))
            idx += 1
    tree.write(output_file)

def parse_tripinfo(tripinfo_file):
    # Returns total travel time for all vehicles
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        total_time = 0
        for trip in root.findall('tripinfo'):
            total_time += float(trip.get('duration'))
        return total_time
    except Exception as e:
        print(f"Error parsing {tripinfo_file}: {e}")
        return float('inf')
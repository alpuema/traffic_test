import os
import sys
import subprocess
import tempfile
import logging

def generate_grid_net(grid_size, final_net_file):
    import sumolib
    with tempfile.NamedTemporaryFile(delete=False, suffix='.net.xml') as temp_net:
        temp_net_name = temp_net.name

    subprocess.run([
        "netgenerate",
        "--grid",
        "--grid.number", str(grid_size),
        "--output-file", temp_net_name
    ], check=True)

    # Read junction IDs
    net = sumolib.net.readNet(temp_net_name)
    junction_ids = [junction.getID() for junction in net.getNodes()]

    subprocess.run([
        "netconvert",
        "--sumo-net-file", temp_net_name,
        "--tls.set", ",".join(junction_ids),
        "--tls.default-type", "static",
        "--tls.green.time", "40",
        "--tls.yellow.time", "3",
        "--tls.red.time", "2",
        "--output-file", final_net_file
    ], check=True)

    os.remove(temp_net_name)

def generate_vtypes(vtype_file):
    with open(vtype_file, "w") as f:
        f.write("""<additional>
    <vType id="car" vClass="passenger" speedDev="0.1" sigma="0.5"/>
    <vType id="truck" vClass="truck" speedDev="0.05" sigma="0.5"/>
</additional>
""")

def generate_routes(final_net_file, route_file, vtype_file):
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        sys.exit("Please set the SUMO_HOME environment variable.")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    subprocess.run([
        sys.executable, random_trips,
        "-n", final_net_file,
        "-o", route_file,
        "-e", "50",
        "--seed", "42",
        "--additional-files", vtype_file
    ], check=True)

def generate_sumocfg(final_net_file, route_file, vtype_file, sumocfg_file, sim_end):
    with open(sumocfg_file, "w") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="{final_net_file}"/>
        <route-files value="{route_file}"/>
        <additional-files value="{vtype_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_end}"/>
    </time>
</configuration>
""")

def run_sumo_simulation(sumocfg_file):
    import traci
    import sumolib
    logging.info("Start simulation...")
    sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumocfg_file])

    tls_ids = traci.trafficlight.getIDList()
    logging.info(f"Traffic lights active: {len(tls_ids)}")

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

    logging.info(f"Simulation ended after {step} steps.")
    traci.close()

def prepare_sumo_scenario(
    grid_size=3,
    sim_end=10000,
    final_net="grid_final.net.xml",
    route_file="grid.rou.xml",
    vtype_file="vtype.add.xml",
    sumocfg_file="grid.sumocfg",
    run_simulation=False
):
    """
    Erases old scenario files (if present) and generates new SUMO scenario files.
    Optionally runs the simulation if run_simulation=True.
    """
    logging.basicConfig(level=logging.INFO)
    # Remove old files
    for fname in [final_net, route_file, vtype_file, sumocfg_file]:
        try:
            if os.path.exists(fname):
                os.remove(fname)
                logging.info(f"Removed old file: {fname}")
        except Exception as e:
            logging.warning(f"Could not remove {fname}: {e}")

    try:
        logging.info("Generating grid network with traffic lights...")
        generate_grid_net(grid_size, final_net)

        logging.info("Generating vehicle types...")
        generate_vtypes(vtype_file)

        logging.info("Generating random routes...")
        generate_routes(final_net, route_file, vtype_file)

        logging.info("Generating SUMO config file...")
        generate_sumocfg(final_net, route_file, vtype_file, sumocfg_file, sim_end)

        logging.info("Scenario generation complete.")
        if run_simulation:
            run_sumo_simulation(sumocfg_file)
        else:
            logging.info("Simulation not run. Set run_simulation=True to run SUMO after generation.")
    except Exception as e:
        logging.error(f"Error during scenario generation: {e}")

# Example usage:
# prepare_sumo_scenario(grid_size=4, sim_end=5000, run_simulation=False)

if __name__ == "__main__":
    prepare_sumo_scenario(grid_size=6, sim_end=5000, run_simulation=False)
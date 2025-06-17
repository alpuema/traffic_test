import os
import sys
import subprocess
import argparse
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

def main():
    import traci
    import sumolib

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--sim-end", type=int, default=100)
    parser.add_argument("--final-net", default="grid_final.net.xml")
    parser.add_argument("--route-file", default="grid.rou.xml")
    parser.add_argument("--vtype-file", default="vtype.add.xml")
    parser.add_argument("--sumocfg-file", default="grid.sumocfg")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        logging.info("Generate grid network with traffic lights...")
        generate_grid_net(args.grid_size, args.final_net)

        logging.info("Generate vehicle types...")
        generate_vtypes(args.vtype_file)

        logging.info("Generate random routes...")
        generate_routes(args.final_net, args.route_file, args.vtype_file)

        logging.info("Generate SUMO config file...")
        generate_sumocfg(args.final_net, args.route_file, args.vtype_file, args.sumocfg_file, args.sim_end)

        logging.info("Start simulation...")
        sumoBinary = sumolib.checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", args.sumocfg_file])

        tls_ids = traci.trafficlight.getIDList()
        logging.info(f"Traffic lights active: {len(tls_ids)}")

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

        logging.info(f"Simulation ended after {step} steps.")
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user.")
    finally:
        try:
            traci.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
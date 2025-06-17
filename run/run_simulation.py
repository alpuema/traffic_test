import os
import sys
import subprocess
import traci
import sumolib
import time

net_file = "grid.net.xml"
route_file = "grid.rou.xml"
sumocfg_file = "grid.sumocfg"

def generate_grid_net():
    subprocess.run([
        "netgenerate",
        "--grid",
        "--grid.number=3",
        "--output-file=" + net_file
    ], check=True)

def generate_grid_routes():
    sumo_home = os.environ["SUMO_HOME"]
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    subprocess.run([
        sys.executable, random_trips,
        "-n", net_file,
        "-o", route_file,
        "-e", "50",
        "--seed", "42",
        "--additional-files", "vtype.add.xml"
    ], check=True)

def generate_sumocfg():
    with open(sumocfg_file, "w") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
        <additional-files value="vtype.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="10000"/>
    </time>
</configuration>
""")

def main():
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please set the SUMO_HOME environment variable.")

    print("Generating grid network...")
    generate_grid_net()
    print("Generating routes...")
    generate_grid_routes()
    print("Generating sumocfg...")
    generate_sumocfg()

    sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumocfg_file])

    # Programmatic traffic light control starts here
    tls_ids = traci.trafficlight.getIDList()
    print("Traffic light IDs:", tls_ids)

    print("Running simulation until all vehicles have arrived...")
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        # Example: Change phase every 20 steps
        for tls_id in tls_ids:
            current_phase = traci.trafficlight.getPhase(tls_id)
            num_phases = traci.trafficlight.getPhaseNumber(tls_id)
            if step % 20 == 0:
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
        traci.simulationStep()
        step += 1
        time.sleep(0.1)
    print(f"Simulation complete in {step} steps.")

    traci.close()
    print("All vehicles have arrived. Simulation done.")

if __name__ == "__main__":
    main()
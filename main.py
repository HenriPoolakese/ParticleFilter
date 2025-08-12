import itertools
import time

from matplotlib import pyplot as plt

from config import CACHE_FILE, robot_step_distance_global, USE_ROAD_NUMBERS, particle_count
from utils.DataLogger import DataLogger
from utils.RoadNetworkManager import load_or_cache_road_data, cache_file

from utils.other import *
from robot.particle import Particle
from robot.robot import Robot






def main():
    print(CACHE_FILE)
    start_time = time.time()

    # Load and enhance the road graph
    road_graph = load_or_cache_road_data()

    # Convert nodes iterator to list to get proper length
    original_nodes = list(road_graph.nodes())
    print(f"Original nodes: {len(original_nodes)}")

    enhanced_graph = enhance_road_graph(road_graph)

    distance_cache = precompute_distances(enhanced_graph, CITIES)

    enhanced_nodes = list(enhanced_graph.nodes())
    print(f"Enhanced nodes: {len(enhanced_nodes)}")
    print(f"Added {len(enhanced_nodes) - len(original_nodes)} intermediate nodes")

    # Verify some enhanced edges
    for u, v, data in list(enhanced_graph.edges(data=True))[:5]:  # Convert to list for slicing
        if '-' in str(u) or '-' in str(v):  # Check for intermediate nodes
            print(f"Enhanced edge between {u} and {v}")

    # Initialize data logger
    data_logger = DataLogger()
    print(f"Simulation ID: {data_logger.simulation_id}")

    # Initialize robot
    robot = Robot("Tallinn", "Tartu", enhanced_graph, data_logger,
                  movement_step_size=robot_step_distance_global)

    # Initialize visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top=0.85)

    # Plot road network
    for u, v, data in enhanced_graph.edges(data=True):
        if 'geometry' in data:
            line = data['geometry']
            xs, ys = line.xy
            ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.7)
        else:
            u_x, u_y = enhanced_graph.nodes[u]['x'], enhanced_graph.nodes[u]['y']
            v_x, v_y = enhanced_graph.nodes[v]['x'], enhanced_graph.nodes[v]['y']
            ax.plot([u_x, v_x], [u_y, v_y], color='gray', linewidth=0.5, alpha=0.7)

    # Plot cities
    for city, (lat, lon) in CITIES.items():
        ax.plot(lon, lat, 'ro', markersize=5)
        ax.text(lon, lat, city, fontsize=9, color='red')

    # Initialize particles
    particles_list = initialize_particles(enhanced_graph, particle_count,distance_cache)

    # Initialize visualization elements
    robot_scatter, = ax.plot([], [], 'bo', markersize=8, label='Robot')
    particle_scatters = []
    for particle in particles_list:
        scatter = ax.plot([], [], color='green', marker='o', markersize=4, alpha=0.5)[0]
        particle_scatters.append(scatter)

    # Initialize other UI elements
    constraint_markers = []
    constraint_texts = []
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment='top')
    pos_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, verticalalignment='top')

    constraints_text = ax.text(0.65, 0.98, "Active constraints:\nNone",
                               transform=ax.transAxes, verticalalignment='top',
                               fontsize=10, bbox=dict(facecolor='white', alpha=0.8,
                                                      edgecolor='black', boxstyle='round,pad=0.5'))

    ax.legend(loc='upper right')
    plt.ion()


    # Main simulation loop
    for u, v, data in enhanced_graph.edges(data=True):
        print(data.get('length', 'No length attribute'))


    SNAPSHOT_INTERVAL = 20
    road_num_count = 0

    for step in range(len(robot.full_path)):

        print("constraint--------------- " + str(robot.active_constraints))

        print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")

        robot.move()
        # Log robot state after movement
        data_logger.log_robot_state(robot, step)
        print(f"\nRobot : {time.time() - start_time:.2f} seconds")

        # Movement
        any_stuck = False
        for particle in particles_list:
            particle.move()

        print(f"\nParticle : {time.time() - start_time:.2f} seconds")


        # Create snapshot of experiment
        if (step % SNAPSHOT_INTERVAL == 0 and SNAPSHOT_INTERVAL > 0) or step == 1:
            data_logger.save_simulation_snapshot(fig, step)


        # Log particles
        data_logger.log_particles(particles_list, step)


        data_logger.log_particles(particles_list, step)
        data_logger.log_constraints(robot.active_constraints, step)

        # When constraints are updated
        if robot.active_constraints or (robot.current_road_number and USE_ROAD_NUMBERS != False):
            road_num_count += 1
            for particle in particles_list:
                particle.needs_weight_update = True


            if road_num_count > 4:
                robot.current_road_number = None
                road_num_count = 0

            data_logger.log_constraints(robot.active_constraints, step)

        print("Particle weights")

        print([p.weight for p in particles_list])

        # Needs weight update
        if True:
            for particle in particles_list:
                # Only update if particle moved or constraints changed
                if particle.needs_weight_update:
                    print("updated weights")
                    particle.update_weight(
                        robot.active_constraints,
                        robot.current_road_number,
                        robot
                    )
                    #particle.weight = particle.weight / len(particles_list)
                    particle.needs_weight_update = False

            """if sum(p.weight for p in particles_list) < 1e-4:
                print(sum(p.weight for p in particles_list))
                print("xxx")
                scatter_particles_randomly(particles_list)
                break"""

            # Normalize weights
            normalize_weights(particles_list)


        print("Particle weights after weighing")

        weights = [p.weight for p in particles_list]
        print("Weights:", weights)
        print("Sum:", sum(weights))
        print(f"\nWeight : {time.time() - start_time:.2f} seconds")

        # Before resampling:
        good_particles = [p for p in particles_list
                          if geodesic(p.position, robot.position).km < 7.0]
        print(f"Good particles before resample: {len(good_particles)}")

        # 3. Conditional Resampling
        needs_resample = (
                robot.active_constraints or  # Force resample on new constraints
                any_stuck or (robot.current_road_number and USE_ROAD_NUMBERS != False ) # Particles need redistribution

        )
        if needs_resample:

            particles_list = resample_particles(particles_list)
            normalize_weights(particles_list)
            data_logger.log_resampling_event()
            if robot.active_constraints:
                robot.active_constraints = {}  # Clear after processing

        print(f"\nresample : {time.time() - start_time:.2f} seconds")

        # After resampling:
        good_particles = [p for p in particles_list if geodesic(p.position, robot.position).km < 7.0]
        print(f"Good particles after resample: {len(good_particles)}")
        weights = [p.weight for p in particles_list]
        print("Weights:", weights)
        print("Sum:", sum(weights))
        # Update visualization
        for i, particle in enumerate(particles_list):
            try:
                lat, lon = particle.get_position()
                particle_scatters[i].set_data([lon], [lat])
                particle_scatters[i].set_alpha(particle.opacity)
                particle_scatters[i].set_color(particle.color)
            except Exception as e:
                print(f"Error plotting particle: {e}")
        robot.current_milestone = None

        # Update constraint visualization
        for marker in constraint_markers:
            marker.remove()
        for text in constraint_texts:
            text.remove()
        constraint_markers = []
        constraint_texts = []

        # Add new constraint markers
        for city, dist in robot.display_constraints.items():
            if city in CITIES:
                lat, lon = CITIES[city]
                marker = ax.plot(lon, lat, 's', color='purple', markersize=10,
                                 markeredgewidth=2, markeredgecolor='black')[0]
                text = ax.text(lon, lat + 0.01, f"{dist}km",
                               fontsize=9, color='purple', ha='center')
                constraint_markers.append(marker)
                constraint_texts.append(text)

        # Update robot position
        robot_scatter.set_data([robot.position[1]], [robot.position[0]])

        # Update text displays
        step_text.set_text(f"Step: {step + 1}")
        pos_text.set_text(f"Position: Lat {robot.position[0]:.6f}, Lon {robot.position[1]:.6f}")

        # Build constraints display
        constraint_lines = ["Active constraints:"]
        if robot.display_constraints:
            sorted_constraints = sorted(robot.display_constraints.items(),
                                        key=lambda x: x[1])
            constraint_lines.extend([f"  • {city}: {dist}±6km"
                                     for city, dist in sorted_constraints])
            color = 'red' if len(robot.display_constraints) > 1 else 'black'
            constraints_text.set_color(color)
        else:
            constraint_lines.append("  None")
        constraints_text.set_text("\n".join(constraint_lines))

        plt.pause(0.01)

        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()

    # Save all logged data to file
    data_logger.save_to_file()
    print(f"Data saved to simulation_{data_logger.simulation_id}")
    # tool2.SimulationAnalyzer.analyze_all(data_logger.simulation_id)




if __name__ == "__main__":
    main()
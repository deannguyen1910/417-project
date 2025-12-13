#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from independent import IndependentSolver
from prioritized import PrioritizedPlanningSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost
from policy_guidance import load_policy
from policy_grid import load_policy_grid
import os

SOLVER = "CBS"

def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)


def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--disjoint', action='store_true', default=False,
                        help='Use the disjoint splitting')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Prioritized}), defaults to ' + str(SOLVER))
    parser.add_argument('--policy-checkpoint', type=str, default=None,
                        help='Path to a trained policy checkpoint to guide CBS successor ordering')
    parser.add_argument('--policy-grid', type=str, default=None,
                        help='Path to a grid of per-cell policy scores (CSV or space-separated) to bias heuristics')
    parser.add_argument('--policy-grid-weight', type=float, default=0.0,
                        help='Weight applied to the policy grid when augmenting the heuristic (higher favors high-scored cells)')
    parser.add_argument('--policy-grid-apply-to-h', action='store_true', default=False,
                        help='If set, subtract policy_weight * grid[x][y] from the heuristic (clamped at 0). If false, grid does not change h.')
    parser.add_argument('--auto-policy-grid', action='store_true', default=False,
                        help='If set, automatically look for checkpoints/<instance_basename>_policy_grid.csv per instance.')

    args = parser.parse_args()


    result_file = open("results.csv", "w", buffering=1)

    for file in sorted(glob.glob(args.instance)):

        print("***Import an instance***")
        my_map, starts, goals = import_mapf_instance(file)
        print_mapf_instance(my_map, starts, goals)

        policy_guidance = None
        policy_grid = None
        policy_weight = args.policy_grid_weight if args.policy_grid_weight is not None else 0.0
        policy_apply_to_h = args.policy_grid_apply_to_h

        # Auto-pick policy grid if requested
        if args.auto_policy_grid and args.policy_grid is None:
            base = os.path.basename(file)
            base_no_ext = os.path.splitext(base)[0]
            candidate = os.path.join("checkpoints", f"{base_no_ext}_policy_grid.csv")
            if os.path.isfile(candidate):
                policy_grid = load_policy_grid(candidate)
                print(f"Auto-loaded policy grid: {candidate}")
            else:
                print(f"No auto policy grid found for {file}, expected at {candidate}")
        if args.policy_checkpoint is not None and args.solver == "CBS":
            try:
                policy_guidance = load_policy(args.policy_checkpoint, my_map)
                print(f"Loaded policy checkpoint: {args.policy_checkpoint}")
            except Exception as e:
                print(f"Warning: failed to load policy checkpoint '{args.policy_checkpoint}': {e}")
        if args.policy_grid is not None and args.solver == "CBS":
            try:
                policy_grid = load_policy_grid(args.policy_grid)
                print(f"Loaded policy grid: {args.policy_grid}")
            except Exception as e:
                print(f"Warning: failed to load policy grid '{args.policy_grid}': {e}")

        if args.solver == "CBS":
            print("***Run CBS***")
            cbs = CBSSolver(my_map, starts, goals, policy_guidance=policy_guidance,
                            policy_grid=policy_grid, policy_weight=policy_weight,
                            policy_apply_to_h=policy_apply_to_h)
            paths = cbs.find_solution(args.disjoint)
        elif args.solver == "Independent":
            print("***Run Independent***")
            solver = IndependentSolver(my_map, starts, goals)
            paths = solver.find_solution()
        elif args.solver == "Prioritized":
            print("***Run Prioritized***")
            solver = PrioritizedPlanningSolver(my_map, starts, goals)
            paths = solver.find_solution()
        else:
            raise RuntimeError("Unknown solver!")

        cost = get_sum_of_cost(paths)
        result_file.write("{},{}\n".format(file, cost))


        if not args.batch:
            print("***Test paths on a simulation***")
            animation = Animation(my_map, starts, goals, paths)
            # animation.save("output.mp4", 1.0)
            animation.show()
    result_file.close()

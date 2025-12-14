#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from independent import IndependentSolver
from prioritized import PrioritizedPlanningSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost, reset_low_level_counters, get_low_level_counters, set_run_override, set_node_limit, NodeLimitExceeded
import time as timer
import multiprocessing as mp
import traceback
import sys


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

def solver(queue, my_map, starts, goals, solver_name, disjoint, mode, partial_k, lookahead_depth, node_limit):
    try:
        # set override and node limit in this child process so a_star sees them
        try:
            set_run_override(mode, partial_k, lookahead_depth)
        except Exception:
            pass
        try:
            set_node_limit(node_limit)
        except Exception:
            pass

        start = timer.time()
        if solver_name == "CBS":
            solver_obj = CBSSolver(my_map, starts, goals)
            paths = solver_obj.find_solution(disjoint)
        elif solver_name == "Independent":
            solver_obj = IndependentSolver(my_map, starts, goals)
            paths = solver_obj.find_solution()
        elif solver_name == "Prioritized":
            solver_obj = PrioritizedPlanningSolver(my_map, starts, goals)
            paths = solver_obj.find_solution()
        else:
            queue.put(('ERROR', f'Unknown solver: {solver_name}'))
            return

        elapsed = timer.time() - start
        low = get_low_level_counters()
        high_generated = getattr(solver_obj, 'num_of_generated', '')
        high_expanded = getattr(solver_obj, 'num_of_expanded', '')

        queue.put(('OK', paths, high_expanded, high_generated, low['expanded'], low['generated'], elapsed))
    except NodeLimitExceeded as ne:
        elapsed = timer.time() - start if 'start' in locals() else 0.0
        queue.put(('NODE_LIMIT', ne.expanded, ne.generated, elapsed))
    except Exception as e:
        tb = traceback.format_exc()
        queue.put(('EXC', str(e), tb))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--disjoint', action='store_true', default=False,
                        help='Use the disjoint splitting')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Prioritized})')
    parser.add_argument('--mode', type=str, default=None,
                        help='low-level mode: astar | partial | lookahead (if omitted, baseline astar)')
    parser.add_argument('--partial_k', type=int, default=None,
                        help='(for partial) number of children to keep per expansion')
    parser.add_argument('--lookahead_depth', type=int, default=None,
                        help='(for lookahead) depth of greedy/bounded lookahead simulation')
    parser.add_argument('--run_all_modes', action='store_true',
                        help='If set, run baseline, partial, lookahead sequentially for each instance')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout per run in seconds (kills solver if exceeded)')
    parser.add_argument('--node_limit', type=int, default=None,
                    help='Stop a run if low-level node expansions exceed this integer (per run), then write -1 to CSV row')

    args = parser.parse_args()


    result_file = open("results.csv", "w", buffering=1)
    result_file.write("instance,solver,mode,partial_k,lookahead_depth,cost,high_expanded,high_generated,low_expanded,low_generated,cpu_time_or_status\n")

    files = sorted(glob.glob(args.instance))
    if not files:
        print("No instances matched pattern:", args.instance)
        sys.exit(1)

    for filepath in files:
        print("***Import an instance***", filepath)
        my_map, starts, goals = import_mapf_instance(filepath)
        print_mapf_instance(my_map, starts, goals)

        # decide which modes to run
        if args.run_all_modes:
            modes_to_run = [
                ('astar', '', ''),  # baseline
                ('partial', args.partial_k if args.partial_k is not None else 2, ''),  # partial_k default 2
                ('lookahead', '', args.lookahead_depth if args.lookahead_depth is not None else 2)  # lookahead_depth default 2
            ]
        else:
            chosen_mode = args.mode if args.mode is not None else 'astar'
            chosen_partial = args.partial_k if args.partial_k is not None else ''
            chosen_lookahead = args.lookahead_depth if args.lookahead_depth is not None else ''
            modes_to_run = [(chosen_mode, chosen_partial, chosen_lookahead)]

        for (mode, partial_k, lookahead_depth) in modes_to_run:
            print(f"Running {args.solver} on {Path(filepath).name} mode={mode} partial_k={partial_k} lookahead_depth={lookahead_depth}")
            reset_low_level_counters()

            # run solver in child process w/ timeout
            q = mp.Queue()
            p = mp.Process(target=solver, args=(q, my_map, starts, goals, args.solver, args.disjoint, mode, partial_k, lookahead_depth, args.node_limit))
            p.start()
            p.join(args.timeout)
            if p.is_alive():
                print(f"Timeout reached ({args.timeout}s). Terminating process.")
                p.terminate()
                p.join()
                # record timeout row
                result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    filepath, args.solver, mode, partial_k if partial_k != '' else '',
                    lookahead_depth if lookahead_depth != '' else '',
                    -1, 'TIMEOUT', 'TIMEOUT', 'TIMEOUT', 'TIMEOUT', 'TIMEOUT'))
            else:
                # process finished, pull queue result
                if q.empty():
                    # unexpected: no message
                    result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        filepath, args.solver, mode, partial_k if partial_k != '' else '',
                        lookahead_depth if lookahead_depth != '' else '',
                        -1, 'NORESULT', 'NORESULT', 'NORESULT', 'NORESULT', 'NORESULT'))
                else:
                    msg = q.get()
                    if msg[0] == 'OK':
                        _, paths, high_expanded, high_generated, low_expanded, low_generated, elapsed = msg
                        cost = get_sum_of_cost(paths)
                        result_file.write("{},{},{},{},{},{},{},{},{},{},{:.6f}\n".format(
                            filepath, args.solver, mode, partial_k if partial_k != '' else '',
                            lookahead_depth if lookahead_depth != '' else '',
                            cost, high_expanded, high_generated, low_expanded, low_generated, elapsed))
                    elif msg[0] == 'NODE_LIMIT':
                        _, low_expanded, low_generated, elapsed = msg
                        print(f"Node limit hit: expanded={low_expanded} generated={low_generated}")
                        result_file.write("{},{},{},{},{},{},{},{},{},{},{:.6f}\n".format(
                        filepath, args.solver, mode, partial_k if partial_k != '' else '',
                        lookahead_depth if lookahead_depth != '' else '',
                        -1, 'NODE_LIMIT', '', low_expanded, low_generated, elapsed))
                    elif msg[0] == 'EXC':
                        _, err_str, tb = msg
                        print("Solver raised an exception:\n", err_str)
                        print(tb)
                        result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            filepath, args.solver, mode, partial_k if partial_k != '' else '',
                            lookahead_depth if lookahead_depth != '' else '',
                            -1, 'EXC', '', '', '', 'EXC'))
                    else:
                        # unknown message type
                        result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            filepath, args.solver, mode, partial_k if partial_k != '' else '',
                            lookahead_depth if lookahead_depth != '' else '',
                            -1, 'ERR', '', '', '', 'ERR'))

        if not args.batch:
            try:
                animation = Animation(my_map, starts, goals, paths)
                animation.show()
            except Exception:
                pass
    
    
    result_file.close()

import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    # print ("heheh", path1)
    if get_location(path1, 0) == get_location(path2, 0): 
        return [[get_location(path1, 0)], 0]

    max_timestep = max(len(path1), len(path2))
    for time in range(1, max_timestep):
        a_curr = get_location(path1, time - 1)
        a_next = get_location(path1, time)
        b_curr = get_location(path2, time - 1)
        b_next = get_location(path2, time)
        if (a_next == b_next):
            return [[a_next], time]
        if ([a_curr, a_next] == [b_next, b_curr]):
            return [[a_curr, a_next], time]

    return None

    pass


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    # print("pathsssss", paths)   
    # print (len(paths))
    collisions = []
    agent_t = len(paths)
    for i in range(agent_t):
        for j in range(i + 1, agent_t):
            collision = detect_collision(paths[i], paths[j])
            # print("hehe", collision)
            if collision is not None:          
                collisions.append({
                    'a1': i, 
                    'a2': j,
                    'loc': collision[0],      
                    'timestep': collision[1]   
                })
            
    return collisions
    # return [{'a1': 0, 'a2': 1, 'loc': [(1, 4)], 'timestep': 3}]
    pass


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    constraints = []
    if len(collision['loc']) == 1: 
        constraints = [
            {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']},
            {'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep']},
        ]
    else:  
        constraints = [
            {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']},
            {'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep']},
        ]
    return constraints
    pass


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    pass


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        # print(root['collisions'])

        # Task 3.2: Testing
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        
        
        # root['cost'] = get_sum_of_cost(root['paths'])
        # root['collisions'] = detect_collisions(root['paths'])
        # self.push_node(root)
        while (len(self.open_list) > 0):
            P = self.pop_node()
            # if len(P['collisions']) == 0:
            #     self.print_results(P)
            #     return P['paths']

            if not P['collisions']:
                self.print_results(P)
                return P['paths']

            collision = P['collisions'][0]
            constraints = standard_splitting(collision) 

            for constraint in constraints:
                new_constraints = [c.copy() for c in P['constraints']]
                new_constraints.append(constraint.copy())

                new_paths = [p[:] for p in P['paths']]

                ai = constraint['agent']
                new_path = a_star(self.my_map, self.starts[ai], self.goals[ai],
                                self.heuristics[ai], ai, new_constraints)
                if new_path is None:
                    continue

                new_paths[ai] = new_path

                Q = {
                    'cost': get_sum_of_cost(new_paths),
                    'constraints': new_constraints,
                    'paths': new_paths,
                    'collisions': detect_collisions(new_paths)
                }
                self.push_node(Q)


        self.print_results(root)
        return root['paths']

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

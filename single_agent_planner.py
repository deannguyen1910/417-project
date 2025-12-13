import heapq

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    # agent int 
    # constraint 
        # {'agent': 2,
        # 'loc': [(3,4)],
        # 'timestep': 5}
    # †able: {timestep: [constraint]}
    table = dict()
    for constraint in constraints:
        if constraint['agent'] == agent:
            timestep = constraint['timestep']
            table.setdefault(timestep, []).append(constraint)
    
    return table
    pass    


def build_constraint_table_disjoint(constraints, agent):
    ##############################
    # Task 4 
    table = dict()
    # print(constraints)
    # remember, for the case for example, agent a, (u, v), t, pos
    # we would think of 
    # agent b, (v, u), t, neg
    # agent b, u, t, neg
    # agent b, v, t - 1, neg
    # but, we dont need those negative vertex constraint, because they should alwasy go together
    # agent a, (u, v), t, pos,
    # agent a, u, t, pos
    # agent a, v, t - 1, pos
    for constraint in constraints:
        timestep = constraint['timestep']
        loc = constraint['loc']
        pos = constraint.get('positive', False)
        agent_c = constraint['agent']

        # vertex
        if len(loc) == 1:
            if pos: # if this is positive constraint 
                if agent_c == agent: # add for its own agent with positive constraint
                    table.setdefault(timestep, []).append({'agent': agent, 'loc': loc, 'timestep': timestep, 'positive': True})
                else: # if there is another positive constraitn for another agent, the this agent should not step onto this loc, a negative constraint
                    table.setdefault(timestep, []).append({'agent': agent, 'loc': loc, 'timestep': timestep, 'positive': False})
            else: # if this is a negative, do same as normal 
                if agent_c == agent:
                    table.setdefault(timestep, []).append({'agent': agent, 'loc': loc, 'timestep': timestep, 'positive': False})
        # edge 
        else: 
            if pos: # if this is a positive constraint
                if agent_c == agent: # add for its own agent with positive edge contraint
                    table.setdefault(timestep, []).append({'agent': agent, 'loc': loc, 'timestep': timestep, 'positive': True})
                else: # if there is another positive constraint of other agent, this agent should not have a negative edge, negative constraint with negative edge.
                    u, v = loc[0], loc[1]
                    table.setdefault(timestep, []).append({'agent': agent, 'loc': [v, u], 'timestep': timestep, 'positive': False})
                    
                    # just in case. At this point, i'm lost :) 
                    # table.setdefault(timestep, []).append({'agent': agent, 'loc': [v], 'timestep': timestep, 'positive': False})
                    # if timestep - 1 >= 0:
                    #     table.setdefault(timestep, []).append({'agent': agent, 'loc': [u], 'timestep': timestep - 1, 'positive': False})

            else: # if this is a negative, do as normal
                if agent_c == agent:
                    table.setdefault(timestep, []).append({'agent': agent, 'loc': loc, 'timestep': timestep, 'positive': False})
    
    return table
    pass

def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path

# def is_constrained(curr_loc, next_loc, next_time, constraint_table):
#     ##############################
#     # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
#     #               any given constraint. For efficiency the constraints are indexed in a constraint_table
#     #               by time step, see build_constraint_table.
#     # print(constraint_table)

#     for steptime, constraints in constraint_table.items():
#         if steptime < next_time:
#             for constraint in constraints:
#                 if constraint.get('isGoal') == True and constraint['loc'] == [next_loc]:
#                     return True     

#     if next_time not in constraint_table:
#         return False

#     for constraint in constraint_table[next_time]:
#         if constraint['loc'] == [next_loc] or [curr_loc, next_loc] == constraint['loc']:
#             return True
        
#     return False

#     pass

def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    for steptime, constraints in constraint_table.items():
        if steptime < next_time:
            for c in constraints:
                if c.get('isGoal') and len(c['loc']) == 1 and c['loc'][0] == next_loc:
                    return True

    # negative
    for c in constraint_table.get(next_time, []):
        if not c.get('positive'):
            if c['loc'] == [next_loc] or c['loc'] == [curr_loc, next_loc]:
                return True

    # positive
    pos = [c for c in constraint_table.get(next_time, []) if c.get('positive')]
    if pos:
        for c in pos:
            if (len(c['loc']) == 1 and c['loc'][0] == next_loc) or \
               (len(c['loc']) == 2 and c['loc'] == [curr_loc, next_loc]):
                return False 
        return True
    
    return False



def _priority(node):
    # Use policy-aware f if present, else standard g + h.
    return node.get('f_val', node['g_val'] + node['h_val'])


def push_node(open_list, node):
    heapq.heappush(open_list, (_priority(node), node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return _priority(n1) < _priority(n2)


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, maxTimeStep = None, disjoint=False,
           policy_guidance=None, policy_grid=None, policy_weight: float = 0.0, policy_apply_to_h: bool = False):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    if disjoint:
        table = build_constraint_table_disjoint(constraints, agent)
    else:
        table = build_constraint_table(constraints, agent)
        
    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values.get(start_loc)
    if h_value is None:
        return None
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}
    root['f_val'] = _priority(root)
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root

    def can_wait_forever_from(curr):
        """
        True iff, by waiting at curr['loc'] forever, the agent will not violate
        any FUTURE constraint for this agent.
        """
        for timestep, constraints in table.items():
            if timestep <= curr['timestep']:
                continue
            for constraint in constraints:
                pos = constraint.get('positive', False)
                loc = constraint['loc']
                # FUTURE negative vertex ban exactly at this square -> waiting here would violate
                if not pos and len(loc) == 1 and loc[0] == curr['loc']:
                    return False
                # FUTURE positive that cannot be satisfied by waiting here:
                #  - any positive EDGE
                #  - or positive VERTEX at a different square
                if pos and (len(loc) == 2 or (len(loc) == 1 and loc[0] != curr['loc'])):
                    return False
        return True
    
    # earliest_goal_timestep = goal_block_until
    # earliest_goal_timestep = 0
    # earliest_goal_timestep = max(table.keys()) if len(table) > 0 else 0



    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if maxTimeStep != None:
            if curr['timestep'] >= maxTimeStep:
                return None
            
        earliest_goal_timestep = max(table.keys()) if len(table) > 0 else 1
        # if curr['loc'] == goal_loc and curr['timestep'] >= earliest_goal_timestep:
            # return get_path(curr)
        if curr['loc'] == goal_loc and can_wait_forever_from(curr): 
            return get_path(curr)

        # Collect successors (4 moves + wait)
        candidates = []
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            in_bounds = not (child_loc[0] < 0 or child_loc[0] >= len(my_map) or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]))
            blocked = (not in_bounds) or my_map[child_loc[0]][child_loc[1]] if in_bounds else True
            candidates.append({'loc': child_loc, 'blocked': blocked, 'timestep': curr['timestep'] + 1})
        candidates.append({'loc': curr['loc'], 'blocked': False, 'timestep': curr['timestep'] + 1})  # wait

        order = list(range(len(candidates)))
        scores = None
        if policy_guidance is not None:
            succs = [c['loc'] for c in candidates]
            blocked = [c['blocked'] for c in candidates]
            scores = policy_guidance.score_successors(curr['loc'], goal_loc, curr['timestep'], succs, blocked)
            order = sorted(order, key=lambda i: scores[i], reverse=True)

        for idx in order:
            cand = candidates[idx]
            if cand['blocked']:
                continue
            policy_bonus = 0.0
            if policy_grid is not None and 0 <= cand['loc'][0] < len(policy_grid) and 0 <= cand['loc'][1] < len(policy_grid[0]):
                policy_bonus = policy_grid[cand['loc'][0]][cand['loc'][1]]
            h_val = h_values.get(cand['loc'])
            if h_val is None:
                continue
            policy_score = 0.0
            if scores is not None:
                policy_score = scores[idx]
            total_bonus = policy_bonus + policy_score
            if policy_apply_to_h and policy_weight != 0.0:
                h_val = max(0.0, h_val - policy_weight * total_bonus)
            f_val = curr['g_val'] + 1 + h_val - (policy_weight * total_bonus)
            child = {'loc': cand['loc'],
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_val,
                    'parent': curr, 
                    'timestep': cand['timestep'],
                    'f_val': f_val}

            if is_constrained(curr['loc'], child['loc'], child['timestep'], table):
                continue

            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions

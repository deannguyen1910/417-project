import heapq

# counters for metrics
LOW_LEVEL_GENERATED = 0
LOW_LEVEL_EXPANDED = 0

def reset_low_level_counters():
    global LOW_LEVEL_GENERATED, LOW_LEVEL_EXPANDED
    LOW_LEVEL_GENERATED = 0
    LOW_LEVEL_EXPANDED = 0

def get_low_level_counters():
    return {'generated': LOW_LEVEL_GENERATED, 'expanded': LOW_LEVEL_EXPANDED}

# runtime override, allowing run_experiments to force a_star mode without changing callers
RUN_OVERRIDE = {'mode': None, 'partial_k': None, 'lookahead_depth': None}

def set_run_override(mode=None, partial_k=None, lookahead_depth=None):
    RUN_OVERRIDE['mode'] = mode
    RUN_OVERRIDE['partial_k'] = partial_k
    RUN_OVERRIDE['lookahead_depth'] = lookahead_depth

class NodeLimitExceeded(Exception):
    # node limit to prevent instances from going on too long
    def __init__(self, expanded, generated):
        super().__init__(f"Node limit exceeded: expanded={expanded}, generated={generated}")
        self.expanded = expanded
        self.generated = generated

NODE_LIMIT = None

def set_node_limit(limit):
    global NODE_LIMIT
    NODE_LIMIT = int(limit) if limit is not None else None
    
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



def push_node(open_list, node):
    global LOW_LEVEL_GENERATED
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))
    LOW_LEVEL_GENERATED += 1
    if NODE_LIMIT is not None and LOW_LEVEL_GENERATED > NODE_LIMIT:
        # raise with current counters
        raise NodeLimitExceeded(LOW_LEVEL_EXPANDED, LOW_LEVEL_GENERATED)

def pop_node(open_list):
    global LOW_LEVEL_EXPANDED
    _, _, _, curr = heapq.heappop(open_list)
    LOW_LEVEL_EXPANDED +=1
     # check limit after expanding a node
    if NODE_LIMIT is not None and LOW_LEVEL_EXPANDED > NODE_LIMIT:
        raise NodeLimitExceeded(LOW_LEVEL_EXPANDED, LOW_LEVEL_GENERATED)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

# small epsilon for deprioritizing children in partial
partial_eps = 0.5

# Added modes for project expansion of A*
# 'astar' is default, also 'partial' and 'lookahead'
# partial_k: the integer limit for children per expansion, as a simple heuristic
# lookahead_depth: depth to simulate greeedily from child to find better ordering
def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, maxTimeStep = None, disjoint=False,
          mode='astar', partial_k = None, lookahead_depth = 0):
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

    if RUN_OVERRIDE['mode'] is not None:
        mode = RUN_OVERRIDE['mode']
        partial_k = RUN_OVERRIDE['partial_k']
        lookahead_depth = RUN_OVERRIDE['lookahead_depth']
    
    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}
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

    def bounded_lookahead_estimate(start_loc_inner, start_g, start_timestep, depth_limit, my_map_inner, h_values_inner, table_inner, goal_loc_inner=None):
        # Run a small bounded A* from start_loc for at most depth_limit node expansions.
        #Returns the smallest f = g + h encountered among explored nodes.
        # min-heap by f, then h
        open_heap = []
        closed = set()
        start_h = h_values_inner.get(start_loc_inner, 999999)
        heapq.heappush(open_heap, (start_g + start_h, start_h, start_loc_inner, start_g, start_timestep))
        best_f = start_g + start_h
        expansions = 0

        while open_heap and expansions < depth_limit:
            f, hval, loc, gval, timestep = heapq.heappop(open_heap)
            if (loc, timestep) in closed:
                continue
            closed.add((loc, timestep))
            expansions += 1

            # If we reached goal (and goal passed in), short-circuit with actual g
            if goal_loc_inner is not None and loc == goal_loc_inner:
                # return tight estimate (g to goal) so child gets prioritized strongly
                return gval

            if f < best_f:
                best_f = f

            # generate successors
            for dir in range(4):
                child_loc = move(loc, dir)
                if (child_loc[0] < 0 or child_loc[0] >= len(my_map_inner) or child_loc[1] < 0 or child_loc[1] >= len(my_map_inner[0])):
                    continue
                if my_map_inner[child_loc[0]][child_loc[1]]:
                    continue
                next_t = timestep + 1
                if is_constrained(loc, child_loc, next_t, table_inner):
                    continue
                child_g = gval + 1
                child_h = h_values_inner.get(child_loc, 999999)
                child_f = child_g + child_h
                if (child_loc, next_t) not in closed:
                    heapq.heappush(open_heap, (child_f, child_h, child_loc, child_g, next_t))

            # also consider waiting
            next_t = timestep + 1
            if not is_constrained(loc, loc, next_t, table_inner):
                child_g = gval + 1
                child_h = h_values_inner.get(loc, 999999)
                child_f = child_g + child_h
                if (loc, next_t) not in closed:
                    heapq.heappush(open_heap, (child_f, child_h, loc, child_g, next_t))

        return best_f

    LOOKAHEAD_CACHE = {}
    LOOKAHEAD_CACHE_MAX = 10000 

    def greedy_lookahead_estimate(start_loc, start_g, start_timestep, depth_limit, my_map, h_values, table, goal_loc=None):
        # cheap cache key: (loc, timestep, depth_limit)
        cache_key = (start_loc, start_timestep, depth_limit)
        if cache_key in LOOKAHEAD_CACHE:
            return LOOKAHEAD_CACHE[cache_key] + start_g  # cache stores delta estimate relative to g=0

        best = start_g + h_values.get(start_loc, 999999)
        # stack entries: (loc, g, timestep, depth)
        stack = [(start_loc, start_g, start_timestep, 0)]
        visited = set()
        while stack:
            loc, g, timestep, depth = stack.pop()
            if depth >= depth_limit:
                continue
            # build candidate moves sorted by h
            candidates = []
            for dir in range(4):
                child_loc = move(loc, dir)
                if child_loc[0] < 0 or child_loc[0] >= len(my_map) or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                    continue
                if my_map[child_loc[0]][child_loc[1]]:
                    continue
                if is_constrained(loc, child_loc, timestep+1, table):
                    continue
                candidates.append((h_values.get(child_loc, 999999), child_loc))
            # waiting candidate
            if not is_constrained(loc, loc, timestep+1, table):
                candidates.append((h_values.get(loc, 999999), loc))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])
            for hval, child_loc in candidates:
                new_g = g + 1
                estimate = new_g + hval
                if estimate < best:
                    best = estimate
                key = (child_loc, timestep+1, depth+1)
                if key not in visited:
                    visited.add(key)
                    # push lower-h candidates first so stack pop explores them last (LIFO)
                    stack.append((child_loc, new_g, timestep+1, depth+1))
        # store a delta so cache reuse works regardless of start_g (we store best - start_g)
        delta = best - start_g
        if len(LOOKAHEAD_CACHE) < LOOKAHEAD_CACHE_MAX:
            LOOKAHEAD_CACHE[cache_key] = delta
        else:
            # simple eviction: pop arbitrary item when full
            LOOKAHEAD_CACHE.pop(next(iter(LOOKAHEAD_CACHE)))
            LOOKAHEAD_CACHE[cache_key] = delta
        return best
    
    # depth limited helper for greedy lookahead
    def greedy_lookahead_sim(start_node, depth_limit):
        # move towards goal, choosing neighbor with smallest h, respecting map/constraints. Return best g+h estimate
        best_estimate = start_node['g_val'] + start_node['h_val']
        stack = [(start_node['loc'], start_node['g_val'], start_node['timestep'], 0)]
        visited = set()
        while stack:
            loc, g_val, timestep, depth = stack.pop()
            if depth >= depth_limit:
                continue
            candidates = []
            for dir in range(4):
                child_loc = move(loc, dir)
                if (child_loc[0] < 0 or child_loc[0] >= len(my_map) or child_loc[1] < 0 or child_loc[1] >= len(my_map[0])):
                    continue
                if my_map[child_loc[0]][child_loc[1]]:
                    continue
                if is_constrained(loc, child_loc, timestep+1, table):
                    continue
                candidates.append((h_values.get(child_loc, 999999), child_loc))
            if not is_constrained(loc,loc, timestep + 1, table):
                candidates.append((h_values.get(loc, 999999), loc))
            candidates.sort(key=lambda x: x[0])
            for hval, child_loc in candidates:
                new_g = g_val +1
                estimate = new_g + hval
                if estimate < best_estimate:
                    best_estimate = estimate
                if (child_loc, timestep + 1) not in visited:
                    visited.add((child_loc, timestep + 1))
                    stack.append((child_loc, new_g, timestep+1, depth+1))
        return best_estimate
    
    # earliest_goal_timestep = goal_block_until
    # earliest_goal_timestep = 0
    # earliest_goal_timestep = max(table.keys()) if len(table) > 0 else 0



    while len(open_list) > 0:
        curr = pop_node(open_list)

        if maxTimeStep is not None and curr['timestep'] >= maxTimeStep:
            return None

        if curr['loc'] == goal_loc and can_wait_forever_from(curr):
            return get_path(curr)

        # enumerate child moves
        children = []
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            if (child_loc[0] < 0 or child_loc[0] >= len(my_map) or child_loc[1] < 0 or child_loc[1] >= len(my_map[0])):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values.get(child_loc, 999999),
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}
            if is_constrained(curr['loc'], child['loc'], child['timestep'], table):
                continue
            children.append(child)

        waiting = {'loc': curr['loc'],
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values.get(curr['loc'], 999999),
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}
        if not is_constrained(curr['loc'], waiting['loc'], waiting['timestep'], table):
            children.append(waiting)

        # if lookahead mode: calculate an estimate for each child and use it to sort children.
        if mode == 'lookahead' and lookahead_depth and len(children) > 0:
            scored = []
            for child in children:
                # pass goal_loc so bounded estimate can short-circuit if it reaches goal
                #sim = greedy_lookahead_sim(child, lookahead_depth)
                #sim = bounded_lookahead_estimate(child['loc'], child['g_val'], child['timestep'],
                #                                lookahead_depth, my_map, h_values, table, goal_loc)
                sim = greedy_lookahead_estimate(child['loc'], child['g_val'], child['timestep'],
                                                lookahead_depth, my_map, h_values, table, goal_loc)
                scored.append((sim, child))
            scored.sort(key=lambda x: (x[0], x[1]['g_val'] + x[1]['h_val']))
            children = [c for _, c in scored]

        # If partial mode: keep only best partial_k children at full priority,
        # but keep the rest in the open with a small epsilon penalty
        if mode == 'partial' and partial_k is not None and len(children) > partial_k:
            # sort children by f, then h
            children.sort(key=lambda n: (n['g_val'] + n['h_val'], n['h_val']))
            selected = children[:partial_k]
            deferred = children[partial_k:]

            # push selected children normally (standard duplicate handling)
            for child in selected:
                key = (child['loc'], child['timestep'])
                if key in closed_list:
                    existing_node = closed_list[key]
                    if compare_nodes(child, existing_node):
                        closed_list[key] = child
                        push_node(open_list, child)
                else:
                    closed_list[key] = child
                    push_node(open_list, child)

            # compute next f for parent to be strictly greater than current f if possible
            curr_f = curr['g_val'] + curr['h_val']
            deferred_fs = [c['g_val'] + c['h_val'] for c in deferred]
            # choose the smallest deferred f that is strictly greater than curr_f
            greater_fs = [f for f in deferred_fs if f > curr_f]
            if greater_fs:
                next_f = min(greater_fs)
            else:
                # fallback: bump by 1 to ensure progress
                next_f = curr_f + 1

            # reinsert parent with updated h so parent's f == next_f
            parent_copy = dict(curr)
            parent_copy['h_val'] = next_f - parent_copy['g_val']
            push_node(open_list, parent_copy)

            continue

        else:
            # normal order (or lookahead-modified order)
            children.sort(key=lambda n: (n['g_val'] + n['h_val'], n['h_val']))

        # push children into open/closed list, handling duplicates
        for child in children:
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions

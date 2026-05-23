import numpy as np
import math
from matplotlib import pyplot as plt
import sys
import os 
from time import sleep
import copy
from PIL import Image

from online_motion_planning.Point import Point
from online_motion_planning.rrt import RRT

class RRT_STAR(RRT):
    def __init__(self, delta_q, p, max_depth, min_dist, radius, threshold_path_rewire_dist):
        super().__init__(delta_q, p, max_depth, min_dist) # inherence from base class
        self.radius = radius # radius for rewiring
        self.threshold_path_rewire_dist = threshold_path_rewire_dist # distance threshold to stop rewiring when having no improvement on the found paths' distance
        
    def near_vertices_in_a_radius(self, G, q, radius):
        vertices = []
        for vertice in G:
            dist = vertice.dist(q)
            if dist <= radius:
                vertices.append(vertice)
        return vertices

    def rewire_qnear_to_qnew(self, C, G, edges, cost, qnear, qnew, radius):
        qmin = qnear
        cost_q_new = cost[G.index(qnear)] + qnear.dist(qnew)
        cost.append(cost_q_new) # temporary qnew cost
        near_vertices = self.near_vertices_in_a_radius(G, qnew, radius)
        for vertice in near_vertices:
            if self.is_segment_free_bisection(vertice, qnew, C, 0) and cost[G.index(vertice)] + vertice.dist(qnew) < cost[-1]:
                qmin = vertice
                cost[-1] = cost[G.index(vertice)] + vertice.dist(qnew)
        G.append(qnew)
        edges.append([G.index(qmin), G.index(qnew)])
        return G, edges, cost

    def find_parent(self, edges, idx_q):
        i = 0
        while edges[i][1] != idx_q:
            i += 1
        return edges[i][0]

    def rewire_qnew_from_qnear(self, C, G, edges, cost, qnew, radius):
        near_vertices = self.near_vertices_in_a_radius(G, qnew, radius)
        for vertice in near_vertices:
            if self.is_segment_free_bisection(vertice, qnew, C, 0) and cost[G.index(qnew)] + vertice.dist(qnew) < cost[G.index(vertice)]:
                cost[G.index(vertice)] = cost[G.index(qnew)] + vertice.dist(qnew)
                parent_idx = self.find_parent(edges,G.index(vertice))
                edges.remove([parent_idx, G.index(vertice)])# remove the edge from parent node to near vertie
                edges.append([G.index(qnew), G.index(vertice)])

        return G, edges, cost

    # ORIGINAL SIGNATURE:
    # def sample(self, C, K, qstart_x, qstart_y, qgoal_x, qgoal_y):
    #
    # WHY CHANGED:
    # Added two optional keyword arguments:
    # - logger=None: routes all diagnostic output through the ROS2 node logger
    #   instead of print(), so messages appear in ros2 launch output with correct
    #   severity levels and can be filtered/throttled.  Falls back to print() when
    #   called standalone (e.g. the __main__ block), so the original usage is
    #   fully backward-compatible.
    # - viz_callback=None: a callable(G, parent_idx, child_idx) injected by the
    #   frontier node to publish incremental RViz LINE_LIST markers as the tree
    #   grows, without making rrt_star.py depend on any ROS publisher directly.
    #   Also added: rejected_collision / added_nodes counters that feed the logger
    #   summary and the periodic debug line every 200 iterations.
    def sample(self, C, K, qstart_x, qstart_y, qgoal_x, qgoal_y, logger=None, viz_callback=None):

        qstart = Point(qstart_x, qstart_y)
        qgoal = Point(qgoal_x, qgoal_y)
        G = [qstart]
        edges = []
        cost = [0.0]
        min_dist_to_goal = float('inf')
        G_min_dist = [qstart]
        edges_min_dist = []

        count_reach_goal = 0
        max_reach_goal = 5

        # Log initial state and validate start
        if logger:
            logger.debug(f"RRT* sample: C.shape={C.shape}, start=({qstart_x}, {qstart_y}), goal=({qgoal_x}, {qgoal_y})")
            start_occupied = self.is_point_occupied(qstart, C)
            goal_occupied = self.is_point_occupied(qgoal, C)
            logger.info(f"RRT* start occupied: {start_occupied}, goal occupied: {goal_occupied}")

        # Track rejection reasons
        rejected_collision = 0
        added_nodes = 0

        for k in range(K):
            qrand = self.rand_conf(C, qgoal)
            qnear = self.nearest_vertex(qrand, G)
            qnew = self.new_conf(qnear, qrand)

            if self.is_segment_free_bisection(qnear, qnew, C, 0):
                G, edges, cost = self.rewire_qnear_to_qnew(C, G, edges, cost, qnear, qnew, self.radius)
                G, edges, cost = self.rewire_qnew_from_qnear(C, G, edges, cost, qnew, self.radius)
                # ORIGINAL: no viz_callback call here.
                # WHY ADDED: allows the frontier node to stream incremental tree
                # edges to RViz without coupling this class to any ROS publisher.
                if viz_callback is not None:
                    viz_callback(G, edges[-1][0], edges[-1][1])
                dist = qnew.dist(qgoal)
                added_nodes += 1

                # Log progress every 200 iterations
                if (k + 1) % 200 == 0 and logger:
                    logger.debug(f"RRT* iter {k+1}/{K}: tree_size={len(G)}, added={added_nodes}, rejected={rejected_collision}, dist_to_goal={dist:.2f}")

                if dist < self.min_dist:
                    G, edges, path = self.fill_path(G, edges)
                    smooth_path = self.smoothing(C, G, path)
                    temp_dist = self.calculate_distace_start_to_goal(smooth_path, G)
                    # ORIGINAL: print(f"Find path after {k} iterations. Distance {temp_dist}")
                    # WHY CHANGED: logger routes through ROS2 so output appears in
                    # launch logs with correct severity; print() kept as fallback.
                    if logger:
                        logger.info(f"RRT* find path after {k} iterations. Distance {temp_dist:.2f}")
                    else:
                        print(f"Find path after {k} iterations. Distance {temp_dist}")
                    count_reach_goal += 1
                    if temp_dist < min_dist_to_goal:
                        if (min_dist_to_goal - temp_dist < self.threshold_path_rewire_dist):
                            # ORIGINAL: print(f"Path converges to minimum distance {temp_dist}. Stop iterating!")
                            if logger:
                                logger.info(f"RRT* path converges to minimum distance {temp_dist:.2f}")
                            else:
                                print(f"Path converges to minimum distance {temp_dist}. Stop iterating!")
                            G.append(qgoal)
                            edges.append([G.index(qnew), G.index(qgoal)])
                            return G, edges, k
                        else:
                            min_dist_to_goal = temp_dist
                            G_min_dist = copy.deepcopy(G)
                            G_min_dist.append(qgoal)
                            edges_min_dist = copy.deepcopy(edges)
                            edges_min_dist.append([G.index(qnew), G_min_dist.index(qgoal)])

                    if count_reach_goal >= max_reach_goal:
                        # ORIGINAL: print("Exceed max attempts to find the shortest path. Stop iterating!")
                        if logger:
                            logger.info(f"RRT* exceed max attempts to find shortest path after {k} iterations")
                        else:
                            print("Exceed max attempts to find the shortest path. Stop iterating!")
                        return G_min_dist, edges_min_dist, K
            else:
                rejected_collision += 1

        if logger:
            logger.warn(f"RRT* sampling completed all {K} iterations without finding path. Final tree: added={added_nodes}, rejected={rejected_collision}, tree_size={len(G)}")
        return G_min_dist, edges_min_dist, K

if __name__ == "__main__":
    # Read input from command line
    image_path = str(sys.argv[1]) # file path
    K = int(sys.argv[2]) # number of iterations
    delta_q = int(sys.argv[3]) # step
    p = float(sys.argv[4]) # probability q_rand to be q_goal
    radius = int(sys.argv[5])
    qstart_x = int(sys.argv[6]) # qstart coordinates
    qstart_y = int(sys.argv[7])
    qgoal_x = int(sys.argv[8]) # qgoal coordinates
    qgoal_y = int(sys.argv[9])
    

    max_depth = round(math.log(delta_q, 2)) + 1 # max iteration for considering a segment free or not
    min_dist = 5

    # Load grid map
    image = Image.open(image_path).convert("L")
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    grid_map[grid_map > 0.5] = 1
    grid_map[grid_map <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    grid_map = (grid_map * -1) + 1
    # Show grid map
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(grid_map)
    fig.colorbar(cax)
    plt.show()
    
    
    threshold_path_rewire_dist = 5
    
    path = []
    rrt_star = RRT_STAR(delta_q, p, max_depth, min_dist, radius, threshold_path_rewire_dist)
    G, edges, iter = rrt_star.sample(grid_map, K, qstart_x, qstart_y, qgoal_x, qgoal_y)
    if iter == K and len(edges) == 0:
        print("Cannot find a path within a defined iteration numbers")
    else:
        G, edges, path = rrt_star.fill_path(G, edges)
        path = rrt_star.smoothing(grid_map, G, path)
        rrt_star.plot(grid_map, G, edges, path)
    
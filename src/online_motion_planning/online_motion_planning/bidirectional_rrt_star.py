import math

from online_motion_planning.Point import Point
from online_motion_planning.rrt_star import RRT_STAR

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON WITH rrt_star.py
# ─────────────────────────────────────────────────────────────────────────────
# rrt_star.py (RRT_STAR) grows a SINGLE tree from the start toward the goal.
# At each iteration it samples one random point, extends the tree one step,
# and checks whether the new node is within min_dist of the goal.  Because
# the tree only grows from one end, it has to cover the entire distance from
# start to goal before a path is found.
#
# This file (BIRRT_STAR) grows TWO trees simultaneously:
#   T_a — forward tree rooted at the start position
#   T_b — backward tree rooted at the goal position
# Each iteration extends BOTH trees and immediately tries to bridge them.
# Because both ends grow toward the middle, the expected search depth is halved
# which means paths are usually found much faster, especially in cluttered maps.
#
# The return signature (G, edges, k) is identical to RRT_STAR.sample() so the
# caller (path_planning_loop) does not need to change: k < K = success,
# k == K = failure.  The returned G / edges are a sequential path
# [start, ..., connection_a, connection_b, ..., goal] rather than the full
# tree, but fill_path() and smoothing() work on it unchanged.
# ─────────────────────────────────────────────────────────────────────────────


class BIRRT_STAR(RRT_STAR):
    """Bidirectional RRT* (BiRRT*).

    Inherits from RRT_STAR so it reuses all of the following without changes:
      - rand_conf()               — random free-cell sampling with goal bias
      - nearest_vertex()          — nearest-neighbour search in a node list
      - new_conf()                — one-step extension toward a target
      - is_point_occupied()       — single-cell occupancy check on binary_map
      - is_segment_free_bisection() — recursive bisection collision check
      - rewire_qnear_to_qnew()    — RRT* rewiring: find best parent for new node
      - rewire_qnew_from_qnear()  — RRT* rewiring: improve neighbours via new node
      - fill_path()               — trace edges backward to reconstruct path indices
      - smoothing()               — greedy path shortening
      - plot()                    — matplotlib debug plot

    The only method overridden is sample().
    Three new private methods support the bidirectional logic:
      - _trace_to_root()   — walk edges backward from any node to the tree root
      - _build_unified()   — merge the two sub-paths into a single G / edges
      - _try_connect()     — attempt to bridge T_b to a new T_a node
    """

    # ── path-reconstruction helpers ───────────────────────────────────────────
    # NEW — these methods do not exist in rrt_star.py.
    # They are needed because BiRRT* builds two separate edge lists (ea, eb) and
    # must merge them into the sequential format that fill_path() expects.

    def _trace_to_root(self, edges, end_idx):
        """Walk the edge list backward from end_idx to the root (index 0).

        rrt_star.py equivalent: fill_path() does the same traversal inline, but
        only for a single tree that always ends at the goal node.  Here we need
        to do the same traversal for EITHER tree (T_a or T_b) starting from
        whichever node formed the bridge, so it is factored out as a helper.

        Returns: list of node indices [0, ..., end_idx]  (root-first order).
        """
        if end_idx == 0:
            return [0]
        path = [end_idx]
        current = end_idx
        while current != 0:
            for parent_idx, child_idx in edges:
                if child_idx == current:
                    path.append(parent_idx)
                    current = parent_idx
                    break
            else:
                break  # disconnected — should not happen in a valid RRT* tree
        path.reverse()
        return path

    def _build_unified(self, Ga, ea, qa_idx, Gb, eb, qb_idx):
        """Merge the two trees at the bridge point qa (in T_a) ↔ qb (in T_b).

        rrt_star.py equivalent: none — single-tree RRT* never needs to merge
        two separate node lists.  In BiRRT*, when the trees connect we must:
          1. Extract the path through T_a from start (Ga[0]) to qa (Ga[qa_idx]).
          2. Extract the path through T_b from goal (Gb[0]) to qb (Gb[qb_idx])
             and REVERSE it so it reads qb → ... → goal.
          3. Concatenate: [start, ..., qa, qb, ..., goal].
          4. Build sequential edges [[0,1],[1,2],...] so fill_path() can trace
             the result in its normal backward-scan loop.

        WHY sequential edges:
          fill_path() scans edges looking for child == current_node.  With edges
          [[0,1],[1,2],...,[n-2,n-1]], the scan always succeeds in O(n) and the
          final path is [0, 1, ..., n-1] — exactly the node order we built.

        Returns: (G_unified, edges_unified)
          G_unified     — list of Point objects: [start, ..., qa, qb, ..., goal]
          edges_unified — [[0,1],[1,2],...] sequential edge list
        """
        # Forward sub-path: start → ... → qa (root-first from _trace_to_root)
        pts_a = [Ga[i] for i in self._trace_to_root(ea, qa_idx)]

        # Backward sub-path: Gb[0]=goal → ... → Gb[qb_idx]=qb
        # Reversed so it reads qb → ... → goal, continuing from qa.
        pts_b = [Gb[i] for i in self._trace_to_root(eb, qb_idx)]
        pts_b.reverse()   # now: [qb, ..., goal]

        G_unified     = pts_a + pts_b
        edges_unified = [[i, i + 1] for i in range(len(G_unified) - 1)]
        return G_unified, edges_unified

    # ── connection attempt ────────────────────────────────────────────────────
    # NEW — does not exist in rrt_star.py.
    # rrt_star.py checks connection as: dist(qnew, qgoal) < min_dist (a fixed
    # point).  BiRRT* checks connection as: dist(qnew_a, nearest_b) < min_dist
    # (a moving frontier, the nearest node of the other tree).

    def _try_connect(self, C, Ga, ea, ca, qa_idx, Gb, eb, cb,
                     viz_cb_passive=None):
        """Try to bridge the passive tree T_b to the active-tree node Ga[qa_idx].

        Called once per successful T_a extension (and once per T_b extension with
        roles swapped) to check whether the two trees can be joined.

        Two cases:

        Case 1 — direct bridge:
          The nearest T_b node (qnear_b) is already within min_dist of qa AND
          a collision-free segment exists between them.
          → Return (qnear_b_idx, total_path_cost) immediately; T_b is unchanged.

        Case 2 — step-then-check:
          qnear_b is farther than min_dist.  Step T_b one delta_q toward qa
          (using the inherited new_conf / rewire methods, so T_b gets a proper
          RRT* node with full rewiring), then re-check the distance.
          → If the new node qnew_b is now within min_dist of qa AND a collision-
            free segment exists, return (qnew_b_idx, cost).
          → Otherwise return (None, inf).  The T_b extension still happened as a
            side-effect, which is desirable: it grows T_b toward T_a even when
            connection is not yet possible.

        viz_cb_passive:
          rrt_star.py does not have this parameter because there is only one tree.
          Here, when Case 2 silently extends T_b inside _try_connect, the growth
          would be invisible to the visualiser.  Passing viz_cb_passive causes
          the new edge to be reported to the RViz callback immediately so both
          trees appear to grow at the same visual rate.

        Returns: (connected_qb_idx, path_cost)  or  (None, inf)
          connected_qb_idx — index into Gb of the node that bridged to qa
          path_cost        — ca[qa_idx] + bridge_dist + cb[qb_idx]
                             (sum of costs from start→qa + bridge + qb→goal)
        """
        qnew_a      = Ga[qa_idx]
        qnear_b     = self.nearest_vertex(qnew_a, Gb)   # inherited from RRT
        qnear_b_idx = Gb.index(qnear_b)

        # Case 1 — direct connection (no T_b extension needed)
        # min_dist is the same threshold used in rrt_star.py to detect goal-reach
        if (qnear_b.dist(qnew_a) < self.min_dist and
                self.is_segment_free_bisection(qnear_b, qnew_a, C, 0)):
            cost = ca[qa_idx] + qnear_b.dist(qnew_a) + cb[qnear_b_idx]
            return qnear_b_idx, cost

        # Case 2 — extend T_b one step toward qa using the same RRT* machinery
        # that is used for normal tree extensions in rrt_star.py's sample loop.
        qnew_b = self.new_conf(qnear_b, qnew_a)   # inherited: step delta_q toward qa
        if not self.is_segment_free_bisection(qnear_b, qnew_b, C, 0):
            return None, float('inf')   # extension blocked — cannot bridge yet

        # Apply full RRT* rewiring to the new T_b node (same as in rrt_star.py)
        Gb, eb, cb = self.rewire_qnear_to_qnew(C, Gb, eb, cb, qnear_b, qnew_b, self.radius)
        Gb, eb, cb = self.rewire_qnew_from_qnear(C, Gb, eb, cb, qnew_b, self.radius)
        qb_new_idx = len(Gb) - 1   # index of the just-added node

        # Notify the visualiser about this silent T_b growth so both trees
        # appear to grow at the same frame rate in RViz.
        if viz_cb_passive is not None:
            viz_cb_passive(Gb, eb[-1][0], eb[-1][1])

        # Re-check: is the new T_b node close enough to qa to bridge?
        if (qnew_b.dist(qnew_a) < self.min_dist and
                self.is_segment_free_bisection(qnew_b, qnew_a, C, 0)):
            cost = ca[qa_idx] + qnew_b.dist(qnew_a) + cb[qb_new_idx]
            return qb_new_idx, cost

        return None, float('inf')   # extended T_b but still not close enough

    # ── main sampler ──────────────────────────────────────────────────────────

    def sample(self, C, K, qstart_x, qstart_y, qgoal_x, qgoal_y,
               logger=None,
               viz_callback=None,       # alias for viz_callback_a (rrt_star compat)
               viz_callback_a=None,     # visualiser callback for T_a (start tree)
               viz_callback_b=None):    # visualiser callback for T_b (goal tree)
        """Bidirectional RRT* main loop.

        COMPARISON WITH rrt_star.py sample():
        ──────────────────────────────────────
        rrt_star.py:
          - One tree G, one edge list, one cost list.
          - Each iteration: sample qrand → extend G → check dist(qnew, qgoal).
          - Tracks min_dist_to_goal and G_min_dist for the best path found.
          - Stops after max_reach_goal (5) connections or K iterations.
          - Returns the full tree G (potentially thousands of nodes).

        BIRRT_STAR:
          - Two trees (Ga/ea/ca for start, Gb/eb/cb for goal).
          - Each iteration: extend T_a → try bridge → extend T_b → try bridge.
          - Tracks best_cost and best_G (the merged sequential path, not the tree).
          - Stops after MAX_CONNECT (5) successful bridges or K iterations.
          - Returns a compact sequential path rather than the full tree.

        Key return-value compatibility:
          Both return (G, edges, k).  For BIRRT*, G is the merged path list and
          edges are sequential [[0,1],[1,2],...], so fill_path() and smoothing()
          work without any changes.  k < K signals success; k == K signals failure.

        Additional parameters vs rrt_star.py:
          viz_callback_a — callback(G, parent_idx, child_idx) for T_a edges.
          viz_callback_b — same for T_b edges.
          viz_callback   — backward-compatible alias wired to viz_callback_a so
                           any caller that passes viz_callback= still works.
        """
        # Backward-compatible alias: if only viz_callback is passed (rrt_star style),
        # treat it as viz_callback_a so T_a is still visualised.
        if viz_callback is not None and viz_callback_a is None:
            viz_callback_a = viz_callback

        qstart = Point(qstart_x, qstart_y)
        qgoal  = Point(qgoal_x,  qgoal_y)

        # ── Two trees instead of one ──────────────────────────────────────────
        # rrt_star.py:  G = [qstart];  edges = [];  cost = [0.0]
        # BIRRT_STAR:   Two separate node/edge/cost lists, one per tree.
        #   Ga / ea / ca — forward tree rooted at start  (index 0 = qstart)
        #   Gb / eb / cb — backward tree rooted at goal  (index 0 = qgoal)
        Ga = [qstart];  ea = [];  ca = [0.0]
        Gb = [qgoal];   eb = [];  cb = [0.0]

        if logger:
            logger.debug(
                f"BiRRT* sample: C={C.shape}, "
                f"start=({qstart_x:.1f},{qstart_y:.1f}), "
                f"goal=({qgoal_x:.1f},{qgoal_y:.1f})"
            )
            logger.info(
                f"BiRRT* start occupied: {self.is_point_occupied(qstart, C)}, "
                f"goal occupied: {self.is_point_occupied(qgoal, C)}"
            )

        # ── Convergence tracking ──────────────────────────────────────────────
        # rrt_star.py uses:
        #   min_dist_to_goal — best smoothed path length found so far
        #   G_min_dist / edges_min_dist — snapshot of tree at best path
        #   count_reach_goal / max_reach_goal=5 — stop after 5 improvements
        #
        # BIRRT_STAR uses:
        #   best_cost  — sum of costs from both trees through the bridge
        #   best_G / best_edges — merged sequential path at best bridge
        #   connect_cnt / MAX_CONNECT=5 — stop after 5 successful bridges
        #
        # The logic is analogous: keep the best result found so far and stop
        # when further improvement is unlikely.
        best_cost    = float('inf')
        best_G       = None
        best_edges   = None
        connect_cnt  = 0
        MAX_CONNECT  = 5      # keep optimising until this many bridges found

        # Counters for the logger summary — same role as in rrt_star.py
        added_a = added_b = rejected_a = rejected_b = 0

        for k in range(K):

            # ── Extend T_a (forward / start tree) ────────────────────────────
            # Identical in structure to the rrt_star.py loop body:
            #   sample qrand_a biased toward qgoal
            #   → find nearest T_a node
            #   → step one delta_q
            #   → collision check
            #   → RRT* rewire (rewire_qnear_to_qnew + rewire_qnew_from_qnear)
            # CHANGED: goal-reach check (dist < min_dist) is replaced by the
            # bridge attempt via _try_connect().
            qrand_a = self.rand_conf(C, qgoal)    # bias toward goal, same as rrt_star
            qnear_a = self.nearest_vertex(qrand_a, Ga)
            qnew_a  = self.new_conf(qnear_a, qrand_a)

            if self.is_segment_free_bisection(qnear_a, qnew_a, C, 0):
                Ga, ea, ca = self.rewire_qnear_to_qnew(C, Ga, ea, ca, qnear_a, qnew_a, self.radius)
                Ga, ea, ca = self.rewire_qnew_from_qnear(C, Ga, ea, ca, qnew_a, self.radius)
                qa_idx = len(Ga) - 1   # index of the just-added node
                added_a += 1

                # Notify T_a visualiser — same mechanism as rrt_star.py's viz_callback
                if viz_callback_a is not None:
                    viz_callback_a(Ga, ea[-1][0], ea[-1][1])

                # ── Bridge attempt after T_a extension ───────────────────────
                # rrt_star.py: checks dist(qnew, qgoal) < min_dist (fixed point).
                # BIRRT_STAR:  checks dist(qnew_a, nearest_T_b_node) < min_dist
                #              via _try_connect, which may also silently extend T_b
                #              one step toward qnew_a (Case 2 in _try_connect).
                qb_idx, cost = self._try_connect(C, Ga, ea, ca, qa_idx, Gb, eb, cb,
                                                 viz_cb_passive=viz_callback_b)
                if qb_idx is not None:
                    connect_cnt += 1
                    if cost < best_cost:
                        best_cost  = cost
                        # Merge the two sub-paths into a unified sequential path.
                        # rrt_star.py equivalent: G.append(qgoal); edges.append([...])
                        # Here we cannot just append because T_b's path must be reversed.
                        best_G, best_edges = self._build_unified(Ga, ea, qa_idx, Gb, eb, qb_idx)
                        if logger:
                            logger.info(
                                f"BiRRT* connection #{connect_cnt} at iter {k}, "
                                f"cost={cost:.2f}, |T_a|={len(Ga)}, |T_b|={len(Gb)}"
                            )
                    # rrt_star.py equivalent: count_reach_goal >= max_reach_goal
                    if connect_cnt >= MAX_CONNECT:
                        if logger:
                            logger.info(
                                f"BiRRT* converged after {connect_cnt} connections "
                                f"({k+1} iterations)."
                            )
                        return best_G, best_edges, k
            else:
                rejected_a += 1

            # ── Extend T_b (backward / goal tree) ────────────────────────────
            # NEW — rrt_star.py has no backward tree.
            # Structure is the mirror image of the T_a block above:
            #   sample qrand_b biased toward qstart (not qgoal)
            #   → find nearest T_b node
            #   → step one delta_q
            #   → full RRT* rewire
            #   → bridge attempt: try to connect T_a to the new T_b node
            #     (roles of Ga/Gb are SWAPPED in the _try_connect call so the
            #      function always sees the "active" tree as its first tree argument)
            qrand_b = self.rand_conf(C, qstart)   # bias toward start (reversed direction)
            qnear_b = self.nearest_vertex(qrand_b, Gb)
            qnew_b  = self.new_conf(qnear_b, qrand_b)

            if self.is_segment_free_bisection(qnear_b, qnew_b, C, 0):
                Gb, eb, cb = self.rewire_qnear_to_qnew(C, Gb, eb, cb, qnear_b, qnew_b, self.radius)
                Gb, eb, cb = self.rewire_qnew_from_qnear(C, Gb, eb, cb, qnew_b, self.radius)
                qb_new_idx = len(Gb) - 1
                added_b += 1

                if viz_callback_b is not None:
                    viz_callback_b(Gb, eb[-1][0], eb[-1][1])

                # Bridge attempt after T_b extension.
                # NOTE: Ga and Gb are SWAPPED here vs the T_a block above.
                # _try_connect(active=Gb, passive=Ga) means:
                #   "try to connect T_a (passive) to the new T_b node (active)".
                # The returned qa_conn_idx is an index into Ga (the passive tree).
                qa_conn_idx, cost = self._try_connect(C, Gb, eb, cb, qb_new_idx, Ga, ea, ca,
                                                      viz_cb_passive=viz_callback_a)
                if qa_conn_idx is not None:
                    connect_cnt += 1
                    if cost < best_cost:
                        best_cost  = cost
                        # _build_unified still receives (Ga, ea, Ga_idx, Gb, eb, Gb_idx)
                        # in the canonical order (forward tree first) so the merged path
                        # always reads [start, ..., qa, qb, ..., goal].
                        best_G, best_edges = self._build_unified(
                            Ga, ea, qa_conn_idx, Gb, eb, qb_new_idx
                        )
                        if logger:
                            logger.info(
                                f"BiRRT* (backward) connection #{connect_cnt} at iter {k}, "
                                f"cost={cost:.2f}, |T_a|={len(Ga)}, |T_b|={len(Gb)}"
                            )
                    if connect_cnt >= MAX_CONNECT:
                        if logger:
                            logger.info(
                                f"BiRRT* converged after {connect_cnt} connections "
                                f"({k+1} iterations)."
                            )
                        return best_G, best_edges, k
            else:
                rejected_b += 1

            # Periodic progress log — same cadence as rrt_star.py (every 200 iters)
            if (k + 1) % 200 == 0 and logger:
                logger.debug(
                    f"BiRRT* iter {k+1}/{K}: |T_a|={len(Ga)}, |T_b|={len(Gb)}, "
                    f"connections={connect_cnt}, best_cost={best_cost:.2f}"
                )

        # ── K iterations exhausted ────────────────────────────────────────────
        # rrt_star.py: always returns G_min_dist (possibly [qstart]) and edges_min_dist.
        # BIRRT_STAR:
        #   If at least one bridge was found → return the best merged path.
        #     The caller sees len(edges) > 0 and treats this as success.
        #   If no bridge was found → return [qstart], [], K.
        #     The caller sees len(edges) == 0 and treats this as failure.
        if logger:
            logger.warn(
                f"BiRRT* completed all {K} iterations without converging. "
                f"|T_a|={len(Ga)} (added={added_a}, rejected={rejected_a}), "
                f"|T_b|={len(Gb)} (added={added_b}, rejected={rejected_b}), "
                f"connections={connect_cnt}"
            )

        if best_G is not None:
            if logger:
                logger.info(f"BiRRT* returning best partial connection, cost={best_cost:.2f}")
            # k == K-1 here; caller checks len(edges)==0 for failure, which is False
            return best_G, best_edges, k

        # Complete failure — no bridge at all across K iterations
        return [qstart], [], K

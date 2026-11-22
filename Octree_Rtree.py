#Authors: Kryštof Žďárský, Ondřej Prokop
#3D point cloud indexing, density and curvature estimation
#Three methods: naive search, 3D voxel grid, KD-tree
#AI tools (Claude Sonnet, Gemini) were used for literature search, summarizing sources, debugging, and code optimization


#Import libraries
import heapq
import numpy as np
import time
import matplotlib.pyplot as plt
from rtree import index

#Load point cloud from file
def loadpoints(file):
    
    #Empty lists for three dimensions
    X, Y, Z = [], [], []
    
    #Empty set for points in point cloud
    seen = set()
    
    #Load file
    with open(file) as f:
        for line in f: 
            
            #Every dimension is in its own variable
            x, y, z = map(float, line.split('\t'))
            
            #If there are two points with the same coordinates, only one will be written in the list
            if (x, y, z) not in seen:
                seen.add((x, y, z))
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))
    return X, Y, Z

#________________________________________________________________
#
#                 3D octree implementation
#
#________________________________________________________________

class OctreeNode:
    def __init__(self, center: np.ndarray, half_size: float, depth: int = 0):
        #Cube centre coordinates [cx, cy, cz]
        self.center = center
        #Half of the cube edge length
        self.half_size = half_size
        #Depth of the octree node
        self.depth = depth
        #List of point index and point coordinates
        self.points = []
        #List of 8 children (if there are any)
        self.children = None

    @property
    #Returns True if there are no children 
    def is_leaf(self) -> bool:
        return self.children is None

#3D Octree for spatial indexing and fast k-NN queries
class Octree:
    def __init__(self, points: np.ndarray, max_points: int = 16):
        self.points = np.asarray(points, dtype=np.float64)
        
        #Maximum capacity of points in node
        self.max_points = max_points
        
        #Finding points with minimum and maximum coordinates
        min_pt = np.min(self.points, axis=0)
        max_pt = np.max(self.points, axis=0)
        
        #Defining center of the node
        center = 0.5 * (min_pt + max_pt)
        
        #Cube edge length is defined by the largest extent (1e-9 is there to prevent errors)
        half_size = 0.5 * float(np.max(max_pt - min_pt)) + 1e-9
        
        #Create the tree root node
        self.root = OctreeNode(center, half_size, depth=0)
        
        #Every point in point cloud gets its index
        for idx, pt in enumerate(self.points):
            self._insert(self.root, idx, pt)

    #Every point in node gets its octant index by its position from node center
    def _octant_index(self, center: np.ndarray, pt: np.ndarray) -> int:
        octant = 0
        if pt[0] >= center[0]:
            octant |= 1
        if pt[1] >= center[1]:
            octant |= 2
        if pt[2] >= center[2]:
            octant |= 4
        return octant

    #Octree node with more points than set threshold is subdivided into eight children nodes
    def _subdivide(self, node: OctreeNode):
        node.children = []
        quarter = node.half_size * 0.5
        for i in range(8):
            offset = np.array([
                quarter if (i & 1) else -quarter,
                quarter if (i & 2) else -quarter,
                quarter if (i & 4) else -quarter], dtype=np.float64)
            node.children.append(OctreeNode(node.center + offset, quarter, node.depth + 1))
        
        #Every point falls into one of the eight child nodes by its coordinates
        for idx, pt in node.points:
            octant = self._octant_index(node.center, pt)
            self._insert(node.children[octant], idx, pt)
        node.points = []

    #Recursively inserts point into the appropriate leaf node
    def _insert(self, node: OctreeNode, idx: int, pt: np.ndarray):
        if node.is_leaf:
            #If the leaf is not full, store the point
            if len(node.points) < self.max_points:
                node.points.append((idx, pt))
                return
            #If the leaf is full, subdivide the node
            self._subdivide(node)

        octant = self._octant_index(node.center, pt)
        self._insert(node.children[octant], idx, pt)

    @staticmethod
    #Computes the minimum square Euclidean distance from point to a node's cube
    #If the point is inside the cube, distance is 0.0
    def _point_sq_distance(pt: np.ndarray, center: np.ndarray, half_size: float) -> float:
        dx = np.maximum(0.0, np.abs(pt - center) - half_size)
        return float(np.dot(dx, dx))

    # finds k nearest neighbors of query point
    def query_knn(self, query_pt: np.ndarray, k: int, exclude_idx: int = None):
        
        #Priority queue, that keeps k nearest neighbors
        heap = []

        #If node is a leaf, the square Euclidean distance of every point from query point is calculated
        def _search(node: OctreeNode):
            if node.is_leaf:
                for idx, pt in node.points:
                    #Skip the query point itself
                    if idx == exclude_idx:
                        continue
                    diff = query_pt - pt
                    d_sq = float(np.dot(diff, diff))
                    if len(heap) < k:
                        heapq.heappush(heap, (-d_sq, idx))
                    elif d_sq < -heap[0][0]:
                        heapq.heapreplace(heap, (-d_sq, idx))
                return

            #Compute minimum distances to all non-empty child cubes
            child_dists = []
            for child in node.children:
                if child.is_leaf and not child.points:
                    continue
                d_box = self._point_sq_distance(query_pt, child.center, child.half_size)
                child_dists.append((d_box, child))
                
            #Sort closest child cubes by their square Euclidean distance
            child_dists.sort(key=lambda x: x[0])

            #There is k neighbors and this cube is farther than the k-th neighbor, so it doesn't have to search anymore
            for d_box, child in child_dists:
                if len(heap) == k and d_box >= -heap[0][0]:
                    break
                _search(child)

        #Convert heap contents into arrays of indices and actual distances sorted in ascending order
        _search(self.root)
        results = sorted([(-d_sq, idx) for d_sq, idx in heap], key=lambda x: x[0])
        indices = np.array([idx for _, idx in results], dtype=np.int64)
        distances = np.sqrt(np.array([d_sq for d_sq, _ in results], dtype=np.float64))
        return indices, distances
    
#________________________________________________________________
#
#                 3D Rtree implementation
#
#________________________________________________________________

#Merge two 3D axis-aligned bounding boxes into the smallest box enclosing both
def R_box_merge(box1: tuple, box2: tuple) -> tuple:
    return (np.minimum(box1[0], box2[0]), np.maximum(box1[1], box2[1]))

#Calculate volume of the 3D bounding box
def R_box_volume(box: tuple) -> float:
    sides = np.maximum(0.0, box[1] - box[0])
    return float(sides[0] * sides[1] * sides[2])

#Calculate the volume increase
def R_box_enlargement(box: tuple, other: tuple) -> float:
    return R_box_volume(R_box_merge(box, other)) - R_box_volume(box)

#Minimum square distance from point to bounding box 
def R_box_min_sq_dist(pt: np.ndarray, box: tuple) -> float:
    dx = np.maximum(0.0, np.maximum(box[0] - pt, pt - box[1]))
    return float(np.dot(dx, dx))

#Node in Rtree
class RTreeNode:
    def __init__(self, is_leaf: bool = True):
        self.is_leaf = is_leaf
        self.entries = []
        #Minimum bounding box
        self.bound_box = None

    #Recomputing node's bounding box from all entries
    def recompute_bound_box(self):
        if not self.entries:
            self.bound_box = None
            return
        b = self.entries[0][0]
        for entry_box, _ in self.entries[1:]:
            b = R_box_merge(b, entry_box)
        self.bound_box = b

#3D Rtrre
class RTree:
    def __init__(self, points: np.ndarray, max_entries: int = 16, min_entries: int = None):
        self.points = np.asarray(points, dtype=np.float64)
        self.M = max_entries
        self.m = min_entries if min_entries is not None else max(2, max_entries // 2)
        self.root = RTreeNode(is_leaf=True)
        #Insert points sequentially into the R-tree
        for idx, pt in enumerate(self.points):
            self.insert(idx, pt)

    #Insert point into the Rtree
    def insert(self, idx: int, pt: np.ndarray):
        pt_box = (pt.copy(), pt.copy())
        split_node = self._insert_recursive(self.root, (pt_box, idx))
        #If the root node splits, create a new root node for both bounding boxes
        if split_node is not None:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries = [
                (self.root.bound_box, self.root),
                (split_node.bound_box, split_node)]
            new_root.recompute_bound_box()
            self.root = new_root

    #Locates the best leaf and inserts points into nnode 
    def _insert_recursive(self, node: RTreeNode, entry: tuple):
        if node.is_leaf:
            node.entries.append(entry)
            node.recompute_bound_box()
            if len(node.entries) > self.M:
                return self._quadratic_split(node)
            return None

        #Pick the child whose bounding box requires the least volume enlargement to include the new point
        best_idx = 0
        best_enl = float("inf")
        best_vol = float("inf")
        entry_box = entry[0]
        for i, (child_box, _) in enumerate(node.entries):
            enl = R_box_enlargement(child_box, entry_box)
            vol = R_box_volume(child_box)
            if enl < best_enl or (enl == best_enl and vol < best_vol):
                best_enl = enl
                best_vol = vol
                best_idx = i

        _, chosen_child = node.entries[best_idx]
        split_child = self._insert_recursive(chosen_child, entry)

        #Update bounding box of the child
        node.entries[best_idx] = (chosen_child.bound_box, chosen_child)

        #If the child split, add the created node to this node's entries
        if split_child is not None:
            node.entries.append((split_child.bound_box, split_child))

        node.recompute_bound_box()
        if len(node.entries) > self.M:
            return self._quadratic_split(node)
        return None

    #Divides an overfull node into two nodes and minimize the total volume and overlap of the final bounding boxes
    def _quadratic_split(self, node: RTreeNode) -> RTreeNode:
       
        entries = node.entries
        n = len(entries)
        
        #A pair of bounding boxes which has together the largest wasted space is devided each into opposite group
        max_waste = -1.0
        seed1, seed2 = 0, 1
        for i in range(n):
            vol_i = R_box_volume(entries[i][0])
            for j in range(i + 1, n):
                waste = R_box_volume(R_box_merge(entries[i][0], entries[j][0])) - vol_i - R_box_volume(entries[j][0])
                if waste > max_waste:
                    max_waste = waste
                    seed1, seed2 = i, j

        group1 = [entries[seed1]]
        bound_box1 = entries[seed1][0]
        group2 = [entries[seed2]]
        bound_box2 = entries[seed2][0]

        remaining = [i for i in range(n) if i != seed1 and i != seed2]

        #Iteratively assign the remaining entries to one of the two groups
        while remaining:
            if len(group1) + len(remaining) == self.m:
                for i in remaining:
                    group1.append(entries[i])
                break
            if len(group2) + len(remaining) == self.m:
                for i in remaining:
                    group2.append(entries[i])
                break

            #Select the entry with the strongest preference for one of the groups
            max_diff = -1.0
            chosen_rem_pos = 0
            chosen_group = 1

            for pos, idx in enumerate(remaining):
                b = entries[idx][0]
                d1 = R_box_enlargement(bound_box1, b)
                d2 = R_box_enlargement(bound_box2, b)
                diff = abs(d1 - d2)
                if diff > max_diff:
                    max_diff = diff
                    chosen_rem_pos = pos
                    if d1 < d2:
                        chosen_group = 1
                    elif d2 < d1:
                        chosen_group = 2
                    else:
                        v1, v2 = R_box_volume(bound_box1), R_box_volume(bound_box2)
                        if v1 < v2:
                            chosen_group = 1
                        elif v2 < v1:
                            chosen_group = 2
                        else:
                            chosen_group = 1 if len(group1) <= len(group2) else 2

            idx = remaining.pop(chosen_rem_pos)
            if chosen_group == 1:
                group1.append(entries[idx])
                bound_box1 = R_box_merge(bound_box1, entries[idx][0])
            else:
                group2.append(entries[idx])
                bound_box2 = R_box_merge(bound_box2, entries[idx][0])

        #Retain the first group in the original node and place the second group in a new sibling node
        node.entries = group1
        node.recompute_bound_box()

        sibling = RTreeNode(is_leaf=node.is_leaf)
        sibling.entries = group2
        sibling.recompute_bound_box()
        return sibling
    #Finds the k nearest neighbors in the R-tree using Branch-and-Bound and a max-heap
    def query_knn(self, query_pt: np.ndarray, k: int, exclude_idx: int = None):
        heap = []

        def _search(node: RTreeNode):
            if node.is_leaf:
                for _, idx in node.entries:
                    if idx == exclude_idx:
                        continue
                    diff = query_pt - self.points[idx]
                    d_sq = float(np.dot(diff, diff))
                    if len(heap) < k:
                        heapq.heappush(heap, (-d_sq, idx))
                    elif d_sq < -heap[0][0]:
                        heapq.heapreplace(heap, (-d_sq, idx))
                return

            #Sort child branches in ascending order of their mindist to the query point
            child_dists = [
                (R_box_min_sq_dist(query_pt, child_box), child)
                for child_box, child in node.entries]
            child_dists.sort(key=lambda x: x[0])

            for d_box, child in child_dists:
                if len(heap) == k and d_box >= -heap[0][0]:
                    break
                _search(child)

        #Convert heap contents into arrays of indices and actual distances sorted in ascending order
        _search(self.root)
        results = sorted([(-d_sq, idx) for d_sq, idx in heap], key=lambda x: x[0])
        indices = np.array([idx for _, idx in results], dtype=np.int64)
        distances = np.sqrt(np.array([d_sq for d_sq, _ in results], dtype=np.float64))
        return indices, distances

#________________________________________________________________
#
#                 Library Rtree implementation
#
#________________________________________________________________

class RTreeLbr:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64)

        #Add third dimension to 2D library Rtree
        p = index.Property()
        p.dimension = 3
        self.idx3d = index.Index(properties=p)

        #Insert each point as a degenerate 3D bounding box
        for i, (x, y, z) in enumerate(self.points):
            self.idx3d.insert(i, (x, y, z, x, y, z))

    def query_knn(self, query_pt: np.ndarray, k: int, exclude_idx: int = None):
        qx, qy, qz = float(query_pt[0]), float(query_pt[1]), float(query_pt[2])
        fetch_k = k + 1 if exclude_idx is not None else k

        #Find k nearest neighbors
        nearest_ids = list(self.idx3d.nearest((qx, qy, qz, qx, qy, qz), fetch_k))
        if exclude_idx is not None:
            nearest_ids = [idx for idx in nearest_ids if idx != exclude_idx]

        #Compute exact Euclidean distances, sort, and trim to exactly k neighbors
        candidates = []
        for idx in nearest_ids:
            diff = query_pt - self.points[idx]
            dist = float(np.sqrt(np.dot(diff, diff)))
            candidates.append((dist, idx))
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:k]

        indices = np.array([idx for _, idx in candidates], dtype=np.int64)
        distances = np.array([dist for dist, _ in candidates], dtype=np.float64)
        return indices, distances

#________________________________________________________________
#
#                    Density and curvature
#
#________________________________________________________________

#Compute density 
def compute_density(points: np.ndarray, tree) -> tuple[float, float]:
    nn_dists = np.empty(len(points), dtype=np.float64)
    for i, pt in enumerate(points):
        _, dists = tree.query_knn(pt, k=1, exclude_idx=i)
        nn_dists[i] = dists[0]
    d_aver = float(np.mean(nn_dists))
    rho = 1.0 / (d_aver ** 3)
    return rho, d_aver

#Compute approximated curvature over k=30 nearest neighbors
def compute_curvature(points: np.ndarray, tree, k: int = 30) -> np.ndarray:
    n = len(points)
    curvatures = np.zeros(n, dtype=np.float64)

    #Find 30 nearest neighbors
    for i, pt in enumerate(points):
        idxs, _ = tree.query_knn(pt, k=k)
        neighbors = points[idxs]

        #3X3 covariance matrix of neighborhood's coordinates
        cov = np.cov(neighbors, rowvar=False)
        eigvals = np.linalg.eigh(cov)[0]
        eigvals = np.maximum(eigvals, 0.0)

        eig_sum = float(np.sum(eigvals))
        if eig_sum > 0.0:
            curvatures[i] = eigvals[0] / eig_sum

    return curvatures

#Render 3D point cloud colored by the curvature
def visualize_curvature(points: np.ndarray, curvatures: np.ndarray, title: str):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=curvatures, cmap="viridis", s=6
    )
    plt.colorbar(sc, ax=ax, label="Křivost κ")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

#________________________________________________________________
#
#                       Main function
#
#________________________________________________________________

if __name__ == "__main__":
    X, Y, Z = loadpoints("tree_18.txt")
    pts = np.column_stack((X, Y, Z))

    #Octree
    t0 = time.perf_counter()
    octree = Octree(pts, max_points=16)
    t_build_oct = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_oct, d_aver_oct = compute_density(pts, octree)
    t_dens_oct = time.perf_counter() - t0

    t0 = time.perf_counter()
    curv_oct = compute_curvature(pts, octree, k=30)
    t_curv_oct = time.perf_counter() - t0

    print(f"[Octree] Build: {t_build_oct:.4f} s | Density: {rho_oct:.4f} (d_aver={d_aver_oct:.4f}, {t_dens_oct:.4f} s) | Curvature (k=30): min = {min(curv_oct):.4f}, max = {max(curv_oct):.4f}, {t_curv_oct:.4f} s")

    #Rtree
    t0 = time.perf_counter()
    rtree = RTree(pts, max_entries=16)
    t_build_rt = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_rt, d_aver_rt = compute_density(pts, rtree)
    t_dens_rt = time.perf_counter() - t0

    t0 = time.perf_counter()
    curv_rt = compute_curvature(pts, rtree, k=30)
    t_curv_rt = time.perf_counter() - t0

    print(f"[R-tree] Build: {t_build_rt:.4f} s | Density: {rho_rt:.4f} (d_aver={d_aver_rt:.4f}, {t_dens_rt:.4f} s) | Curvature (k=30): min = {min(curv_rt):.4f}, max = {max(curv_rt):.4f}, {t_curv_rt:.4f} s")

    #Library Rtree
    t0 = time.perf_counter()
    rtree_lbr = RTreeLbr(pts)
    t_build_rt_lbr = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_rt_lbr, d_aver_rt_lbr = compute_density(pts, rtree_lbr)
    t_dens_rt_lbr = time.perf_counter() - t0

    t0 = time.perf_counter()
    curv_rt_lbr = compute_curvature(pts, rtree_lbr, k=30)
    t_curv_rt_lbr = time.perf_counter() - t0

    print(f"[R-tree (library)] Build: {t_build_rt_lbr:.4f} s | Density: {rho_rt_lbr:.4f} (d_aver={d_aver_rt_lbr:.4f}, {t_dens_rt_lbr:.4f} s) | Curvature (k=30): min = {min(curv_rt_lbr):.4f}, max = {max(curv_rt_lbr):.4f}, {t_curv_rt_lbr:.4f} s")

    #Check numerical agreement across all structures
    print(f"Maximum difference in curvature Octree & R-tree: {np.max(np.abs(curv_oct - curv_rt)):.2e} and Octree & R-tree(library): {np.max(np.abs(curv_oct - curv_rt_lbr)):.2e}")

    #Curvature visualization
    visualize_curvature(pts, curv_oct, "Approximated PCA Curvature of point cloud (k = 30)")
#Authors: Kryštof Žďárský, Ondřej Prokop
#3D point cloud indexing, density and curvature estimation
#Three methods: naive search, 3D voxel grid, KD-tree
#AI tools (Claude Sonnet, Gemini) were used for literature search, summarizing sources, debugging, and code optimization

import math
import time
import heapq
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RUN_NAIVE = True #kill switch for naive search (shortens the whole run of code)

#loading points - Loads 3D point cloud points separated by tab and filters out the duplicates
def loadpoints(file):
    X, Y, Z = [], [], []
    seen = set()
    with open(file) as f:
        for line in f: 
            x, y, z = map(float, line.split('\t'))
            if (x, y, z) not in seen:
                seen.add((x, y, z))
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))
    return X, Y, Z

#naive search - Brute-force search excluding the query point itself 
def getNN(xq, yq, zq, X, Y, Z):
    dmin = math.inf
    xn, yn, zn, = X[0], Y[0], Z[0]
    
    for i in range(len(X)):
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx*dx + dy*dy + dz*dz)**0.5
        
        if d < dmin and d > 0:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
    return xn, yn, zn, dmin

#compute point cloud density (rho = 1 / d_aver^3) using naive O(N^2) search
def computeDensityNaive(X, Y, Z):
    dist = []
    for i in range(len(X)):
        xn, yn, zn, dmin = getNN(X[i], Y[i], Z[i], X, Y, Z)
        dist.append(dmin)
    d_aver =  sum(dist)/ len(dist)
    rho = 1 / d_aver**3
    return rho, d_aver, dist
    
def drawVoxels(x_min, y_min, z_min, dx, dy, dz, V):
    n_r = V.shape[0]
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_aspect('equal')
    ax.set_box_aspect((dx, dy, dz))
    
    xedges = np.linspace(x_min, x_min + dx, n_r+1)
    yedges = np.linspace(y_min, y_min + dy, n_r+1)
    zedges = np.linspace(z_min, z_min + dz, n_r+1)
    VX, VY, VZ = np.meshgrid(xedges, yedges, zedges, indexing="ij")
    
    ax.voxels(VX, VY, VZ, V, edgecolor='k')
    plt.show()

def init_spat_index(X, Y, Z, n_r):
    min_x, min_y, min_z = min(X), min(Y), min(Z)
    dx = max(X) - min_x
    dy = max(Y) - min_y
    dz = max(Z) - min_z
    
    return min_x, min_y, min_z, dx, dy, dz

#3D voxel cell index
def get3Dindex(x, y, z, dx, dy, dz, min_x, min_y, min_z, n_r):
    c = 0.99 #Scaling factor < 1 ensures max boundary points stay within [0, n_r - 1]
    ix = int(c * n_r * (x - min_x)/dx)
    iy = int(c * n_r * (y - min_y)/dy)
    iz = int(c * n_r * (z - min_z)/dz)
    return ix, iy, iz

#Flatten 3D voxel coordinates (ix, iy, iz) into a 1D hash key
def hash(ix, iy, iz, n_r):
    return ix + iy * n_r + iz * n_r * n_r

#Convert 1D hash key back to 3D voxel coordinates
def unhash(j, n_r):
    ix = j % n_r
    iy = (j // n_r)% n_r
    iz = j // (n_r * n_r)
    return ix, iy, iz

def voxelize(X, Y, Z, dx, dy, dz, min_x, min_y, min_z, n_r):
    V = np.zeros((n_r, n_r, n_r), dtype=bool)
    for x, y, z in zip(X, Y, Z):
        ix, iy, iz = get3Dindex(x, y, z, dx, dy, dz, min_x, min_y, min_z, n_r)
        V[ix, iy, iz] = True
    return V

#Build spatial hash table mapping voxel keys to point indices
def createSpatialIDXs(X, Y, Z, dx, dy, dz, min_x, min_y, min_z, n_r):
    H = defaultdict(list)
    J = []
    for i in range (len(X)):
        ix, iy, iz = get3Dindex(X[i], Y[i], Z[i], dx, dy, dz, min_x, min_y, min_z, n_r)
        j = hash(ix, iy, iz, n_r)
        H[j].append(i)
        J.append(j)
    return H, J

#Retrieve point indices from the target voxel and its neighborhood within radius r
def cellCandidates(H, j, n_r, r):
    ix, iy, iz = unhash(j, n_r)
    out = []
    for a in range (max(ix - r, 0), min(ix + r, n_r - 1) + 1):
        for b in range (max(iy - r, 0), min(iy + r, n_r - 1) + 1):
            for c in range (max(iz - r, 0), min(iz + r, n_r - 1) + 1):
                out.extend(H.get(hash(a, b, c, n_r), [])) 
    return out

def computeDensity(X, Y, Z, idxs, min_h, neighbors=False):
    H, J, n_r = idxs
    dist = []
    
    for i in range(len(X)):
        r = 1 if neighbors else 0
        dmin = math.inf
        while True:
            for k in cellCandidates(H, J[i], n_r, r):
                dx, dy, dz = X[i] - X[k], Y[i] - Y[k], Z[i] - Z[k]
                d = (dx*dx + dy*dy + dz*dz)**0.5
                if 0 < d < dmin:
                    dmin = d
            if (dmin <= r * min_h) or r > n_r: #Stopping criterion: guarantees no closer point exists outside the current search shell
                break
            r += 1
            
        if dmin < math.inf:
            dist.append(dmin)
    d_aver = sum(dist)/len(dist)
    rho = 1/(d_aver**3)
    return rho, d_aver, dist

#Estimate local surface curvature (surface variation) via PCA
def pcaCurvature(neigh):
    P = np.array(neigh)
    C = P - P.mean(axis=0)
    cov = C.T @ C / len(P)
    w = np.linalg.eigvalsh(cov)
    S = w.sum()
    return w[0] / S if S > 0 else 0.0

#estimates curvature for each point using k-NN retrieved via expanding voxel shells
def computeCurvature(X, Y, Z, idxs, min_h, k=30, neighbors=True):
    H, J, n_r = idxs
    kappa = []
    
    for i in range(len(X)):
        r = 1 if neighbors else 0
        while True:
            cand = cellCandidates(H, J[i], n_r, r)
            if len(cand) >= k:
                d2 = sorted(((X[c] - X[i])**2 + (Y[c] - Y[i])**2 + (Z[c] - Z[i])**2, c) 
                    for c in cand)
                if d2[k - 1][0] <= (r * min_h) ** 2 or r > n_r:
                    break
            elif r > n_r:
                break
            r += 1
        
        neigh = [(X[c], Y[c], Z[c]) for _, c in d2[:k]]
        kappa.append(pcaCurvature(neigh))
    return kappa

class KDNode:
    
    def __init__(self, point=None, axis=None, left=None, right=None, point_indices=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
        self.point_indices = point_indices
        
#Balanced 3D KD-Tree with leaf buckets for fast k-NN queries         
class KDTree:
    
    def __init__(self, data, leaf_size = 16):
        self.data = np.array(data)
        self.pts = self.data.tolist()
        self.leaf_size = leaf_size
        self.root = self._build_tree(list(range(len(self.pts))),0)
        
    def _build_tree(self, idxs, depth):
        
        if len(idxs)<= self.leaf_size or depth >= 50:
            return KDNode(point_indices=idxs)
        
        axis = depth % 3 #Cycle through X, Y, Z axes and split at the median point
        idxs = sorted(idxs, key=lambda i: self.pts[i][axis])
        m = len(idxs) // 2
        
        node = KDNode(point=self.pts[idxs[m]], axis=axis)
        node.left = self._build_tree(idxs[:m], depth + 1)
        node.right = self._build_tree(idxs[m:], depth + 1)
        return node
    
    def _search(self, node, q, k, heap, skip_zero):
        
        if node.point_indices is not None:
            for i in node.point_indices:
                p = self.pts[i]
                d2 = (p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2
                if skip_zero and d2 == 0:
                    continue
                if len(heap) < k:
                    heapq.heappush(heap, (-d2, i)) #Use a max-heap (storing negative squared distances) to track the k closest points
                elif d2 < -heap[0][0]:
                    heapq.heapreplace(heap, (-d2, i))
            return
        
        diff = q[node.axis] - node.point[node.axis]
        if diff < 0: 
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left
        
        self._search(near, q, k, heap, skip_zero)
        
        if len(heap) < k or diff * diff < -heap[0][0]: #Pruning: search the far branch only if the splitting plane is closer than the k-th neighbor
            self._search(far, q, k, heap, skip_zero)
               
    def query(self, q, k, skip_zero=False):
        heap = []
        self. _search(self.root, q, k, heap, skip_zero)
        result = sorted((-nd, i) for nd, i in heap)
        return [(math.sqrt(d2), i) for d2, i in result]

#Compute density using KD-tree
def computeDensityKD(tree):
    dist = []
    for q in tree.pts:
        d, i = tree.query(q, 1, skip_zero=True)[0]
        dist.append(d)
    d_aver = sum(dist)/len(dist)
    rho = 1 / d_aver**3
    return rho, d_aver, dist

#Compute curvature using KD-tree
def computeCurvatureKD(tree, k=30):
    kappa = []
    for q in tree.pts:
        res = tree.query(q, k)
        neigh = [tree.pts[i] for d, i in res]
        kappa.append(pcaCurvature(neigh))
    return kappa

def drawCurvature(X, Y, Z, kappa, dx, dy, dz, title, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X, Y, Z, c=kappa, s=1, cmap="magma_r")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Approximated curvature")
    ax.set_title(title)
    ax.set_box_aspect((dx, dy, dz))
    plt.savefig(filename, dpi=150)
    plt.show()       
            
#hlavni funkce
def main():
    X, Y, Z = loadpoints("tree_18.txt")
    n_r = 30
    
    results = {}
    
    #naive_search
    if RUN_NAIVE:
        t0 = time.perf_counter()
        rho, d_aver, _ = computeDensityNaive(X, Y, Z)
        t = time.perf_counter() - t0
        print(f"Density (naive): rho = {rho:.2f}, d_aver = {d_aver:.5f} m, {t:.2f} s")
        results["Naive"] = (rho, d_aver, t, None)
        
    #voxels
    t0 = time.perf_counter()
    min_x, min_y, min_z, dx, dy, dz = init_spat_index(X, Y, Z, n_r)
    H, J = createSpatialIDXs(X, Y, Z, dx, dy, dz, min_x, min_y, min_z, n_r)
    idxs = (H, J, n_r)
    min_h = min(dx, dy, dz) / (0.99 * n_r)
    print(f"Voxel index made in: {time.perf_counter() - t0:.2f}s")
    

    t0 = time.perf_counter()
    rho, d_aver, _ = computeDensity(X, Y, Z, idxs, min_h, neighbors=False)
    t = time.perf_counter() - t0
    print(f"Density (voxels): rho = {rho:.2f}, d_aver = {d_aver:.5f} m, {t:.2f} s")

    t0 = time.perf_counter()
    kappa = computeCurvature(X, Y, Z, idxs, min_h, neighbors=False)
    t_curv = time.perf_counter() - t0
    print(f"Curvature (voxels): min = {min(kappa):.4f}, max = {max(kappa):.4f}, {t_curv:.2f} s")
    results["Voxels"] = (rho, d_aver, t, t_curv)

    #KD-tree
    t0 = time.perf_counter()
    tree = KDTree(list(zip(X, Y, Z)))
    print(f"KD-tree made in: {time.perf_counter() - t0:.2f} s")
    
    t0 = time.perf_counter()
    rho, d_aver, _ = computeDensityKD(tree)
    t = time.perf_counter() - t0
    print(f"Density (kd-tree): rho = {rho:.2f}, d_aver = {d_aver:.5f} m, {t:.2f} s")
    
    t0 = time.perf_counter()
    kappa_kd = computeCurvatureKD(tree)
    t_curv = time.perf_counter() - t0
    print(f"Curvature (kd-tree): min = {min(kappa_kd):.2f}, max = {max(kappa_kd):.4f}, {t_curv:.2f} s")    
    results["KD-tree"] = (rho, d_aver, t, t_curv)
    
    #voxel curvature differend to KD-tree curvature
    diff = max(abs(a - b) for a, b in zip(kappa, kappa_kd))
    print(f"Difference voxel vs KD-tree curvature is: {diff:.4f}")
    
    #tabs
    print()
    print(f"{'Method':28s}{'rho':>10s}{'d_aver [m]':>13s}{'Density [s]':>13s}{'Curvature [s]':>13s}")
    for name, (rho, d_aver, t_d, t_c) in results.items():
        t_c_text = "-" if t_c is None else f"{t_c:.2f}"
        print(f"{name:28s}{rho:10.2f}{d_aver:13.5f}{t_d:13.2f}{t_c_text:>13s}")
    
    #pictures
    drawCurvature(X, Y, Z, kappa, dx, dy, dz, "Curvature - voxels", "curvature_voxels.png")
    drawCurvature(X, Y, Z, kappa_kd, dx, dy, dz, "Curvature - kd-tree", "curvature_kdtree.png")
    

    #Graph1:
    print("\nCalculating time depending on size of input")
    g = random.Random(42) #Benchmark 1: Execution time vs. point cloud size N (fixed seed for reproducibility)
    rm = list(range(len(X)))
    g.shuffle(rm)
    X_shuf = [X[i] for i in rm]
    Y_shuf = [Y[i] for i in rm]
    Z_shuf = [Z[i] for i in rm]
    
    sizes_fast = [500, 1000, 2500, 5000, 10000, 15000, 20000, len(X)]
    sizes_fast = [s for s in sizes_fast if s <= len(X)]
    sizes_naive = [s for s in sizes_fast if s <= 10000]
    
    t_naive_N = []
    t_vox_dens_N, t_vox_curv_N = [], []
    t_KD_dens_N, t_KD_curv_N = [], []
    
    for s in sizes_naive:
        t0 = time.perf_counter()
        computeDensityNaive(X_shuf[:s], Y_shuf[:s], Z_shuf[:s])
        t_naive_N.append(time.perf_counter() - t0)
    
    for s in sizes_fast:
        Xs, Ys, Zs = X_shuf[:s], Y_shuf[:s], Z_shuf[:s]
        
        #voxelize
        t0 = time.perf_counter()
        mx, my, mz, sdx, sdy, sdz = init_spat_index(Xs, Ys, Zs, n_r)
        sH, sJ = createSpatialIDXs(Xs, Ys, Zs, sdx, sdy, sdz, mx, my, mz, n_r)
        s_min_h = min(sdx, sdy, sdz)/(0.99*n_r)
        computeDensity(Xs, Ys, Zs, (sH, sJ, n_r), s_min_h, neighbors=False)
        t_vox_dens_N.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        computeCurvature(Xs, Ys, Zs, (sH, sJ, n_r), s_min_h, neighbors=False)
        t_vox_curv_N.append(time.perf_counter() - t0)
        
        #KD-tree
        t0 = time.perf_counter()
        itree =  KDTree(list(zip(Xs, Ys, Zs)))
        computeDensityKD(itree)
        t_KD_dens_N.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        computeCurvatureKD(itree)
        t_KD_curv_N.append(time.perf_counter() - t0)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5.5))
    ax1.plot(sizes_naive, t_naive_N, color = "tab:blue", label = "1. Naive search")
    ax1.plot(sizes_fast, t_vox_dens_N, color = "tab:orange", label = f"2. Voxelize (n_r = {n_r})")
    ax1.plot(sizes_fast, t_KD_dens_N, color = "tab:green", label = "3. KD-Tree")
    ax1.set_title("Density (1-NN): time vs number of points N")
    ax1.set_xlabel("Number of points N")
    ax1.set_ylabel("Calculation time [s]")
    ax1.legend()
    ax1.grid(True, linestyle = "--", alpha = 0.5)
    
    ax2.plot(sizes_fast, t_vox_curv_N, color = "tab:orange", label = f"2. Voxelize ( n_r {n_r})")
    ax2.plot(sizes_fast, t_KD_curv_N, color = "tab:green", label = "3. KD-tree")
    ax2.set_title("Curvature (30-NN PCA): time vs number of points N")
    ax2.set_xlabel("Number of points N")
    ax2.set_ylabel("Calculation time [s]")
    ax2.legend()
    ax2.grid(True, linestyle = "--", alpha = 0.5)
    
    plt.tight_layout()
    plt.savefig("time_vs_N.png", dpi = 300)
    plt.show()
    
    #Graph2: 
    print("Calculating dependence of calculation time on voxel size")
    nr_values = [15, 25, 35, 50, 75, 100, 130] #Benchmark 2: Voxelization performance vs. grid resolution (n_r) and voxel edge size
    voxelsize_min = []
    t_voxelsize_dens = []
    t_voxelsize_curv = []
    
    for n in nr_values:
        t0 = time.perf_counter()
        mx, my, mz, sdx, sdy, sdz = init_spat_index(X, Y, Z, n)
        tH, tJ = createSpatialIDXs(X, Y, Z, sdx, sdy, sdz, mx, my, mz, n)
        t_min_h = min(sdx, sdy, sdz) / (0.99 * n)
        computeDensity(X, Y, Z, (tH, tJ, n), t_min_h, neighbors=False)
        t_d = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        computeCurvature(X, Y, Z, (tH, tJ, n), t_min_h, neighbors=False)
        t_c = time.perf_counter() - t0
        
        voxelsize_min.append(t_min_h)
        t_voxelsize_dens.append(t_d)
        t_voxelsize_curv.append(t_c)
        print(f"  n_r = {n:3d} (h_min = {t_min_h:.3f} m) -> density: {t_d:.2f} s, curvature: {t_c:.2f} s")
    
    plt.figure(figsize = (8.5, 5.5))
    plt.plot(voxelsize_min, t_voxelsize_dens, color = "tab:orange", label = "Density (1NN + index)")
    plt.plot(voxelsize_min, t_voxelsize_curv, color = "tab:green", label = "Curvature (30-NN PCA)")
    plt.title("Voxelization: dependence computation time on voxel edge")
    plt.xlabel("Minimal voxel edge h_min [m]")
    plt.ylabel("Calculation time [s]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha = 0.5)
    plt.tight_layout()
    plt.savefig("time_vs_voxel_size.png", dpi = 300)
    plt.show()
          
    V = voxelize(X, Y, Z, dx, dy, dz, min_x, min_y, min_z, n_r)
    drawVoxels(min_x, min_y, min_z, dx, dy, dz, V)
    
if __name__ == "__main__":
    main()
    
    
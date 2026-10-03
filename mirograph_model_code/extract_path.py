import os
import argparse
import pickle
import networkx as nx
from collections import deque

def is_induced_path(G, path):
    if len(path) <= 1:
        return True
    
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return False
    
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if G.has_edge(path[i], path[j]):
                return False
    
    return True

def bfs_distance_from_t(adj, t):
    
    dist = {}
    queue = deque([t])
    dist[t] = 0
    
    while queue:
        u = queue.popleft()
        for w in adj.get(u, []):
            if w not in dist:
                dist[w] = dist[u] + 1
                queue.append(w)
    
    return dist

def enumerate_induced_paths(G, s, t, L, max_p=10):
    if s not in G:
        return None, None, f"Source '{s}' not found in graph"
    if t not in G:
        return None, None, f"Target '{t}' not found in graph"
    
    adj = {}
    for node in G.nodes():
        adj[node] = list(G.neighbors(node))
    
    dist = bfs_distance_from_t(adj, t)
    
    if s not in dist:
        return [], None, f"No path exists from '{s}' to '{t}'"
    
    all_paths = []
    
    def dfs(path):
        nonlocal all_paths
        
        if len(all_paths) >= max_p:
            return
        
        u = path[-1]
        length = len(path) - 1
        
        if length > L:
            return
        
        if u == t:
            all_paths.append(list(path))
            return
        
        if dist.get(u, float('inf')) + length > L:
            return
        
        for w in adj.get(u, []):
            if w in path:
                continue
            
            if w == t:
                if length + 1 <= L:
                    ok = True
                    for x in path[:-1]:
                        if x in adj.get(w, []):
                            ok = False
                            break
                    
                    if ok:
                        path.append(w)
                        all_paths.append(list(path))
                        path.pop()
                        if len(all_paths) >= max_p:
                            return
                continue
            
            if length + 1 > L:
                continue
            
            ok = True
            for x in path[:-1]:
                if x in adj.get(w, []):
                    ok = False
                    break
            
            if not ok:
                continue
            
            path.append(w)
            dfs(path)
            path.pop()
    
    dfs([s])
    
    if not all_paths:
        return [], None, "No induced paths found"
    
    subgraph_nodes = set()
    for path in all_paths:
        subgraph_nodes.update(path)
    G_sub = G.subgraph(subgraph_nodes).copy()
    
    return all_paths, G_sub, "Success"

def extract_enclosing_subgraph(base_dir, u, v, year=2021, max_k=3, max_p=10, output_path=None):
    print(f"\n[1] Loading graph data for year {year}...")
    try:
        G, id_to_concept = graph_module.load_graph_data(base_dir, year)
        print(f"✅ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"❌ Graph loading error: {e}")
        return None, None, f"Graph loading failed: {e}"
    
    if u not in G:
        return None, None, f"Source node '{u}' not found in graph"
    if v not in G:
        return None, None, f"Target node '{v}' not found in graph"
    
    print(f"\n[2] Extracting induced paths and enclosing subgraph: '{u}' <-> '{v}'")
    print(f"   Parameters: max_k={max_k}, max_p={max_p}")
    print(f"   Algorithm: Induced path exploration (DFS)")
    
    paths, G_sub, msg = enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    
    if paths is None or G_sub is None:
        print(f"❌ {msg}")
        return paths, G_sub, msg
    
    print(f"\n[3] Results:")
    print(f"   - Found {len(paths)} induced path(s)")
    print(f"   - Subgraph nodes: {G_sub.number_of_nodes()}")
    print(f"   - Subgraph edges: {G_sub.number_of_edges()}")
    
    if paths:
        print(f"\n   Induced paths found:")
        for i, path in enumerate(paths, 1):
            print(f"     Path {i}: {' -> '.join(path)}")
            is_valid = is_induced_path(G, path)
            if not is_valid:
                print(f"            ✗ Invalid induced path!")
                for idx_i in range(len(path)):
                    for idx_j in range(idx_i + 2, len(path)):
                        if G.has_edge(path[idx_i], path[idx_j]):
                            print(f"               Violation: {path[idx_i]} <-> {path[idx_j]} (distance={idx_j-idx_i})")
            else:
                print(f"            ✓ Valid induced path")
    
    print(f"\n   Subgraph nodes: {', '.join(sorted(G_sub.nodes()))}")
    print(f"\n   Subgraph edges:")
    for edge in sorted(G_sub.edges(data=True)):
        u_edge, v_edge, data = edge
        weight = data.get('citation_weight', 0)
        print(f"     {u_edge} <-> {v_edge} [weight={weight}]")
    
    if output_path:
        print(f"\n[4] Saving subgraph to: {output_path}")
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            if output_path.endswith('.graphml'):
                nx.write_graphml(G_sub, output_path)
            elif output_path.endswith('.gpickle') or output_path.endswith('.pkl'):
                with open(output_path, 'wb') as f:
                    pickle.dump(G_sub, f)
            else:
                output_path = output_path + '.graphml' if '.' not in output_path else output_path
                nx.write_graphml(G_sub, output_path)
            
            print(f"✅ Subgraph saved successfully")
            
            path_info_path = output_path.rsplit('.', 1)[0] + '_paths.txt'
            with open(path_info_path, 'w', encoding='utf-8') as f:
                f.write(f"Source: {u}\n")
                f.write(f"Target: {v}\n")
                f.write(f"Year: {year}\n")
                f.write(f"Number of paths: {len(paths)}\n")
                f.write(f"Subgraph nodes: {G_sub.number_of_nodes()}\n")
                f.write(f"Subgraph edges: {G_sub.number_of_edges()}\n\n")
                f.write("Paths:\n")
                for i, path in enumerate(paths, 1):
                    f.write(f"  Path {i}: {' -> '.join(path)}\n")
            print(f"✅ Path info saved to: {path_info_path}")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
    
    return paths, G_sub, msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=str, required=True, help="Source concept name")
    parser.add_argument("--v", type=str, required=True, help="Target concept name")
    parser.add_argument("--year", type=int, default=2021, help="Graph data year (default: 2021)")
    parser.add_argument("--max_k", type=int, default=5, help="Maximum path length (k-hop) (default: 3)")
    parser.add_argument("--max_p", type=int, default=10, help="Maximum number of paths to find (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Output file path for subgraph (supports .graphml, .gpickle, .pkl)")
    
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths, G_sub, msg = extract_enclosing_subgraph(
        base_dir=base_dir,
        u=args.u,
        v=args.v,
        year=args.year,
        max_k=args.max_k,
        max_p=args.max_p,
        output_path=args.output
    )

    print(paths)

if __name__ == "__main__":
    exit(main())


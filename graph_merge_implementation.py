"""
Simple graph merger - just append nodes and edges from all graphs
"""

import json
import os
from pathlib import Path

def merge_graphs(graph_dir: str = './graphs', 
                output_path: str = './merged_graph.json'):
    """
    Load all JSON graphs from directory and append nodes + edges.
    """
    merged_nodes = []
    merged_edges = []
    metadata_list = []
    
    # Get all JSON files
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.json')])
    
    for filename in graph_files:
        filepath = os.path.join(graph_dir, filename)
        print(f"Loading {filename}...")
        
        with open(filepath, 'r') as f:
            graph = json.load(f)
            
            # Append nodes
            merged_nodes.extend(graph.get('nodes', []))
            
            # Append edges
            merged_edges.extend(graph.get('edges', []))
            
            # Track metadata
            if 'metadata' in graph:
                metadata_list.append(graph['metadata'])
            
            print(f"  +{len(graph.get('nodes', []))} nodes, +{len(graph.get('edges', []))} edges")
    
    # Create merged graph
    merged_graph = {
        'metadata': {
            'title': 'Merged Knowledge Graph',
            'source_graphs': len(graph_files),
            'total_nodes': len(merged_nodes),
            'total_edges': len(merged_edges),
            'source_titles': [m.get('title', 'Unknown') for m in metadata_list]
        },
        'nodes': merged_nodes,
        'edges': merged_edges
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(merged_graph, f, indent=2)
    
    print(f"\n✓ Merged {len(graph_files)} graphs")
    print(f"  Total nodes: {len(merged_nodes)}")
    print(f"  Total edges: {len(merged_edges)}")
    print(f"  Saved to: {output_path}")
    
    return merged_graph


if __name__ == '__main__':
    merge_graphs(graph_dir='./graphs', output_path='./merged_graph.json')
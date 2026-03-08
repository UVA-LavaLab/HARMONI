import os
import json
import re
from typing import Any, Dict, List, Tuple
from enum import Enum

import networkx as nx


def _is_simple_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _safe_attr_value(value: Any, max_len: int = 200, max_depth: int = 2, visited: set = None) -> Any:
    """
    Safely convert a value to a JSON-serializable format.
    Prevents infinite recursion with depth limiting and visited tracking.
    """
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion with object identity tracking
    obj_id = id(value)
    if obj_id in visited:
        return f"<circular reference: {type(value).__name__}>"
    
    if _is_simple_value(value):
        return value
    if value is None:
        return None
    
    # Handle Enum types - convert to their name
    if isinstance(value, Enum):
        # It's an Enum, return its name
        return value.name
    
    # Depth limit to prevent excessive recursion
    if max_depth <= 0:
        return f"<max depth reached: {type(value).__name__}>"
    
    # Handle lists/arrays
    if isinstance(value, (list, tuple)):
        visited.add(obj_id)
        try:
            result = [_safe_attr_value(item, max_len, max_depth - 1, visited) for item in value[:20]]  # Limit list size to reduce memory
            if len(value) > 20:
                result.append(f"... ({len(value) - 20} more items)")
            return result
        finally:
            visited.discard(obj_id)
    
    # Handle dictionaries
    if isinstance(value, dict):
        visited.add(obj_id)
        try:
            result = {}
            for k, v in list(value.items())[:20]:  # Limit dict size to reduce memory
                if _is_simple_value(k) or isinstance(k, str):
                    result[str(k)] = _safe_attr_value(v, max_len, max_depth - 1, visited)
            if len(value) > 20:
                result["..."] = f"({len(value) - 20} more keys)"
            return result
        finally:
            visited.discard(obj_id)
    
    # Handle objects (like HarmoniTensor) - convert to dict
    # Skip Enum types (already handled above)
    if hasattr(value, '__dict__') and not isinstance(value, Enum):
        # Skip complex objects that are likely to be large (LogicUnit, DRAM, etc.)
        type_name = type(value).__name__
        if type_name in ['LogicUnit', 'DRAMConfig', 'MemorySystem', 'BaseLogicNode', 'Root', 'Channel', 'Rank', 'Chip', 'Bankgroup', 'Bank']:
            return f"<{type_name} object>"
        
        visited.add(obj_id)
        try:
            obj_dict = {}
            # Only get attributes from __dict__ to avoid getting too many attributes
            if hasattr(value, '__dict__'):
                for attr, attr_value in list(value.__dict__.items())[:15]:  # Limit attributes to reduce memory
                    if not attr.startswith('_') and not callable(attr_value):
                        try:
                            # Skip instruction_queue, connections, mapped_task, dram, and other large objects
                            if attr in ['instruction_queue', 'connections', 'mapped_task', 'dram', 'children', 'parents', 'all_nodes', 'channels']:
                                obj_dict[attr] = "..."
                            # Check if attr_value is an Enum before serializing
                            elif isinstance(attr_value, Enum):
                                obj_dict[attr] = attr_value.name
                            # Skip complex objects that are likely to be large
                            elif hasattr(attr_value, '__dict__') and not isinstance(attr_value, Enum):
                                attr_type_name = type(attr_value).__name__
                                if attr_type_name in ['LogicUnit', 'DRAMConfig', 'MemorySystem', 'BaseLogicNode']:
                                    obj_dict[attr] = f"<{attr_type_name} object>"
                                else:
                                    obj_dict[attr] = _safe_attr_value(attr_value, max_len, max_depth - 1, visited)
                            else:
                                obj_dict[attr] = _safe_attr_value(attr_value, max_len, max_depth - 1, visited)
                        except Exception:
                            pass
            # Also check for common HarmoniTensor attributes
            for attr in ['name', 'tag', 'precision', 'shape', 'stride', 'addr_offset', 
                        'chip_idx', 'numel', 'size', 'mapping', 'requests', 'locations',
                        'row_accesses', 'col_accesses', 'dram']:
                if attr not in obj_dict and hasattr(value, attr):
                    try:
                        # Skip dram field, just show "..."
                        if attr == 'dram':
                            obj_dict[attr] = "..."
                        else:
                            attr_value = getattr(value, attr, None)
                            if not callable(attr_value):
                                if isinstance(attr_value, Enum):
                                    obj_dict[attr] = attr_value.name
                                else:
                                    obj_dict[attr] = _safe_attr_value(attr_value, max_len, max_depth - 1, visited)
                    except Exception:
                        pass
            return obj_dict if obj_dict else f"<{type(value).__name__}>"
        except Exception as e:
            return f"<{type(value).__name__}: error serializing>"
        finally:
            visited.discard(obj_id)
    
    # Fallback to string representation
    try:
        text = str(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def _collect_until_token_one(graph: nx.DiGraph) -> Tuple[List[str], List[Tuple[str, str]]]:
    nodes: List[str] = []
    edges: List[Tuple[str, str]] = []
    cutoff_regex = re.compile(r"^decode_LMHead_B_T1\b")

    encountered: set = set()
    for node in nx.topological_sort(graph):
        nodes.append(node)
        encountered.add(node)
        if cutoff_regex.match(str(node)):
            break

    # collect edges among included nodes
    included = set(nodes)
    for u, v in graph.edges():
        if u in included and v in included:
            edges.append((u, v))

    return nodes, edges


def _filter_heads(graph: nx.DiGraph, nodes: List[str]) -> List[str]:
    """
    Filter nodes to only include:
    - Nodes with head == -1 (no head assignment)
    - Nodes with head < num_heads/4 and head < kv_heads/4
    """
    # Find maximum head value to determine num_heads and kv_heads
    max_head = -1
    for node in nodes:
        head = graph.nodes[node].get('head', -1)
        if head != -1 and head > max_head:
            max_head = head
    
    # If no heads found, return all nodes
    if max_head == -1:
        return nodes
    
    # Calculate thresholds (assuming max_head represents kv_heads, and num_heads might be larger)
    # For safety, we'll use max_head as kv_heads and assume num_heads >= kv_heads
    kv_heads = max_head + 1  # heads are 0-indexed
    num_heads = kv_heads  # Default assumption, will filter based on kv_heads/4
    
    # Calculate quarter thresholds
    kv_heads_quarter = kv_heads // 4
    num_heads_quarter = kv_heads // 4  # Using same for now, can be adjusted if needed
    
    # Filter nodes
    filtered_nodes = []
    for node in nodes:
        head = graph.nodes[node].get('head', -1)
        if head == -1:
            # Include nodes without head assignment
            filtered_nodes.append(node)
        elif head < kv_heads_quarter:
            # Include nodes with head < kv_heads/4
            filtered_nodes.append(node)
    
    return filtered_nodes


def _compute_topo_levels(graph: nx.DiGraph, nodes: List[str]) -> Dict[str, int]:
    level: Dict[str, int] = {}
    for n in nodes:
        preds = [p for p in graph.predecessors(n) if p in nodes]
        if not preds:
            level[n] = 0
        else:
            level[n] = max(level[p] for p in preds) + 1
    return level


def _elements_for_cytoscape(graph: nx.DiGraph, nodes: List[str], edges: List[Tuple[str, str]]):
    elements = []

    # Compute deterministic topological positions (levels as rows)
    level = _compute_topo_levels(graph, nodes)
    levels_to_nodes: Dict[int, List[str]] = {}
    for n in nodes:
        levels_to_nodes.setdefault(level[n], []).append(n)

    # Calculate spacing based on max nodes per level to prevent label overlap
    max_nodes_per_level = max(len(lvl_nodes) for lvl_nodes in levels_to_nodes.values()) if levels_to_nodes else 1
    # Increase spacing to accommodate labels (label max-width is 120px, plus padding and margins)
    # Use larger spacing when there are many nodes per level to prevent overlap
    # Reduced by 15% to make edges shorter
    base_x_spacing = 238  # Base spacing for nodes (280 * 0.85, reduced by 15%)
    label_width = 120  # Max label width from text-max-width
    label_padding = 30  # Additional padding for label background and margins
    # Ensure labels have plenty of space - add extra buffer for overlap prevention
    x_spacing = max(base_x_spacing, int((label_width + label_padding + 100) * 0.85))  # Reduced by 15%
    y_spacing = 153  # Reduced from 180 by 15% (180 * 0.85)
    positions: Dict[str, Dict[str, float]] = {}
    for lvl, lvl_nodes in levels_to_nodes.items():
        for idx, n in enumerate(lvl_nodes):
            positions[n] = {"x": idx * x_spacing, "y": lvl * y_spacing}

    # Build node elements with simplified attributes for display panel
    import sys
    total_nodes = len(nodes)
    for idx, n in enumerate(nodes):
        if idx % 100 == 0 and idx > 0:
            print(f"Processing node {idx}/{total_nodes}...", file=sys.stderr, flush=True)
        attrs: Dict[str, Any] = graph.nodes[n]
        simple_attrs: Dict[str, Any] = {}
        for k, v in attrs.items():
            simple_attrs[k] = _safe_attr_value(v)

        # store label for details; hide via style until high zoom
        label = str(n)
        tag = simple_attrs.get("tag")
        ntype = simple_attrs.get("type")
        base = str(tag or ntype or "default")
        hue = (abs(hash(base)) % 360)
        color = f"hsl({hue}, 65%, 60%)"

        node_data = {
            "id": str(n),
            "label": label,
            "attrs": simple_attrs,
            "color": color,
        }
        elements.append({"data": node_data, "position": positions[n]})

    # Build edge elements
    for u, v in edges:
        elements.append({"data": {"id": f"{u}->{v}", "source": str(u), "target": str(v)}})

    return elements


def dump_partial_dfg_html(model_dfg, output_html_path: str = "outputs/dfg_partial_token1.html") -> str:
    """
    Create an interactive HTML visualization (Cytoscape.js) of a partial DFG containing
    nodes up to and including the first occurrence of decode_LMHead_B_T1 in topological order.
    Only shows nheads/4 and kvheads/4 (first quarter of heads).

    Interactions:
      - Fit to screen
      - Zoom in/out; labels appear only when sufficiently zoomed in
      - Click a node to view all its attributes in a side panel

    Returns the absolute path to the generated HTML file.
    """
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    graph: nx.DiGraph = model_dfg.graph
    nodes, edges = _collect_until_token_one(graph)
    
    # Filter to only show nheads/4 and kvheads/4
    filtered_nodes = _filter_heads(graph, nodes)
    filtered_set = set(filtered_nodes)
    
    # Filter edges to only include edges between filtered nodes
    filtered_edges = [(u, v) for u, v in edges if u in filtered_set and v in filtered_set]
    
    elements = _elements_for_cytoscape(graph, filtered_nodes, filtered_edges)

    # Prepare data for embedding in HTML
    elements_json = json.dumps(elements)

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>DFG Visualization (up to decode_LMHead_B_T1, nheads/4, kvheads/4)</title>
  <script src=\"https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js\"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji'; }}
    #toolbar {{ display: flex; gap: 8px; padding: 10px; border-bottom: 1px solid #ddd; align-items: center; }}
    #container {{ display: flex; height: calc(100vh - 50px); }}
    #cy {{ flex: 1; height: 100%; background: #fafafa; min-width: 200px; }}
    #resizer {{ width: 4px; background: #ddd; cursor: col-resize; flex-shrink: 0; }}
    #resizer:hover {{ background: #bbb; }}
    #details {{ width: 360px; min-width: 200px; max-width: 80%; padding: 12px; overflow: auto; border-left: 1px solid #eee; }}
    #details pre {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 8px; border-radius: 6px; }}
    #attrs-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    #attrs-table th {{ background: #f6f8fa; padding: 8px 12px; text-align: left; border-bottom: 2px solid #ddd; font-weight: 600; color: #333; position: sticky; top: 0; }}
    #attrs-table td {{ padding: 8px 12px; border-bottom: 1px solid #eee; word-break: break-word; }}
    #attrs-table tr:hover {{ background: #f9f9f9; }}
    #attrs-table .field-name {{ font-weight: 500; color: #555; min-width: 140px; }}
    #attrs-table .field-value {{ color: #222; }}
    #attrs-table .field-empty {{ color: #999; font-style: italic; }}
    .tensor-info {{ margin: 8px 0; padding: 8px; background: #f0f7ff; border-left: 3px solid #4a90e2; border-radius: 4px; }}
    .tensor-field {{ margin: 4px 0; font-size: 12px; }}
    .tensor-field-name {{ font-weight: 600; color: #4a90e2; }}
    button {{ padding: 6px 10px; border: 1px solid #bbb; border-radius: 6px; background: #fff; cursor: pointer; }}
    button:hover {{ background: #f3f3f3; }}
  </style>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap\" rel=\"stylesheet\">
  <style> body {{ font-family: 'Inter', sans-serif; }} </style>
  <script>
    const elements = {elements_json};
  </script>
</head>
<body>
  <div id=\"toolbar\">
    <button id=\"fitBtn\">Fit to screen</button>
    <button id=\"zoomInBtn\">Zoom in</button>
    <button id=\"zoomOutBtn\">Zoom out</button>
    <span style=\"margin-left:8px;color:#666\">Nodes up to <b>decode_LMHead_B_T1</b> (nheads/4, kvheads/4)</span>
  </div>
  <div id=\"container\">
    <div id=\"cy\"></div>
    <div id=\"resizer\"></div>
    <div id=\"details\">
      <h3 style=\"margin-top:0\">Node details</h3>
      <div id=\"meta\" style=\"color:#666; margin-bottom: 8px;\">Click any node to view its attributes.</div>
      <table id=\"attrs-table\">
        <thead>
          <tr>
            <th style=\"width: 40%;\">Field</th>
            <th style=\"width: 60%;\">Value</th>
          </tr>
        </thead>
        <tbody id=\"attrs-tbody\">
          <tr>
            <td colspan=\"2\" style=\"text-align: center; color: #999; padding: 20px;\">Click a node to view its attributes</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    function makeStyles() {{
      return [
        {{ selector: 'node', style: {{
          'width': 18,
          'height': 12,
          'background-color': 'data(color)',
          'label': 'data(label)',
          'font-size': 12,
          'min-zoomed-font-size': 30, // labels appear only when zoomed in significantly
          'text-valign': 'center',
          'text-halign': 'center',
          'color': '#111',
          'text-wrap': 'wrap',
          'text-max-width': 120,
          'text-margin-y': 8, // Add vertical margin to labels
          'text-margin-x': 4, // Add horizontal margin to labels
          'border-width': 1,
          'border-color': '#888',
          'shape': 'round-rectangle',
          'text-background-color': '#fff',
          'text-background-opacity': 0.8,
          'text-background-padding': '3px',
          'text-background-shape': 'round-rectangle'
        }} }},
        {{ selector: 'edge', style: {{
          'width': 1.2,
          'curve-style': 'bezier',
          'line-color': '#bbb',
          'target-arrow-color': '#bbb',
          'target-arrow-shape': 'triangle'
        }} }},
        {{ selector: 'node:selected', style: {{
          'border-width': 3,
          'border-color': '#1f77b4'
        }} }}
      ];
    }}

    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements: elements,
      layout: {{ name: 'preset' }}
    }});

    cy.style(makeStyles());

    // Buttons
    document.getElementById('fitBtn').onclick = () => {{ cy.fit(); }};
    document.getElementById('zoomInBtn').onclick = () => {{ cy.zoom(cy.zoom() * 1.2); }};
    document.getElementById('zoomOutBtn').onclick = () => {{ cy.zoom(cy.zoom() / 1.2); }};

    // Initial fit after layout
    cy.ready(() => {{ cy.fit(); }});

    // Define all HarmoniTensor fields
    const HARMONI_TENSOR_FIELDS = [
      'name', 'tag', 'precision', 'shape', 'stride', 'addr_offset', 
      'chip_idx', 'numel', 'size', 'mapping', 'requests', 'locations', 
      'row_accesses', 'col_accesses', 'dram'
    ];

    // Format a value for display
    function formatValue(value) {{
      if (value === null || value === undefined) {{
        return '<span class="field-empty">(not set)</span>';
      }}
      if (typeof value === 'object') {{
        if (Array.isArray(value)) {{
          if (value.length === 0) {{
            return '<span class="field-empty">[]</span>';
          }}
          // Check if it's an array of objects (like tensors)
          if (typeof value[0] === 'object' && value[0] !== null) {{
            return value.map((item, idx) => {{
              if (item && typeof item === 'object' && 
                  ('name' in item || 'tag' in item) &&
                  ('shape' in item || 'precision' in item || 'size' in item || 'numel' in item) &&
                  !('frequency' in item || 'num_inbuf' in item || 'num_outbuf' in item || 'supported_ops' in item)) {{
                // Likely a HarmoniTensor
                return formatTensor(item, '[' + idx + ']');
              }}
              return '<div class="tensor-info">Item ' + idx + ': ' + formatValue(item) + '</div>';
            }}).join('');
          }}
          return '[' + value.join(', ') + ']';
        }}
        // Check if it's a HarmoniTensor-like object (must have multiple HarmoniTensor-specific fields)
        // Exclude objects that are clearly not HarmoniTensor (like LogicUnit which has name but not tensor fields)
        if (('name' in value || 'tag' in value) && 
            ('shape' in value || 'precision' in value || 'size' in value || 'numel' in value) &&
            !('frequency' in value || 'num_inbuf' in value || 'num_outbuf' in value || 'supported_ops' in value)) {{
          return formatTensor(value);
        }}
        // Regular object
        const keys = Object.keys(value);
        if (keys.length === 0) {{
          return '<span class="field-empty">{{}}</span>';
        }}
        return keys.map(k => `${{k}}: ${{formatValue(value[k])}}`).join('<br>');
      }}
      if (typeof value === 'string' && value.length > 100) {{
        return value.substring(0, 100) + '...';
      }}
      return String(value);
    }}

    // Format a HarmoniTensor object showing all fields
    function formatTensor(tensor, prefix = '') {{
      if (!tensor || typeof tensor !== 'object') {{
        return formatValue(tensor);
      }}
      let html = `<div class="tensor-info">${{prefix ? prefix + ' ' : ''}}HarmoniTensor`;
      if (tensor.name) {{
        html += `: ${{tensor.name}}`;
      }}
      html += '</div>';
      
      // Show all HarmoniTensor fields
      HARMONI_TENSOR_FIELDS.forEach(field => {{
        // Skip dram field, just show "..."
        if (field === 'dram') {{
          html += `<div class="tensor-field"><span class="tensor-field-name">${{field}}:</span> <span class="field-empty">...</span></div>`;
          return;
        }}
        const value = tensor[field];
        if (value !== undefined && value !== null) {{
          html += `<div class="tensor-field"><span class="tensor-field-name">${{field}}:</span> ${{formatValue(value)}}</div>`;
        }} else {{
          html += `<div class="tensor-field"><span class="tensor-field-name">${{field}}:</span> <span class="field-empty">(not set)</span></div>`;
        }}
      }});
      
      return html;
    }}

    // Format attributes in table format
    function formatAttributesTable(attrs) {{
      const tbody = document.getElementById('attrs-tbody');
      if (!tbody) {{
        console.error('attrs-tbody element not found');
        return;
      }}
      tbody.innerHTML = '';
      
      if (!attrs || typeof attrs !== 'object') {{
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="2" class="field-empty">No attributes available</td>';
        tbody.appendChild(row);
        return;
      }}
      
      // Define common node fields to show first in the specified order
      const commonFields = ['token', 'phase', 'layer', 'head', 'kernel', 'type', 'tag', 'ip', 'op'];
      const allFields = new Set([...commonFields, ...Object.keys(attrs)]);
      
      // Create ordered list: common fields first, then others
      const orderedFields = [];
      const seenFields = new Set();
      
      // Add common fields in order
      commonFields.forEach(field => {{
        if (field in attrs || allFields.has(field)) {{
          orderedFields.push(field);
          seenFields.add(field);
        }}
      }});
      
      // Add remaining fields
      Object.keys(attrs).forEach(field => {{
        if (!seenFields.has(field)) {{
          orderedFields.push(field);
        }}
      }});
      
      orderedFields.forEach(field => {{
        const row = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.className = 'field-name';
        nameCell.textContent = field;
        
        const valueCell = document.createElement('td');
        valueCell.className = 'field-value';
        
        if (field in attrs) {{
          const value = attrs[field];
          if (field === 'ip' || field === 'op') {{
            // Special handling for input/output tensors
            if (Array.isArray(value)) {{
              valueCell.innerHTML = value.map(t => formatTensor(t)).join('');
            }} else if (value) {{
              valueCell.innerHTML = formatTensor(value);
            }} else {{
              valueCell.innerHTML = '<span class="field-empty">(not set)</span>';
            }}
          }} else {{
            valueCell.innerHTML = formatValue(value);
          }}
        }} else {{
          valueCell.innerHTML = '<span class="field-empty">(not set)</span>';
        }}
        
        row.appendChild(nameCell);
        row.appendChild(valueCell);
        tbody.appendChild(row);
      }});
    }}

    // Node click handler: show attributes
    cy.on('tap', 'node', (evt) => {{
      const node = evt.target;
      const data = node.data();
      document.getElementById('meta').textContent = data.label;
      formatAttributesTable(data.attrs);
    }});

    // Resizer functionality for details pane
    const resizer = document.getElementById('resizer');
    const details = document.getElementById('details');
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    resizer.addEventListener('mousedown', (e) => {{
      isResizing = true;
      startX = e.clientX;
      startWidth = details.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    }});

    document.addEventListener('mousemove', (e) => {{
      if (!isResizing) return;
      const diff = startX - e.clientX; // Invert because dragging left increases width
      const newWidth = startWidth + diff;
      const minWidth = 200;
      const maxWidth = window.innerWidth * 0.8;
      const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      details.style.width = clampedWidth + 'px';
      cy.resize(); // Resize cytoscape to fit new container size
    }});

    document.addEventListener('mouseup', () => {{
      if (isResizing) {{
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }}
    }});
  </script>
</body>
</html>
""".replace("{elements_json}", elements_json)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(output_html_path)



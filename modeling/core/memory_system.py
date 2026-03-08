from modeling.core.logic import *
from graphviz import Digraph
import pydot 
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
import numpy as np
from utils.logging_util import logger
from config.logic_config import get_logic_unit_config
from args import parse_args
from modeling.core.dram_info import dram
from config.network_config import make_default_network_config
from misc.type import LinkLevel, CommType
from modeling.perf.network_latency_estimator import comm_type_based_latency

# This code defines a memory system hierarchy with channels, ranks, chips, bank groups, banks, and logic units.
class MemorySystem:
    def __init__(self, dram_config, logic_unit_config):
        """
        Initializes the memory system hierarchy.

        Args:
            dram_config 
            logic_unit_config (dict): Configuration for LogicUnits at each level.
        """

        self.num_channels = dram_config.num_channels
        self.num_ranks = dram_config.num_ranks_per_channel
        self.num_wt_ranks = dram_config.num_wt_ranks_per_channel
        self.num_kv_ranks = dram_config.num_kv_ranks_per_channel
        self.num_chips = dram_config.num_chips_per_rank
        self.num_bankgroups = dram_config.num_bankgroups_per_chip
        self.num_banks = dram_config.num_banks_per_bankgroup
        
        # Initialize hash table for logic unit lookup
        self.logic_unit_hash = {}
        self.logic_unit_partial_index = {}
        self._invalid_id_hierarchy_reported = set()
        
        self.all_nodes = {}

        if "rank" in logic_unit_config:
            num_ranks = {
                "": self.num_ranks
            }
        else:
            num_ranks = {
                "wt_": self.num_wt_ranks,
                "kv_": self.num_kv_ranks
            }
        
        # Create root node and its logic unit
        self.root = Root("root_0")
        self.all_nodes["root_0"] = self.root  # Dictionary to store all nodes for easy access

        # Add root logic unit if configured
        if "root" in logic_unit_config:
            self.add_logic_units(self.root, logic_unit_config["root"]["num_lus"],
            {"root": 0}, logic_unit_config["root"], parents=[]) 
        
        self.channels = []
        self.topo = make_default_network_config(
            num_channels=self.num_channels,
            num_ranks=self.num_wt_ranks,  # assumes wt/kv symmetry for network fanout
            num_chips=self.num_chips,
            chip_interface=dram_config.chip_interface,
        )
        self.network_cfg_dict = self.topo.to_dict()

        for c in range(self.num_channels):
            channel = Channel(f"channel_{c}")
            self.all_nodes[f"channel_{c}"] = channel
            self.root.add_child(channel)  
            if "channel" in logic_unit_config:
                self.add_logic_units(channel, logic_unit_config["channel"]["num_lus"], {"channel": c}, logic_unit_config["channel"], parents=["root_0"])

            for type in num_ranks.keys():
                for r in range(num_ranks[type]):
                    rank = Rank(f"channel_{c}_{type}rank_{r}")
                    channel.add_child(rank)
                    self.all_nodes[f"channel_{c}_{type}rank_{r}"] = rank
                    if f"{type}rank" in logic_unit_config:
                        self.add_logic_units(rank, logic_unit_config[f"{type}rank"]["num_lus"], {"channel": c, f"{type}rank": r}, logic_unit_config[f"{type}rank"], parents=[f"channel_{c}"])

                    for ch in range(self.num_chips):
                        chip = Chip(f"channel_{c}_{type}rank_{r}_chip_{ch}")
                        rank.add_child(chip)
                        self.all_nodes[f"channel_{c}_{type}rank_{r}_chip_{ch}"] = chip
                        if "chip" in logic_unit_config:
                            self.add_logic_units(chip, logic_unit_config["chip"]["num_lus"], {"channel": c, f"{type}rank": r, "chip": ch}, logic_unit_config["chip"], parents=[f"channel_{c}_{type}rank_{r}"])

                        for bg in range(self.num_bankgroups):
                            bankgroup = Bankgroup(f"channel_{c}_{type}rank_{r}_chip_{ch}_bg_{bg}")
                            chip.add_child(bankgroup)
                            self.all_nodes[f"channel_{c}_{type}rank_{r}_chip_{ch}_bg_{bg}"] = bankgroup
                            if "bankgroup" in logic_unit_config:
                                self.add_logic_units(bankgroup, logic_unit_config["bankgroup"]["num_lus"], {"channel": c, f"{type}rank": r, "chip": ch, "bankgroup": bg}, logic_unit_config["bankgroup"], parents=[f"channel_{c}_{type}rank_{r}_chip_{ch}"])

                            for b in range(self.num_banks):
                                bank = Bank(f"channel_{c}_{type}rank_{r}_chip_{ch}_bg_{bg}_b_{b}")
                                bankgroup.add_child(bank)
                                self.all_nodes[f"channel_{c}_{type}rank_{r}_chip_{ch}_bg_{bg}_b_{b}"] = bank
                                if "bank" in logic_unit_config:
                                    self.add_logic_units(bank, logic_unit_config["bank"]["num_lus"], {"channel": c, f"{type}rank": r, "chip": ch, "bankgroup": bg, "bank": b}, logic_unit_config["bank"], parents=[f"channel_{c}_{type}rank_{r}_chip_{ch}_bg_{bg}"])

            self.channels.append(channel)

        #Connections
        if "channel" in logic_unit_config:
            for i in range(self.num_channels):
                current_channel = f"channel_{i}"
                self.connect_nodes(
                    "root_0",
                    current_channel,
                    weight=self.topo.BW[LinkLevel.ROOT_CH],
                    link_latency=self.topo.l[LinkLevel.ROOT_CH],
                    overhead_latency=self.topo.o[LinkLevel.ROOT_CH],
                )
                if self.topo.intranode_ring:
                    next_channel = f"channel_{(i + 1) % self.num_channels}"  # Wrap around to form a ring
                    if current_channel != next_channel:
                        self.connect_nodes(
                            current_channel,
                            next_channel,
                            weight=self.topo.BW[LinkLevel.CH_RANK],
                            link_latency=self.topo.l[LinkLevel.CH_RANK],
                            overhead_latency=self.topo.o[LinkLevel.CH_RANK],
                        )  # PCIe
                    
                for type in num_ranks.keys():
                    if f"{type}rank" in logic_unit_config:
                        for k in range(num_ranks[type]):
                            current_rank = f"{current_channel}_{type}rank_{k}"
                            self.connect_nodes(
                                current_channel,
                                current_rank,
                                weight=self.topo.BW[LinkLevel.CH_RANK],
                                link_latency=self.topo.l[LinkLevel.CH_RANK],
                                overhead_latency=self.topo.o[LinkLevel.CH_RANK],
                            )  # Connect rank to its parent channel

                            if self.topo.intranode_ring:
                                # Create a ring-like structure between all ranks' logic nodes within the channel
                                next_rank = f"{current_channel}_{type}rank_{(k + 1) % num_ranks[type]}"
                                #print("Before connect-nodes in rank ", "channel", i, "rank ", k)
                                if current_rank != next_rank:
                                    self.connect_nodes(
                                        current_rank,
                                        next_rank,
                                        weight=self.topo.BW[LinkLevel.CH_RANK],
                                        link_latency=self.topo.l[LinkLevel.CH_RANK],
                                        overhead_latency=self.topo.o[LinkLevel.CH_RANK],
                                    )
 
  
                            # Connect all chips within the current rank
                            if "chip" in logic_unit_config:
                                for l in range(self.num_chips):
                                    current_chip = f"{current_rank}_chip_{l}"
                                    self.connect_nodes(
                                        current_rank,
                                        current_chip,
                                        weight=self.topo.BW[LinkLevel.RANK_CHIP],
                                        link_latency=self.topo.l[LinkLevel.RANK_CHIP],
                                        overhead_latency=self.topo.o[LinkLevel.RANK_CHIP],
                                    )  # Connect chip to its parent rank

                                    if self.topo.intranode_ring:
                                        # Create a ring-like structure between all chips' logic nodes within the rank
                                        next_chip = f"{current_rank}_chip_{(l + 1) % self.num_chips}"
                                        #print("Before connect-nodes in chip ", "rank ", k, "chip ", l)
                                        if current_chip != next_chip:
                                            self.connect_nodes(
                                                current_chip,
                                                next_chip,
                                                weight=self.topo.BW[LinkLevel.RANK_CHIP],
                                                link_latency=self.topo.l[LinkLevel.RANK_CHIP],
                                                overhead_latency=self.topo.o[LinkLevel.RANK_CHIP],
                                            )

                                    # Connect all bank groups within the current chip
                                    if "bankgroup" in logic_unit_config:
                                        for bg in range(self.num_bankgroups):
                                            current_bankgroup = f"{current_chip}_bg_{bg}"
                                            self.connect_nodes(current_chip, current_bankgroup, weight=1, link_latency=10, overhead_latency=25)  # Connect bank group to its parent chip

                                            if self.topo.intranode_ring:
                                                # Create a ring-like structure between all bank groups' logic nodes within the chip
                                                next_bankgroup = f"{current_chip}_bg_{(bg + 1) % self.num_bankgroups}"
                                                if current_bankgroup != next_bankgroup:
                                                    self.connect_nodes(current_bankgroup, next_bankgroup, weight=1, link_latency=10, overhead_latency=30)

                                            # Connect all banks within the current bank group
                                            if "bank" in logic_unit_config:
                                                for b in range(self.num_banks):
                                                    current_bank = f"{current_bankgroup}_b_{b}"
                                                    self.connect_nodes(current_bankgroup, current_bank, weight=1, link_latency=5, overhead_latency=30)  # Connect bank to its parent bank group

                                                    if self.topo.intranode_ring:
                                                        # Create a ring-like structure between all banks' logic nodes within the bank group
                                                        next_bank = f"{current_bankgroup}_b_{(b + 1) % self.num_banks}"
                                                        if current_bank != next_bank:
                                                            self.connect_nodes(current_bank, next_bank, weight=1, link_latency=5, overhead_latency=35)

        self.logic_unit_hash = {}  # For exact ID matches
        self._initialize_hash_table()

        self.build_routing_table()

    def _initialize_hash_table(self):
        """Initialize hash table for quick logic unit lookup"""
        for node_id, node in self.all_nodes.items():
            if isinstance(node, LogicUnit):
                # Create hash key for exact match
                exact_key = self._create_hash_key(node.id)
                self.logic_unit_hash[exact_key] = node

                # Create a key for the partial index based on location,
                # which is used for faster lookups in task_mapping.
                partial_key_components = {}
                if 'channel' in node.id:
                    partial_key_components['channel'] = node.id['channel']
                if 'chip' in node.id:
                    partial_key_components['chip'] = node.id['chip']
                
                if 'wt_rank' in node.id:
                    partial_key_components['wt_rank'] = node.id['wt_rank']
                elif 'kv_rank' in node.id:
                    partial_key_components['kv_rank'] = node.id['kv_rank']
                
                if partial_key_components:
                    # The key format must match what find_matching_logic_units creates
                    partial_key = tuple(sorted(partial_key_components.items()))
                    if partial_key not in self.logic_unit_partial_index:
                        self.logic_unit_partial_index[partial_key] = []
                    self.logic_unit_partial_index[partial_key].append(node)

    def _create_hash_key(self, id_dict):
        """Create a hash key from an ID dictionary"""
        return tuple(sorted(id_dict.items()))

    def add_logic_units(self, node, num_lus, id_template, config, parents=[]):
        """
        Adds LogicUnits to the given node with the specified configuration.
        Also updates hash table.

        Args:
            node (BaseLogicNode): The node to which LogicUnits will be added.
            num_lus (int): Number of LogicUnits to add.
            id_template (dict): Template for the LogicUnit ID (e.g., {"channel": 0, "rank": 1}).
            config (dict): Configuration for the LogicUnits (e.g., supported_ops).
        """
        assert num_lus == 1
        for lu in range(num_lus):
            logic_unit = LogicUnit(
                #id={**id_template, "logic_unit": lu},
                id={**id_template},
                num_inbuf=1,
                num_outbuf=1,
                inbuf_size=1024,  # Example size
                outbuf_size=1024,  # Example size
                supported_ops=config.get("supported_ops", []),
                parents=[self.all_nodes[parent_id] for parent_id in parents if parent_id in self.all_nodes]
            )
            node.add_child(logic_unit)
            self.all_nodes[logic_unit.name] = logic_unit
            
            # Update hash table
            key = self._create_hash_key(logic_unit.id)
            self.logic_unit_hash[key] = logic_unit

    def get_logic_unit_by_id(self, id):
        """
        Retrieves a list of LogicUnit(s) that matches the ID (even partially)
        Args:
            id (dict): The ID of the LogicUnit to retrieve.
        """
        # Try exact match first
        exact_key = self._create_hash_key(id)
        if exact_key in self.logic_unit_hash:
            return [self.logic_unit_hash[exact_key]]

        matching_units = []
        for node_id, node in self.all_nodes.items():
            if isinstance(node, LogicUnit):
                if all(key in node.id and node.id[key] == value for key, value in id.items()):
                    matching_units.append(node)
        return matching_units

    def get_all_logic_units(self):
        """Returns a flat list of all LogicUnits in the hierarchy."""
        logic_units = []

        def traverse(node):
            if isinstance(node, LogicUnit):
                logic_units.append(node)
                return
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return logic_units

    def connect_nodes(self, node1_id, node2_id, weight=1.0, link_latency=0.0, overhead_latency=0.0):
        """
        Create a connection between two nodes with weight and latency information.
        
        Args:
            node1_id (str): ID of the first node
            node2_id (str): ID of the second node
            weight (float): Weight of the connection
            link_latency (float): Latency of the physical link in nanoseconds
            overhead_latency (float): Router overhead latency at the source and destination nodes in nanoseconds
        """

        if node1_id in self.all_nodes and node2_id in self.all_nodes:            
            node1 = self.all_nodes[node1_id]
            node2 = self.all_nodes[node2_id]
            # Only connect LogicUnit nodes
            if isinstance(node1, LogicUnit) and isinstance(node2, LogicUnit):
                node1.add_connection(node2, weight, link_latency, overhead_latency)
                node2.add_connection(node1, weight, link_latency, overhead_latency)  # Make it bidirectional

                return True

        return False

    def get_node_connections(self, node_id):
        """
        Get all connections for a specific node.
        
        Args:
            node_id (str): ID of the node
            
        Returns:
            list: List of connected nodes
        """
        if node_id in self.all_nodes:
            return self.all_nodes[node_id].get_connections()
        return []  

    def to_networkx(self):
        """Convert the memory system to a NetworkX graph"""
        G = nx.DiGraph()
        
        #To add the root node
        self.root.to_networkx(G)
        
        return G

    def visualize_networkx_pydot(self, filename="outputs/networkx_logicnet.png"):
        """Visualize the memory system using the NetworkX graph and pydot/Graphviz."""

        G = self.to_networkx()

        pydot_graph = to_pydot(G)
        pydot_graph.set_rankdir("LR")  # Left-to-right, or use "TB" for top-to-bottom
        pydot_graph.set_splines("true")
        pydot_graph.set_overlap("false")
        pydot_graph.set_prog("dot")  # or "neato", "fdp", "sfdp"
        pydot_graph.set_size('"10,10!"')
        pydot_graph.set_dpi(350)
        try:
            pydot_graph.write_png(filename)
            logger.info(f"NetworkX-based graph image saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to render NetworkX-based graph: {e}")

    def __repr__(self):
        return f"MemorySystem(channels={len(self.channels)})"

    def build_topology(self):
        """Build topology file for BookSim simulator"""
        G = self.to_networkx()
        
        with open("outputs/topo.txt", "w") as f: #CLEANUP: add path fpr topo.txt
            for node in G.nodes():
                # Get neighbors of the current node
                neighbors = list(G.neighbors(node))
                
                # Format: router <node> node <node> router <neighbor1> router <neighbor2> ...
                line = f"router {node} node {node}"
                for neighbor in neighbors:
                    line += f" router {neighbor}"
                
                f.write(line + "\n")
        
        logger.info(f"Topology file written to topo.txt with {G.number_of_nodes()} nodes")
        
    def build_routing_table(self):
        G = self.to_networkx()
        #logger.debug("Nodes in NetworkX graph:", G.nodes())
        self.routing_table = {}
        logic_units = self.get_all_logic_units()
        
        if len(logic_units) == 0:
            logger.warning("No logic units")
        for src in logic_units:
            for dst in logic_units:
                if src.id == dst.id:
                    continue
                try:
                    path = nx.shortest_path(G, src.name, dst.name)
                    weights = [G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]
                    link_latencies = [G[path[i]][path[i+1]]['link_latency'] for i in range(len(path)-1)]
                    overhead_latencies = [G[path[i]][path[i+1]]['overhead_latency'] for i in range(len(path)-1)]
                    
                    hops = len(path) - 1
                    assert hops >= 0, f"Number of hops is negative for path {path} from {src.name} to {dst.name}"
                    bandwidth = min(weights)
                    total_link_latency = sum(link_latencies)
                    total_overhead_latency = sum(overhead_latencies)
                    
                    self.routing_table[(self._id_tuple(src.id), self._id_tuple(dst.id))] = {
                        "hops": hops,
                        "bandwidth": bandwidth,
                        "path": path,
                        "hop_bandwidths": weights,
                        "total_link_latency": total_link_latency,
                        "total_overhead_latency": total_overhead_latency,
                        "hop_link_latencies": link_latencies,
                        "hop_overhead_latencies": overhead_latencies
                    }
                    
                except nx.NetworkXNoPath:
                    continue

    def _id_tuple(self, id_dict):
        return tuple(sorted(id_dict.items()))

    def _log_invalid_id_hierarchy(self, node_id):
        """Log unsupported lower-level IDs that miss required higher-level IDs."""
        keys = set(node_id.keys())
        rank_keys = {"rank", "wt_rank", "kv_rank"}
        has_rank = any(k in keys for k in rank_keys)
        violations = []

        if has_rank and "channel" not in keys:
            violations.append("rank/wt_rank/kv_rank is present without channel")
        if "chip" in keys and not has_rank:
            violations.append("chip is present without rank/wt_rank/kv_rank")
        if "chip" in keys and "channel" not in keys:
            violations.append("chip is present without channel")

        if not violations:
            return

        report_key = (self._id_tuple(node_id), tuple(sorted(violations)))
        if report_key in self._invalid_id_hierarchy_reported:
            return

        logger.error(
            "Invalid communication ID hierarchy: %s. node_id=%s",
            "; ".join(violations),
            node_id,
        )
        self._invalid_id_hierarchy_reported.add(report_key)

    def _get_node_level(self, node_id):
        """Map a logic-unit ID dictionary to strict network hierarchy level."""
        keys = set(node_id.keys())
        rank_keys = {"rank", "wt_rank", "kv_rank"}
        present_rank_keys = [k for k in rank_keys if k in keys]

        if keys == {"root"}:
            return 0

        if keys in ({"channel"}, {"root", "channel"}):
            return 1

        if len(present_rank_keys) == 1:
            rank_key = present_rank_keys[0]
            if keys in ({"channel", rank_key}, {"root", "channel", rank_key}):
                return 2
            if keys in (
                {"channel", rank_key, "chip"},
                {"root", "channel", rank_key, "chip"},
            ):
                return 3

        raise ValueError(
            "Communication cost model is not available for this topology level. "
            f"node id: {node_id}. Supported forms: "
            "{root}, {channel}|{root,channel}, "
            "{channel,rank|wt_rank|kv_rank}|{root,channel,rank|wt_rank|kv_rank}, "
            "{channel,rank|wt_rank|kv_rank,chip}|"
            "{root,channel,rank|wt_rank|kv_rank,chip}."
        )

    def get_comm_info(self, src_id, dst_id, data_size, optype=None):
        self._log_invalid_id_hierarchy(src_id)
        self._log_invalid_id_hierarchy(dst_id)

        # Path 1: legacy routing-table latency for None/non-enum optype.
        if optype is None or not isinstance(optype, CommType):
            key = (self._id_tuple(src_id), self._id_tuple(dst_id))
            info = self.routing_table.get(key)
            if info is None:
                return None

            serialization_time = (
                data_size / info["bandwidth"] if info["bandwidth"] > 0 else float("inf")
            )
            static_latency = info["total_link_latency"] + info["total_overhead_latency"]
            queue_size = self.network_cfg_dict["Q"][-1]
            buffering_time = (
                max(0.0, (data_size - queue_size) / info["bandwidth"])
                if info["bandwidth"] > 0 else float("inf")
            )
            comm_time = serialization_time + static_latency + buffering_time
        else:
            # Path 2: operation-aware analytical latency.
            s = self._get_node_level(src_id)
            d = self._get_node_level(dst_id)
            comm_time = comm_type_based_latency(s, d, data_size, self.network_cfg_dict, optype)

        # Convert ns to microseconds
        comm_time = comm_time / 1000

        # in pJ
        comm_energy = data_size * 8 * 4.4 #PCIe energy in CENT 4.4 pJ/bit

        return {
            "comm_time": comm_time,
            "comm_energy": comm_energy
        }

    def print_routing_table(self): #CLEANUP write to a file
        print("Routing Table:")
        print(f"{'SRC_ID':<40} {'DST_ID':<40} {'HOPS':<5} {'BANDWIDTH':<10} {'LINK_LAT':<10} {'OVERHEAD_LAT':<12} PATH")
        for (src_id, dst_id), info in self.routing_table.items():
            src_str = ','.join(f"{k}:{v}" for k, v in src_id)
            dst_str = ','.join(f"{k}:{v}" for k, v in dst_id)
            hops = info['hops']
            bandwidth = info['bandwidth']
            link_latency = info['total_link_latency'] # ns
            overhead_latency = info['total_overhead_latency'] # ns
 
            path = '->'.join(info['path'])
            print(f"{src_str:<40} {dst_str:<40} {hops:<5} {bandwidth:<10} {link_latency:<10.2f} {overhead_latency:<12.2f} {path}")


#Get logic unit configuration
logic_unit_config = get_logic_unit_config()
memsys = MemorySystem(dram, logic_unit_config)

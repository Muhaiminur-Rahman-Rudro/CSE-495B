"""
Tree-of-Graph (ToG) Prompting Strategy

An advanced prompting technique that extends Tree-of-Thought by representing
reasoning as a directed graph rather than a tree, allowing for:
- Revisiting and refining previous reasoning nodes
- Merging multiple reasoning paths
- Detecting and resolving contradictions
- More flexible exploration of the reasoning space

Reference: Inspired by graph-based reasoning approaches and knowledge graph integration.
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

from .base import BasePromptStrategy, PromptResult


@dataclass
class GraphNode:
    """Represents a node in the reasoning graph."""
    id: str
    thought: str
    node_type: str  # "hypothesis", "evidence", "conclusion", "subgoal"
    evaluation_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class GraphEdge:
    """Represents an edge (relationship) between reasoning nodes."""
    source_id: str
    target_id: str
    relation: str  # "supports", "contradicts", "refines", "leads_to", "merges"
    weight: float = 1.0


class ReasoningGraph:
    """
    A directed graph structure for representing reasoning paths.
    
    Unlike a tree, this allows:
    - Multiple parents (merging paths)
    - Cycles (iterative refinement)
    - Edge types (supporting/contradicting evidence)
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self._node_counter = 0
    
    def add_node(
        self,
        thought: str,
        node_type: str = "hypothesis",
        score: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a node to the graph and return its ID."""
        node_id = f"n{self._node_counter}"
        self._node_counter += 1
        
        self.nodes[node_id] = GraphNode(
            id=node_id,
            thought=thought,
            node_type=node_type,
            evaluation_score=score,
            metadata=metadata or {}
        )
        return node_id
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "leads_to",
        weight: float = 1.0
    ) -> None:
        """Add a directed edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist")
        
        edge = GraphEdge(source_id, target_id, relation, weight)
        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)
        self.reverse_adjacency[target_id].append(source_id)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[GraphNode]:
        """Get all child nodes."""
        return [self.nodes[nid] for nid in self.adjacency.get(node_id, [])]
    
    def get_parents(self, node_id: str) -> List[GraphNode]:
        """Get all parent nodes."""
        return [self.nodes[nid] for nid in self.reverse_adjacency.get(node_id, [])]
    
    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target_id == node_id]
    
    def find_contradictions(self) -> List[Tuple[str, str]]:
        """Find pairs of nodes connected by contradiction edges."""
        contradictions = []
        for edge in self.edges:
            if edge.relation == "contradicts":
                contradictions.append((edge.source_id, edge.target_id))
        return contradictions
    
    def get_best_path(self, start_id: str, end_type: str = "conclusion") -> List[str]:
        """Find the highest-scoring path from start to a conclusion node."""
        best_path = []
        best_score = float('-inf')
        
        def dfs(current_id: str, path: List[str], score: float, visited: Set[str]):
            nonlocal best_path, best_score
            
            node = self.nodes[current_id]
            new_score = score + node.evaluation_score
            new_path = path + [current_id]
            
            if node.node_type == end_type:
                if new_score > best_score:
                    best_score = new_score
                    best_path = new_path
                return
            
            for child_id in self.adjacency.get(current_id, []):
                if child_id not in visited:
                    dfs(child_id, new_path, new_score, visited | {child_id})
        
        dfs(start_id, [], 0, {start_id})
        return best_path
    
    def to_reasoning_trace(self) -> List[str]:
        """Convert graph to a linear reasoning trace."""
        # Topological sort with score-based ordering
        in_degree = defaultdict(int)
        for edge in self.edges:
            in_degree[edge.target_id] += 1
        
        # Start with nodes that have no incoming edges
        queue = [nid for nid in self.nodes if in_degree[nid] == 0]
        queue.sort(key=lambda x: -self.nodes[x].evaluation_score)
        
        trace = []
        visited = set()
        
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            
            node = self.nodes[node_id]
            trace.append(f"[{node.node_type.upper()}] {node.thought}")
            
            # Add children sorted by score
            children = [(cid, self.nodes[cid].evaluation_score) 
                       for cid in self.adjacency.get(node_id, [])]
            children.sort(key=lambda x: -x[1])
            
            for child_id, _ in children:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
            
            queue.sort(key=lambda x: -self.nodes[x].evaluation_score)
        
        return trace


class TreeOfGraphPrompting(BasePromptStrategy):
    """
    Tree-of-Graph prompting - represents reasoning as a graph structure
    that allows merging paths, revisiting nodes, and detecting contradictions.
    
    Key features:
    1. Graph-based reasoning structure (not just tree)
    2. Multiple reasoning path exploration
    3. Path merging when similar conclusions reached
    4. Contradiction detection and resolution
    5. Evidence aggregation from multiple sources
    """
    
    @property
    def name(self) -> str:
        return "tree_of_graph"
    
    def __init__(
        self,
        model,
        tokenizer,
        generation_config: Optional[Dict[str, Any]] = None,
        num_initial_thoughts: int = 3,
        max_depth: int = 4,
        exploration_width: int = 2,
        merge_threshold: float = 0.8,
        enable_contradiction_resolution: bool = True,
    ):
        super().__init__(model, tokenizer, generation_config)
        
        self.num_initial_thoughts = num_initial_thoughts
        self.max_depth = max_depth
        self.exploration_width = exploration_width
        self.merge_threshold = merge_threshold
        self.enable_contradiction_resolution = enable_contradiction_resolution
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format base prompt."""
        parts = []
        if context:
            parts.append(f"Context: {context}\n")
        parts.append(f"Problem: {question}")
        return "\n".join(parts)
    
    def _generate_initial_hypotheses(self, question: str) -> List[Tuple[str, str]]:
        """Generate initial reasoning hypotheses with different approaches."""
        prompt = f"""Given the problem, generate {self.num_initial_thoughts} distinct initial hypotheses or approaches to solve it.
Each hypothesis should represent a different reasoning strategy.

Problem: {question}

For each hypothesis, provide:
1. The approach/hypothesis
2. The type: "analytical", "intuitive", or "systematic"

Format each as:
HYPOTHESIS 1: [approach]
TYPE: [type]

HYPOTHESIS 2: [approach]
TYPE: [type]

Generate {self.num_initial_thoughts} hypotheses:"""

        output = self._generate_text(prompt, temperature=0.8, max_new_tokens=400)
        
        hypotheses = []
        current_hyp = None
        current_type = "hypothesis"
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('HYPOTHESIS'):
                if current_hyp:
                    hypotheses.append((current_hyp, current_type))
                match = re.search(r'HYPOTHESIS \d+:\s*(.+)', line)
                current_hyp = match.group(1) if match else line
            elif line.startswith('TYPE:'):
                current_type = line.replace('TYPE:', '').strip().lower()
        
        if current_hyp:
            hypotheses.append((current_hyp, current_type))
        
        return hypotheses[:self.num_initial_thoughts]
    
    def _expand_node(
        self,
        question: str,
        node: GraphNode,
        graph: ReasoningGraph,
    ) -> List[str]:
        """Expand a node by generating follow-up reasoning steps."""
        # Get context from parent nodes
        parents = graph.get_parents(node.id)
        parent_context = "\n".join([f"- {p.thought}" for p in parents]) if parents else "Starting point"
        
        prompt = f"""Continue the reasoning from the current thought.

Problem: {question}

Previous reasoning:
{parent_context}

Current thought: {node.thought}

Generate {self.exploration_width} possible next steps in the reasoning. Each should either:
- Provide supporting evidence
- Draw a sub-conclusion
- Identify a potential issue
- Propose a refinement

Format:
STEP 1: [next reasoning step]
RELATION: [supports/refines/leads_to]

STEP 2: [next reasoning step]
RELATION: [supports/refines/leads_to]

Next steps:"""

        output = self._generate_text(prompt, temperature=0.7, max_new_tokens=300)
        
        new_node_ids = []
        current_step = None
        current_relation = "leads_to"
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('STEP'):
                if current_step:
                    # Add node and edge
                    new_id = graph.add_node(current_step, node_type="evidence")
                    graph.add_edge(node.id, new_id, relation=current_relation)
                    new_node_ids.append(new_id)
                
                match = re.search(r'STEP \d+:\s*(.+)', line)
                current_step = match.group(1) if match else None
            elif line.startswith('RELATION:'):
                rel = line.replace('RELATION:', '').strip().lower()
                if rel in ['supports', 'refines', 'leads_to', 'contradicts']:
                    current_relation = rel
        
        if current_step:
            new_id = graph.add_node(current_step, node_type="evidence")
            graph.add_edge(node.id, new_id, relation=current_relation)
            new_node_ids.append(new_id)
        
        return new_node_ids
    
    def _evaluate_node(self, question: str, node: GraphNode, graph: ReasoningGraph) -> float:
        """Evaluate the quality/promise of a reasoning node."""
        # Get path context
        parents = graph.get_parents(node.id)
        path_context = " -> ".join([p.thought[:50] for p in parents[-3:]]) if parents else "Start"
        
        prompt = f"""Evaluate this reasoning step for solving the problem.

Problem: {question}

Reasoning path: {path_context}
Current step: {node.thought}

Rate from 1-10 based on:
- Logical validity
- Progress toward solution
- Clarity and specificity

Score (1-10):"""

        output = self._generate_text(prompt, max_new_tokens=20)
        
        match = re.search(r'(\d+(?:\.\d+)?)', output)
        if match:
            return min(10, max(1, float(match.group(1)))) / 10
        return 0.5
    
    def _check_for_merge(
        self,
        graph: ReasoningGraph,
        node1_id: str,
        node2_id: str
    ) -> bool:
        """Check if two nodes should be merged (similar conclusions)."""
        node1 = graph.get_node(node1_id)
        node2 = graph.get_node(node2_id)
        
        if not node1 or not node2:
            return False
        
        prompt = f"""Do these two reasoning steps reach essentially the same conclusion?

Step 1: {node1.thought}
Step 2: {node2.thought}

Answer (yes/no):"""

        output = self._generate_text(prompt, max_new_tokens=10)
        return 'yes' in output.lower()
    
    def _resolve_contradiction(
        self,
        question: str,
        node1: GraphNode,
        node2: GraphNode,
        graph: ReasoningGraph
    ) -> str:
        """Resolve a contradiction between two nodes."""
        prompt = f"""Two reasoning paths have reached contradictory conclusions. Analyze and resolve.

Problem: {question}

Conclusion 1: {node1.thought}
Conclusion 2: {node2.thought}

Analyze which is more likely correct and why, or synthesize a resolution:"""

        output = self._generate_text(prompt, max_new_tokens=200)
        
        # Add resolution node
        resolution_id = graph.add_node(
            output.strip(),
            node_type="resolution",
            metadata={"resolves": [node1.id, node2.id]}
        )
        graph.add_edge(node1.id, resolution_id, relation="resolved_by")
        graph.add_edge(node2.id, resolution_id, relation="resolved_by")
        
        return resolution_id
    
    def _generate_conclusion(
        self,
        question: str,
        graph: ReasoningGraph
    ) -> Tuple[str, str]:
        """Generate final conclusion from the reasoning graph."""
        # Gather high-scoring nodes
        top_nodes = sorted(
            graph.nodes.values(),
            key=lambda n: n.evaluation_score,
            reverse=True
        )[:5]
        
        evidence = "\n".join([
            f"- [{n.node_type}] {n.thought}" for n in top_nodes
        ])
        
        prompt = f"""Based on the reasoning graph, provide the final answer.

Problem: {question}

Key reasoning nodes:
{evidence}

Synthesize these into a final answer:
Final Answer:"""

        output = self._generate_text(prompt, max_new_tokens=150)
        
        # Add conclusion node
        conclusion_id = graph.add_node(
            output.strip(),
            node_type="conclusion"
        )
        
        # Link from top evidence nodes
        for node in top_nodes[:3]:
            graph.add_edge(node.id, conclusion_id, relation="supports")
        
        return conclusion_id, output.strip()
    
    def _build_reasoning_graph(
        self,
        question: str,
        context: Optional[str] = None
    ) -> ReasoningGraph:
        """Build the complete reasoning graph."""
        graph = ReasoningGraph()
        
        # Generate initial hypotheses
        hypotheses = self._generate_initial_hypotheses(question)
        
        # Add root nodes
        frontier = []
        for thought, thought_type in hypotheses:
            node_id = graph.add_node(thought, node_type="hypothesis")
            score = self._evaluate_node(question, graph.get_node(node_id), graph)
            graph.nodes[node_id].evaluation_score = score
            frontier.append((node_id, 0))  # (node_id, depth)
        
        # Expand graph iteratively
        explored = set()
        while frontier:
            # Sort by score and take best
            frontier.sort(key=lambda x: -graph.nodes[x[0]].evaluation_score)
            node_id, depth = frontier.pop(0)
            
            if node_id in explored or depth >= self.max_depth:
                continue
            explored.add(node_id)
            
            node = graph.get_node(node_id)
            
            # Expand node
            new_node_ids = self._expand_node(question, node, graph)
            
            # Evaluate and add to frontier
            for new_id in new_node_ids:
                new_node = graph.get_node(new_id)
                score = self._evaluate_node(question, new_node, graph)
                new_node.evaluation_score = score
                frontier.append((new_id, depth + 1))
            
            # Check for merging opportunities
            if len(new_node_ids) >= 2:
                for i, id1 in enumerate(new_node_ids):
                    for id2 in new_node_ids[i+1:]:
                        if self._check_for_merge(graph, id1, id2):
                            # Create merge node
                            merge_thought = f"Merged: {graph.nodes[id1].thought[:50]}..."
                            merge_id = graph.add_node(merge_thought, node_type="merge")
                            graph.add_edge(id1, merge_id, relation="merges")
                            graph.add_edge(id2, merge_id, relation="merges")
        
        # Handle contradictions
        if self.enable_contradiction_resolution:
            contradictions = graph.find_contradictions()
            for node1_id, node2_id in contradictions:
                self._resolve_contradiction(
                    question,
                    graph.get_node(node1_id),
                    graph.get_node(node2_id),
                    graph
                )
        
        return graph
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> PromptResult:
        """Generate answer using tree-of-graph reasoning."""
        
        # Build reasoning graph
        graph = self._build_reasoning_graph(question, context)
        
        # Generate conclusion
        conclusion_id, final_answer = self._generate_conclusion(question, graph)
        
        # Extract reasoning trace
        reasoning_trace = graph.to_reasoning_trace()
        
        # Build raw output summary
        raw_output = f"Reasoning Graph Summary:\n"
        raw_output += f"- Nodes: {len(graph.nodes)}\n"
        raw_output += f"- Edges: {len(graph.edges)}\n"
        raw_output += f"- Contradictions found: {len(graph.find_contradictions())}\n\n"
        raw_output += "Reasoning Trace:\n"
        raw_output += "\n".join(reasoning_trace)
        raw_output += f"\n\nFinal Answer: {final_answer}"
        
        return PromptResult(
            final_answer=self.extract_answer(final_answer),
            reasoning_trace=reasoning_trace,
            raw_output=raw_output,
            num_reasoning_steps=len(graph.nodes),
            metadata={
                "strategy": self.name,
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "num_hypotheses": self.num_initial_thoughts,
                "max_depth": self.max_depth,
                "contradictions_resolved": len(graph.find_contradictions()),
                "graph_structure": {
                    "nodes": [
                        {"id": n.id, "type": n.node_type, "score": n.evaluation_score}
                        for n in graph.nodes.values()
                    ],
                    "edges": [
                        {"from": e.source_id, "to": e.target_id, "relation": e.relation}
                        for e in graph.edges
                    ]
                }
            }
        )

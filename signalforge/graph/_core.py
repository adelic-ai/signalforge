"""
signalforge.graph._core

Core graph machinery: Op, Node, Artifact, GraphPipeline.

The computation graph is lazy — calling Op(...)(input_node) creates a Node
but does not execute. resolve() derives the SamplingPlan from operator
constraints. build() materializes artifacts in topological order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ._types import ArtifactType


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """Wraps a pipeline object with metadata about how it was produced."""
    type: ArtifactType
    value: Any
    producing_op: Optional["Op"] = None
    plan: Any = None  # SamplingPlan, set after resolve
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Artifact({self.type.value})"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """A node in the lazy computation graph."""

    _counter: int = 0

    def __init__(self, op: "Op", parents: Tuple["Node", ...] = ()) -> None:
        self.op = op
        self.parents = parents
        self._artifact: Optional[Artifact] = None
        Node._counter += 1
        self._name = f"{op.__class__.__name__}_{Node._counter}"

    @property
    def output_type(self) -> ArtifactType:
        return self.op.output_type

    @property
    def artifact(self) -> Artifact:
        if self._artifact is None:
            raise RuntimeError(f"Node {self._name!r} has not been built yet.")
        return self._artifact

    @property
    def is_built(self) -> bool:
        return self._artifact is not None

    def __repr__(self) -> str:
        built = "built" if self.is_built else "pending"
        return f"Node({self._name}, {self.output_type.value}, {built})"


# ---------------------------------------------------------------------------
# Op (abstract base)
# ---------------------------------------------------------------------------

class Op(ABC):
    """
    Abstract base for graph operators.

    Subclasses declare input_types and output_type, implement execute(),
    and optionally override contribute_constraints().
    """

    input_types: Tuple[ArtifactType, ...]
    output_type: ArtifactType

    def __init__(self, **params: Any) -> None:
        self.params = params

    def __call__(self, *inputs: Union[Node, List[Node]]) -> Node:
        """Create a new Node linked to the input nodes (lazy)."""
        # Support Stack([a, b, c]) syntax
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = tuple(inputs[0])

        # Validate input count
        if hasattr(self, 'variadic_inputs') and self.variadic_inputs:
            if len(inputs) < 1:
                raise ValueError(f"{self.__class__.__name__} requires at least 1 input.")
        else:
            expected = len(self.input_types)
            if len(inputs) != expected:
                raise ValueError(
                    f"{self.__class__.__name__} expects {expected} input(s), got {len(inputs)}."
                )

        # Validate input types
        if not (hasattr(self, 'variadic_inputs') and self.variadic_inputs):
            for i, (node, expected_type) in enumerate(zip(inputs, self.input_types)):
                if node.output_type != expected_type:
                    raise TypeError(
                        f"{self.__class__.__name__} input {i}: expected "
                        f"{expected_type.value}, got {node.output_type.value}."
                    )

        return Node(self, parents=tuple(inputs))

    @abstractmethod
    def execute(self, *inputs: Artifact, plan: Any) -> Artifact:
        """Materialize this op given resolved input artifacts."""
        ...

    def contribute_constraints(self) -> Dict[str, Any]:
        """Return constraints this op places on the SamplingPlan."""
        return {}


# ---------------------------------------------------------------------------
# GraphPipeline
# ---------------------------------------------------------------------------

class GraphPipeline:
    """
    Container for a computation graph.

    Usage:
        x = Input(...)
        b = Bin(...)(x)
        m = Measure(...)(b)
        pipeline = GraphPipeline(x, m)
        pipeline.resolve()
        result = pipeline.build(records)
    """

    def __init__(
        self,
        inputs: Union[Node, List[Node]],
        outputs: Union[Node, List[Node]],
    ) -> None:
        self.inputs = (inputs,) if isinstance(inputs, Node) else tuple(inputs)
        self.outputs = (outputs,) if isinstance(outputs, Node) else tuple(outputs)
        self._nodes = self._topo_sort()
        self._plan: Any = None
        self._resolved = False
        self._built = False

    def _topo_sort(self) -> List[Node]:
        """Return nodes in execution order (parents before children)."""
        visited: set = set()
        order: list = []

        def _visit(node: Node) -> None:
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            for parent in node.parents:
                _visit(parent)
            order.append(node)

        for out in self.outputs:
            _visit(out)
        return order

    @property
    def plan(self) -> Any:
        if self._plan is None:
            raise RuntimeError("Pipeline has not been resolved yet.")
        return self._plan

    def resolve(self, records: Any = None, **overrides: Any) -> "GraphPipeline":
        """
        Walk the graph, collect constraints, derive SamplingPlan.

        If records are provided and no grain is specified, grain is
        estimated from the data using inter-event statistics.

        Parameters
        ----------
        records : list[CanonicalRecord], optional
            Data for grain estimation. If not given, uses defaults.
        **overrides
            Explicit SamplingPlan parameters that override derived values.
            E.g. resolve(horizon=360, grain=1).

        Returns self for chaining.
        """
        from ._resolve import collect_constraints, derive_plan
        constraints = collect_constraints(self._nodes)
        constraints.update(overrides)
        self._plan = derive_plan(constraints, records=records)
        self._resolved = True
        return self

    def build(self, records: Any = None) -> Union[Artifact, Tuple[Artifact, ...]]:
        """
        Materialize the graph in topological order.

        Parameters
        ----------
        records : list[CanonicalRecord], optional
            Data to feed into Input nodes.

        Returns
        -------
        Artifact or tuple of Artifacts
        """
        if not self._resolved:
            self.resolve(records=records)

        for node in self._nodes:
            if node.is_built:
                continue

            if not node.parents:
                # Input node — inject records
                node._artifact = node.op.execute(records=records, plan=self._plan)
            else:
                parent_artifacts = tuple(p.artifact for p in node.parents)
                node._artifact = node.op.execute(*parent_artifacts, plan=self._plan)

        self._built = True

        if len(self.outputs) == 1:
            return self.outputs[0].artifact
        return tuple(o.artifact for o in self.outputs)

    def summary(self) -> str:
        """Print the graph structure."""
        lines = [f"GraphPipeline ({len(self._nodes)} nodes)"]
        lines.append(f"  resolved: {self._resolved}")
        if self._resolved:
            lines.append(f"  plan: horizon={self._plan.horizon}, cbin={self._plan.cbin}")
        lines.append("")
        for i, node in enumerate(self._nodes):
            parents = ", ".join(p._name for p in node.parents) or "(input)"
            params = ", ".join(f"{k}={v!r}" for k, v in node.op.params.items())
            lines.append(f"  [{i}] {node._name}: {node.output_type.value}")
            lines.append(f"      op: {node.op.__class__.__name__}({params})")
            lines.append(f"      from: {parents}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "resolved" if self._resolved else "unresolved"
        return f"GraphPipeline({len(self._nodes)} nodes, {state})"

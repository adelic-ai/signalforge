"""
signalforge.graph

Keras-style functional computation graph for composing pipeline stages.

    from signalforge.graph import Input, Bin, Measure, Engineer, Assemble, Pipeline

    x = Input()
    b = Bin()(x)
    m = Measure(profile="continuous")(b)
    e = Engineer()(m)
    a = Assemble()(e)

    pipeline = Pipeline(x, a)
    pipeline.resolve(horizon=360, grain=1)
    result = pipeline.build(records)
"""

from ._core import Artifact, GraphPipeline, Node, Op
from ._ops import AssembleOp, BinOp, EngineerOp, InputOp, MeasureOp
from ._multi_ops import BaselineOp, ResidualOp, StackOp
from ._types import ArtifactType, parse_duration


# ---------------------------------------------------------------------------
# User-facing factories
# ---------------------------------------------------------------------------

def Input(**kwargs) -> Node:
    """Create an Input node. Returns a Node directly (no parents)."""
    op = InputOp(**kwargs)
    return Node(op)


def Bin(**kwargs) -> BinOp:
    """Create a Bin operator. Call with (input_node) to wire into graph."""
    return BinOp(**kwargs)


def Measure(**kwargs) -> MeasureOp:
    """Create a Measure operator. Call with (binned_node) to wire into graph."""
    return MeasureOp(**kwargs)


def Engineer(**kwargs) -> EngineerOp:
    """Create an Engineer operator. Call with (surfaces_node) to wire into graph."""
    return EngineerOp(**kwargs)


def Assemble(**kwargs) -> AssembleOp:
    """Create an Assemble operator. Call with (tensors_node) to wire into graph."""
    return AssembleOp(**kwargs)


def Baseline(**kwargs) -> BaselineOp:
    """Create a Baseline operator. Call with (surfaces_node) to wire into graph."""
    return BaselineOp(**kwargs)


def Residual(**kwargs) -> ResidualOp:
    """Create a Residual operator. Call with (measured_node, baseline_node)."""
    return ResidualOp(**kwargs)


def Stack(**kwargs) -> StackOp:
    """Create a Stack operator. Call with ([node1, node2, ...])."""
    return StackOp(**kwargs)


Pipeline = GraphPipeline

__all__ = [
    "Input",
    "Bin",
    "Measure",
    "Engineer",
    "Assemble",
    "Baseline",
    "Residual",
    "Stack",
    "Pipeline",
    "Artifact",
    "ArtifactType",
    "Node",
    "Op",
    "parse_duration",
]

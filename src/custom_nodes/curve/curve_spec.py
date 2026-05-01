from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class CurveSpec:
    """Serialisable description of a 1-D curve.

    Control points are (x, y) pairs in normalised [0, 1] space.
    The engine clips outputs to [range_min, range_max] after interpolation.
    """

    points: list[tuple[float, float]]
    domain_min: float = 0.0
    domain_max: float = 1.0
    range_min: float = 0.0
    range_max: float = 1.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "points": self.points,
                "domain_min": self.domain_min,
                "domain_max": self.domain_max,
                "range_min": self.range_min,
                "range_max": self.range_max,
            }
        )

    @classmethod
    def from_json(cls, data: str | dict) -> CurveSpec:
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, list):
            # Bare [[x,y],...] format from the JS widget
            return cls(points=[tuple(p) for p in data])
        return cls(
            points=[tuple(p) for p in data["points"]],
            domain_min=data.get("domain_min", 0.0),
            domain_max=data.get("domain_max", 1.0),
            range_min=data.get("range_min", 0.0),
            range_max=data.get("range_max", 1.0),
        )

    # Factory methods

    @classmethod
    def identity(cls, n_points: int = 5) -> CurveSpec:
        """Linear identity curve where output == input."""
        xs = [i / (n_points - 1) for i in range(n_points)]
        return cls(points=[(x, x) for x in xs])
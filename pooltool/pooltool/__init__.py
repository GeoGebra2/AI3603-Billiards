"""The top-level API for the pooltool library

**Important and highly used objects are placed in this top-level API**. For example,
``System`` can be imported directly from the top module:

    >>> import pooltool as pt
    >>> system = pt.System.example()

Alternatively, it can be imported directly from its lower-level API location:

    >>> from pooltool.system import System
    >>> system = System.example()

If the object you're looking for isn't in this top-level API, **search for it in
the subpackages/submodules** listed below. Relatedly, if you believe that an objects deserves to
graduate to the top-level API, **your input is valuable** and such changes can be
considered.
"""

# This is a placeholder that is replaced during package building (`poetry build`)
__version__ = "0.0.0"

from . import ai
from .ai import aim as aim
from .ai import pot as pot
try:
    from .ani import image as image
except Exception:
    image = None
from . import constants
from . import events
from . import evolution
from . import game
try:
    from . import interact
except Exception:
    interact = None
from . import layouts
from . import objects
from . import physics
from . import ptmath
from . import ruleset
from . import serialize
from . import system
from .events import EventType
from .evolution import continuize, interpolate_ball_states, simulate
from .game.datatypes import GameType
try:
    from .interact import Game, show
except Exception:
    Game = None
    show = None
from .layouts import generate_layout, get_rack
from .objects import (
    Ball,
    BallParams,
    Cue,
    Table,
    TableType,
)
from .ruleset import Player, get_ruleset
from .system import MultiSystem, System

__all__ = [
    # subpackages
    "events",
    "evolution",
    "game",
    "objects",
    "physics",
    "ptmath",
    "ruleset",
    "system",
    "utils",
    # submodules
    "constants",
    "interact",
    "layouts",
    # non-documented
    "serialize",
    "image",
    "ai",
    "pot",
    "aim",
    # objects
    "EventType",
    "GameType",
    "Game",
    "Ball",
    "BallParams",
    "Cue",
    "Table",
    "TableType",
    "Player",
    "MultiSystem",
    "System",
    # functions
    "continuize",
    "interpolate_ball_states",
    "simulate",
    "show",
    "generate_layout",
    "get_rack",
    "get_ruleset",
]

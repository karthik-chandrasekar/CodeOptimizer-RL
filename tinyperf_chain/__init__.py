"""TinyPerf-Chain: five ~2M role specialists for verified Python optimization.

An extension package on top of :mod:`tinyperf` (the TinyPerf-RL code base).  Nothing in the legacy
package is rewritten; the chain adds typed role interfaces, a budgeted deterministic controller, a
stricter verifier, a data factory, role trainers, an inference engine and an evaluation suite.

Role specialists (all share one tokenizer, none share weights or KV caches)::

    M1  target selector     -> ``TARGET fNN`` | ``NO_TARGET``
    M2  planner             -> ``PLAN <transform>; key=value; ...`` | ``NO_PLAN``
    M3  patch writer        -> ``<EDIT> one replacement function </EDIT>`` | ``ABANDON``
    M4  probe selector      -> ``PROBES pNN,pNN`` | ``PROBES none``
    M5  repairer            -> same payload as M3

Everything that decides correctness, speed, budgets, input validity or acceptance is deterministic code
outside the models (see :mod:`tinyperf_chain.controller` and :mod:`tinyperf_chain.adapters.verifier`).
"""

SCHEMA_VERSION = "chain_v1"
ROLES = ("M1", "M2", "M3", "M4", "M5")
__version__ = "1.0.0"

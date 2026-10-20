def get_property_adapter(name, config):
    if name == "caco2":
        from .caco2 import Caco2Adapter
        return Caco2Adapter(config)
    if name == "unbound_fraction":
        from .unbound_fraction import UnboundFractionAdapter
        return UnboundFractionAdapter(config)
    if name == "terminal_t_half":
        from .terminal_t_half import TerminalTHalfAdapter
        return TerminalTHalfAdapter(config)
    if name == "bioavailability":
        from .bioavailability import BioavailabilityAdapter
        return BioavailabilityAdapter(config)
    raise KeyError(f"Unsupported inference endpoint: {name}")

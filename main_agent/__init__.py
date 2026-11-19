"""Main agent (the Investigator) and the entry point that assembles the full agent."""

__all__ = ["AgentConfig", "ForensicAgent", "Trajectory", "build_agent", "load_config"]


def __getattr__(name):
    if name in {"AgentConfig", "ForensicAgent", "Trajectory"}:
        from main_agent.loop import AgentConfig, ForensicAgent, Trajectory

        return {"AgentConfig": AgentConfig, "ForensicAgent": ForensicAgent, "Trajectory": Trajectory}[name]
    if name in {"build_agent", "load_config"}:
        from main_agent.config import build_agent, load_config

        return {"build_agent": build_agent, "load_config": load_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Optional integration glue for llm-wiki.

Everything under this package is machine- or agent-CLI-specific plumbing that
sits on top of the core `llm_wiki` package: syncing an agent runtime's memory
directory from the vault (`agent_memory`), a deterministic daily ingest
pipeline (`ingest`), and macOS LaunchAgents to run that pipeline unattended
(`macos`). The core package works fully without any of this — nothing here is
required to build the index, run recall, or evaluate the vault.
"""

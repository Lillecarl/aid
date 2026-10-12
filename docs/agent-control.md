# Agent control authentication

The worker ROUTER carries worker reports and agent control requests over CURVE.
Each worker launch receives separate reporting and control keypairs. The daemon owns both key registries.
Only the reporting key can identify a worker. A control key identifies `agent:<public-key>` through ZAP.
The daemon resolves that public key to its session for every request.
Revocation therefore rejects requests from existing connections, not only new connections.

The launcher passes credentials through `WorkerArgs` on stdin and exports `AID_AGENT_CONNECTION` inside the worker.
Specs do not hold these credentials. External agents receive them even when `inherit_env` is false.
The built-in MCP server receives them in its private configuration.
`aid.connect()` selects authenticated control when credentials exist, including calls with explicit daemon paths.
An agent session without credentials fails closed. Invalid credentials never select the administrator socket.
Command workers use their configured endpoint and TLS trust for both reporting and control.

Agents retain read-only session queries and observation. The daemon stamps each agent message with its authenticated sender.
An agent can receive its own mailbox, compact its own conversation, and submit its own hooks.
Supervisor requests must name the authenticated caller. The daemon also checks the subject's supervision link.
Permission answers retain the tool allow-list. Interruptions retain the worker's turn check.
Agents cannot send ordinary prompts, stop sessions, answer their own permissions, or change plugin registrations.
Dynamic session creation requires a separate delegation policy and remains unavailable through agent control.

Keys rotate on worker restart. Worker exit and failed launch revoke control keys.
Resumed engines receive fresh MCP credentials. Hook fallback uses authenticated control and handles authentication rejection without waiting.
Local CLI and web clients retain administrator control. Plugin connections retain their own keys and grants.

This boundary authenticates protocol requests; it does not confine arbitrary code running as the daemon's Unix user.
Such code can open the local administrator socket or read another same-user process's files.
Observation remains shared. Protecting administrator sockets and private histories requires process confinement and a separate read-access policy.

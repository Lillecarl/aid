# Message delivery

The daemon records accepted messages in session history. `inbox.json` holds a dispatch checkpoint, not another message store.
Messages after the checkpoint wait for delivery. The daemon restores them and wakes their session after restart.

Worker startup occurs before dispatch. Failed startup leaves the messages pending.
The daemon checkpoints a batch before sending its wake prompt. A rejected socket send returns the batch to the inbox.
New messages remain behind that batch. An explicit session start also retries pending delivery.

The checkpoint names the dispatched turn until it ends. Restart records `delivery_uncertain` for an interrupted dispatch.
The session requests attention. The daemon does not replay that batch: the agent may already have applied effects.
Inspect the history before requesting that work again. This protocol does not promise exactly-once effects or recovery after power loss.

Interactive Claude consumes messages through `ReceiveMessages`. The checkpoint advances when the daemon takes that batch for its reply.
The channel has no acknowledgement from Claude. A lost reply does not cause automatic replay.

Existing sessions lack a checkpoint. Migration starts after their current history, because old messages may already have effects.
Migration cannot recover their previous in-memory inbox. Messages accepted after migration use the checkpoint.

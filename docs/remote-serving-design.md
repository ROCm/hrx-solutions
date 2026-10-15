# Remote serving boundary

The former `loom-serve-setup remote` interface, tracked SSH tunnels and Pi
configuration helpers are retired with the setup/start tooling. Loom Hem does
not provide remote provisioning, SSH target inventories or client configuration
commands. The earlier remote-serving design is not an implemented Hem interface.

Use the [current serving walkthrough](solutions/llm-serving.md) on the machine
that owns the GPU and model assets. An operator can separately arrange SSH
access, port forwarding and application configuration using their normal tools.
Preserve their usual SSH authentication and host verification behavior.

`loom-hem instance` commands can select a native endpoint with `--url URL`
instead of `--server NAME`. This does not transfer model files, prove remote
build compatibility or grant process ownership. Exported registration documents
contain server-local paths. Managed acquisition and registration require the
matching local build and asset context; observation of an endpoint does not
adopt its process into Hem.

For legacy deployments, keep the original tooling available until its services
and tunnels have been stopped deliberately. Old state is not automatically
imported, replayed or deleted. Register checkpoint/tokenizer pairs explicitly
with Hem and start a new empty server; there is no remote migration command.

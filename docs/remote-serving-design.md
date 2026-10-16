# Remote serving

The [running guide](solutions/llm-serving.md) describes operating a foreground
Loom Serve process directly. Follow [remote clients](serving/remote-clients.md)
for operator-created SSH forwarding, then [Pi configuration](solutions/llm-serving.md#connect-pi)
for a static client configuration against that endpoint.

Model assets, installed pipeline sources and deployment paths remain on the
inference host. Posting a registration document does not transfer files.
Provisioning, automatic restart and service installation are outside this manual.

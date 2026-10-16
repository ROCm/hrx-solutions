"""One complete portable module per declared model/data-type variant."""

def bytecode_catalogs(name, variants):
    linker = "@hrx_system//loom/src/loom/tools/loom-link"
    generator = "//build_tools:loom_catalog"
    native.genrule(
        name = "validate_catalog",
        srcs = native.glob(["llm/**/*.loom"]) + [
            "catalog_sources.bzl",
            "llm/qwen/qwen_08b/sources.txt",
            "llm/qwen/qwen_27b/sources.txt",
            "llm/qwen/qwen_08b/compatibility.json",
            "llm/qwen/qwen_27b/compatibility.json",
        ],
        tools = [generator],
        outs = ["catalog.validated"],
        cmd = "$(execpath %s) validate --inventory=$(location catalog_sources.bzl) --stamp=$@" % generator,
    )
    targets = []
    for architecture, variant in variants.items():
        prepare = "llm/qwen/" + architecture + "/prepare.loom"
        prepared = "prepare_" + architecture
        native.genrule(
            name = prepared,
            srcs = [prepare, ":validate_catalog"],
            tools = [generator],
            outs = ["intermediate/qwen/" + architecture + "/prepare.loom"],
            cmd = "$(execpath %s) prepare --source=$(location %s) --output=$@" % (generator, prepare),
        )
        inputs = [":" + prepared if "llm/" + path == prepare else "llm/" + path for path in variant["sources"]]
        target = "bytecode_" + architecture
        # Explicit external tool label avoids the pinned macro's caller-relative label.
        native.genrule(
            name = target,
            srcs = inputs + [":validate_catalog"],
            tools = [linker],
            outs = ["bytecode/" + variant["output"]],
            cmd = "$(execpath %s) %s --mode=merge --to=bc --output=$@" % (
                linker,
                " ".join(["$(location %s)" % path for path in inputs]),
            ),
        )
        targets.append(":" + target)
    native.filegroup(name = name, srcs = targets, visibility = ["//visibility:public"])

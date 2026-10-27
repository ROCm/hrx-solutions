"""One fixed Loom compiler; new catalog membership and closures are build data."""

def bytecode_catalogs(name, catalogs):
    linker = "@hrx_system//loom/src/loom/tools/loom-link"
    generator = "//build_tools:loom_catalog"
    sources = {}
    notices = {}
    jobs = {}
    for reference, catalog in catalogs.items():
        for notice in catalog["notices"]:
            notices[notice] = True
        for output, group in catalog["groups"].items():
            path = catalog["root"] + "/" + output
            if path in jobs and jobs[path] != group:
                fail("Conflicting compiler products: " + path)
            jobs[path] = group
            for source in group["sources"]:
                sources[source] = True
    native.genrule(
        name = "validate_catalog",
        srcs = sorted(sources) + sorted(notices) + sorted(catalogs) + ["catalog_sources.bzl"],
        tools = [generator],
        outs = ["catalog.validated"],
        cmd = "$(execpath %s) validate --inventory=$(location catalog_sources.bzl) --stamp=$@" % generator,
    )
    targets = []
    for index, (path, group) in enumerate(jobs.items()):
        inputs = group["sources"]
        policy = group["policy"]
        if policy != None:
            prepared = "prepare_program_" + str(index)
            native.genrule(
                name = prepared,
                srcs = [policy["source"], ":validate_catalog"],
                tools = [generator],
                outs = ["intermediate/" + str(index) + ".loom"],
                cmd = "$(execpath %s) prepare --source=$(location %s) --from-policy='%s' --policy='%s' --output=$@" % (
                    generator, policy["source"], policy["from"], policy["to"],
                ),
            )
            inputs = [":" + prepared if source == policy["source"] else source for source in inputs]
        target = "compiled_program_" + str(index)
        native.genrule(
            name = target,
            srcs = inputs + [":validate_catalog"],
            tools = [linker],
            outs = ["bytecode/" + path],
            cmd = "$(execpath %s) %s --mode=merge --to=bc --output=$@" % (
                linker, " ".join(["$(location %s)" % source for source in inputs]),
            ),
        )
        targets.append(":" + target)
    native.filegroup(name = name, srcs = targets, visibility = ["//visibility:public"])

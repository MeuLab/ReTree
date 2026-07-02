import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert retrieval traces to Sankey-ready JSON")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    nodes = []
    links = []
    node_index = {}

    def add_node(name):
        if name not in node_index:
            node_index[name] = len(nodes)
            nodes.append({"name": name})
        return node_index[name]

    query = add_node(trace["query"])
    relation = add_node(trace["relation_branch"])
    links.append({"source": query, "target": relation, "value": 1.0})
    for proto in trace["selected_prototypes"]:
        p = add_node(proto["name"])
        links.append({"source": relation, "target": p, "value": proto["score"]})
        for ent in proto["entities"]:
            e = add_node(ent["name"])
            links.append({"source": p, "target": e, "value": ent.get("score", 0.1)})
    Path(args.output).write_text(json.dumps({"nodes": nodes, "links": links}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Render docs/architecture.puml to SVG + PNG.
#
# Uses the locally-extracted Graphviz under tools/gv if present; otherwise relies
# on a system `dot` (install with: sudo apt install graphviz).
# PlantUML jar is expected at tools/plantuml-latest.jar (override with PLANTUML_JAR).
set -euo pipefail
cd "$(dirname "$0")/.."

GV="$PWD/tools/gv/usr"
if [ -x "$GV/bin/dot" ]; then
  export GRAPHVIZ_DOT="$GV/bin/dot"
  export LD_LIBRARY_PATH="$GV/lib/x86_64-linux-gnu:$GV/lib/x86_64-linux-gnu/graphviz"
  export GVBINDIR="$GV/lib/x86_64-linux-gnu/graphviz"
fi

JAR="${PLANTUML_JAR:-tools/plantuml-latest.jar}"
java -jar "$JAR" -tsvg docs/architecture.puml
java -jar "$JAR" -tpng docs/architecture.puml
echo "wrote docs/architecture.svg and docs/architecture.png"

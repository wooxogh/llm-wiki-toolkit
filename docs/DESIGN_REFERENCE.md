# Design Reference: Net Visualization UI/UX

Reference material collected while scoping the Tree+Graph visualization UX
described in the architecture spec (Section: Net Visualization UX —
Obsidian-like graph view with a Windows-Explorer-style tree pane).

## Visual references

### Obsidian
https://obsidian.md/
- **Graph view**: documents as nodes, links as edges, forming natural
  clusters. Closest existing reference for our base Net visualization.
- **File explorer (sidebar)**: Windows-Explorer-style folder/file tree.
  Reference for the Tree pane's edit UX (move/rename/delete).
- **Canvas**: free-placement whiteboard view, distinct from graph view.
  Lower priority — doesn't match our structural requirement.

### Logseq
https://logseq.com/
- Open-source alternative with a near-identical graph+outline structure.
  Unlike Obsidian, the implementation itself
  (`github.com/logseq/logseq`) is inspectable — useful if we need to see
  how an existing project solved the same rendering problem, not just
  screenshots.

### Roam Research
https://roamresearch.com/
- Originator of the bidirectional-linking concept. Reference for graph
  layout behavior at higher link density.

### Neo4j Bloom ⭐ highest-priority reference
https://neo4j.com/product/bloom/
- Professional graph-DB UI that color-codes and provides a legend per
  **relationship type**. Directly relevant: our NET has multiple typed
  edges (SUPPORTS/CONTRADICTS/SUPERSEDES/OVERRIDES/DUPLICATE_OF) that need
  the same kind of visual disambiguation.

### TheBrain
https://www.thebrain.com/
- Commercial mind-mapping tool showing hierarchy (tree) and cross-links
  (graph) in the same view simultaneously — conceptually close to our
  "Tree backbone + typed cross-links" requirement.

## Implementation library candidates

Both MIT-licensed, both have React bindings.

### Cytoscape.js (primary candidate)
https://js.cytoscape.org/
- Supports **compound graphs** (nested hierarchy) — fits the requirement
  to render Tree structure and typed cross-links in the same graph
  simultaneously.
- React wrapper: `react-cytoscapejs`.
- Live demos under the site's "Demos" menu.

### AntV G6
https://g6.antv.antgroup.com/en/examples
- Supports **Combo** (hierarchical grouping), native React integration via
  the Graphin toolkit (`antvis/Graphin`).
- More actively updated than Cytoscape.js as of this writing, richer
  built-in theming/animation.
- Live examples in the Gallery page.

### Lighter-weight alternative (not recommended as primary)
`react-force-graph` — simpler force-directed rendering, but weaker support
for per-edge-type styling and compound/hierarchical layout than the two
above.

## Recommendation

- Visual behavior reference: **Neo4j Bloom**'s relationship-type color +
  legend pattern.
- Implementation: **Cytoscape.js** as first choice given the compound-graph
  fit for Tree+typed-edges; AntV G6 as a fallback if richer theming or
  more active upstream maintenance becomes a priority.

## Verification note

Library capability claims above (compound graphs, Combo, React wrappers)
were confirmed against each project's own documentation. Screenshots
referenced under "Visual references" should be captured directly from the
linked pages by whoever assembles the design doc — this file intentionally
contains no embedded images.

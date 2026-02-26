# Layout Composition Patterns

Sub-layouts structure UI elements within panels using composable containers. Each container is a context manager that automatically positions widgets.

## Containers

### Row

Places children horizontally with `ImGui::SameLine()` between them.

```python
with layout.row() as row:
    row.button("A")
    row.button("B")
    row.button("C")
```

### Column

Standard vertical stacking (default ImGui behavior). Useful for applying state to a group of widgets.

```python
with layout.column() as col:
    col.enabled = False
    col.label("All children disabled")
    col.button("Can't click")
```

### Split

Two-column layout. The `factor` controls the width ratio of the first column.

```python
with layout.split(0.3) as split:
    split.label("Label")        # 30% width
    split.prop(self, "value")   # 70% width
```

### Box

Bordered container with theme-aware background.

```python
with layout.box() as box:
    box.heading("Section")
    box.prop(self, "setting")
```

### GridFlow

Responsive grid. Columns auto-calculated from available width if `columns=0`.

```python
with layout.grid_flow(columns=3) as grid:
    for item in items:
        grid.button(item.name)
```

## Nesting

Containers nest arbitrarily. Create sub-layouts from sub-layouts:

```python
with layout.box() as box:
    box.heading("Outer")
    with box.row() as row:
        row.button("A")
        with row.column() as col:
            col.label("Nested")
            col.button("B")
```

## State Cascading

State properties cascade from parent to child sub-layouts:

- `enabled` — ANDed: if parent is disabled, children are disabled
- `active` — ANDed: same semantics
- `alert` — one-shot: highlights the next widget with error styling

```python
with layout.column() as outer:
    outer.enabled = False       # everything below is disabled
    outer.prop(self, "a")

    with outer.row() as row:    # inherits disabled state
        row.button("X")        # disabled
        row.button("Y")        # disabled
```

## prop_enum

Toggle buttons that set a string property value. Selected state is styled with the primary theme color.

```python
with layout.row() as row:
    row.prop_enum(self, "mode", "fast", "Fast")
    row.prop_enum(self, "mode", "balanced", "Balanced")
    row.prop_enum(self, "mode", "quality", "Quality")
```

Also available directly on `layout`:

```python
layout.prop_enum(self, "mode", "fast", "Fast")
```

## Alert

Persistent styling that highlights all widgets while active. Cascades to child sub-layouts:

```python
with layout.column() as col:
    col.alert = value > threshold
    col.prop(self, "value")         # red if alert is True
    col.prop(self, "other_value")   # also red (alert persists)
    with col.row() as row:
        row.label("Child")          # also red (inherited from parent)
```

## Method Delegation

`SubLayout` exposes all `UILayout` widget methods. The ~60 most common are explicitly bound for performance. All others delegate via `__getattr__`:

```python
with layout.row() as row:
    row.same_line()         # delegated to UILayout
    row.begin_group()       # delegated to UILayout
    row.end_group()         # delegated to UILayout
```

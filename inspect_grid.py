import harmonica as hm

filename = "data/raw/261_TMI.grd"

print("=" * 60)
print("NGSA SHEET 261 — GRID INSPECTION")
print("=" * 60)

grid = hm.load_oasis_montaj_grid(filename)

print("\nOBJECT TYPE:")
print(type(grid))

print("\nDATA:")
print(grid)

print("\nDIMS / SHAPE:")
print(grid.dims, grid.shape if hasattr(grid, "shape") else "n/a")
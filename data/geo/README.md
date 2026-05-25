## France departments GeoJSON

The interactive map page (`dashboard/app.py`) can render a true department-level choropleth if you provide:

- `data/geo/fr_departements.geojson`

Expected format:
- Each feature represents a department.
- Department code must be available at: `feature.properties.code`
  - Example: `"01"`, `"2A"`, `"75"`, etc.

If the file is not present, the UI falls back to a ranked bar chart + a department selector.

Optional:
- You can also set `GEOJSON_URL` (environment variable) to a publicly accessible GeoJSON URL.
  The dashboard will download it at runtime and cache it.

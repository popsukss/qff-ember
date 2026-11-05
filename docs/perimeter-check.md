---
title: CWFIS perimeter temporal coverage and Yellowknife hotspot check
created: 2026-11-04
status: complete
---

# CWFIS perimeter check

**Verdict: B — the annual perimeter ZIPs are not daily histories; dated hotspots support reconstruction of approximate daily outlines.** This supports a project using reconstructed growth proxies, not a claim of forecasting independently observed daily fire boundaries.

Measured on 2026-11-04 from six downloaded perimeter archives and the complete 2023 hotspot archive. Other hotspot years were checked by HTTP HEAD only. Counts refer to CWFIS polygon IDs unless explicitly identified as NWT agency incidents.

## 1. Downloads, formats, CRS and attributes

All URLs below returned HTTP 200. Sizes are exact compressed bytes, from downloaded files for the six perimeter ZIPs and 2023 hotspots, and from HTTP Content-Length for the other hotspot ZIPs. The real directory is the [NRCan archive index](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/); the proposed path is correct. No requested ZIP returned 404.

| Perimeter archive / download URL | ZIP bytes | Uncompressed SHP bytes | Polygon records |
| --- | --- | --- | --- |
| [2018_perimeters.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2018_perimeters.zip) | 822,016 | 979,072 | 544 |
| [2019_perimeters.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2019_perimeters.zip) | 430,347 | 519,924 | 222 |
| [2020_perimeters.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2020_perimeters.zip) | 107,791 | 127,480 | 77 |
| [2021_perimeters.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2021_perimeters.zip) | 1,200,110 | 1,419,764 | 557 |
| [2022_perimeters.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2022_perimeters.zip) | 610,192 | 765,468 | 399 |
| [2023_perimeters.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2023_perimeters.zip) | 5,935,382 | 7,486,024 | 1144 |

| Hotspot archive / download URL | ZIP bytes | Inspection |
| --- | --- | --- |
| [2012_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2012_hotspots.zip) | 32,256,524 | HEAD only; contents not inspected |
| [2013_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2013_hotspots.zip) | 35,567,834 | HEAD only; contents not inspected |
| [2014_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2014_hotspots.zip) | 54,815,373 | HEAD only; contents not inspected |
| [2015_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2015_hotspots.zip) | 59,148,725 | HEAD only; contents not inspected |
| [2016_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2016_hotspots.zip) | 26,394,939 | HEAD only; contents not inspected |
| [2017_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2017_hotspots.zip) | 75,927,992 | HEAD only; contents not inspected |
| [2018_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2018_hotspots.zip) | 71,337,235 | HEAD only; contents not inspected |
| [2019_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2019_hotspots.zip) | 55,241,229 | HEAD only; contents not inspected |
| [2020_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2020_hotspots.zip) | 62,957,981 | HEAD only; contents not inspected |
| [2021_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2021_hotspots.zip) | 115,958,844 | HEAD only; contents not inspected |
| [2022_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2022_hotspots.zip) | 54,807,581 | HEAD only; contents not inspected |
| [2023_hotspots.zip](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2023_hotspots.zip) | 216,213,914 | Downloaded and parsed |

Each perimeter ZIP contains exactly `cc_apt_buf.shp`, `.shx`, `.dbf` and `.prj`; all 2,943 geometries read as Polygon and all passed the geometry validity check. Each is one annual layer, not a directory of daily layers. The 2023 hotspot ZIP contains one CSV, `2023_hotspots.csv`, **748,475,815 uncompressed bytes**, with **3,674,409 rows and 35 columns**.

**Perimeter CRS:** 2021–2023 resolve to EPSG:3978, NAD83 / Canada Atlas Lambert, in metres. The 2018–2020 PRJs declare NAD83 Lambert Conformal Conic with central meridian −95°, latitude of origin 49°, standard parallels 49°/77°, and zero false easting/northing; the reader does not assign an EPSG code to those WKT strings. They use the same listed projection parameters as EPSG:3978. Full WKT is retained in `check-results.json`.

**Hotspot coordinates:** CSV columns `lat` and `lon` are decimal degrees, not EPSG:3978 easting/northing. The CSV has no embedded CRS declaration. WGS84 / EPSG:4326 is an explicit processing assumption for these coordinates, not a measured datum declaration. The generic [hotspot metadata XML](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/fireM3_hotspots_metadata_NAP_ISO_19115_2003.xml) lists EPSG:3978 for the product; that does not make decimal degree CSV values projected metres.

### Every perimeter attribute

Names are uppercase in 2018–2022; 2023 uses lowercase and adds `consis_id`. Dates are stored as text, not native DBF date/time fields. Definitions for `uid`, `hcount`, `firstdate`, `lastdate`, and hectares are documented in [NRCan's perimeter attribute sheet](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/fireM3_buffered_hotspots_metadata_attributes_EN_FR.pdf).

**2018–2022 schema (examples from 2018):**

| Field | Read type | Three observed example values | Meaning |
| --- | --- | --- | --- |
| UID | int64 | 10779442; 10779292; 10779455 | Unique polygon identifier; not an agency incident number |
| HCOUNT | int64 | 81; 1021; 13 | Number of contributing hotspots, per provider |
| AREA | float64 | 1187.46; 11718.0; 367.347 | Provider area in hectares; divide by 100 for km² |
| FIRSTDATE | object | 2018-07-06 09:00:00; 2018-07-15 18:30:00; 2018-04-15 17:40:00 | Earliest contributing hotspot timestamp, UTC |
| LASTDATE | object | 2018-07-07 20:18:00; 2018-07-29 20:00:00; 2018-04-15 19:24:00 | Latest contributing hotspot timestamp, UTC |

**2023 schema:**

| Field | Read type | Three observed example values | Meaning |
| --- | --- | --- | --- |
| uid | int64 | 237811; 237812; 237813 | Unique polygon identifier; not an agency incident number |
| hcount | int64 | 6930; 3842; 6057 | Number of contributing hotspots, per provider |
| firstdate | object | 2023/06/02 16:24:00.000; 2023/06/19 16:11:00.000; 2023/06/02 17:15:00.000 | Earliest contributing hotspot timestamp, UTC |
| lastdate | object | 2023/06/26 17:13:00.000; 2023/09/06 08:18:00.000; 2023/08/10 18:13:00.000 | Latest contributing hotspot timestamp, UTC |
| area | float64 | 80096.0078; 69765.4141; 71176.9688 | Provider area in hectares; divide by 100 for km² |
| consis_id | int64 | 47303.0; 49167.0; 54006.0 | Additional numeric ID; definition and temporal stability not established by the attribute sheet |

The 2019–2022 files have the same five field names and read types as 2018. Per-file examples for all years are retained in the JSON. `consis_id` has one null; the other perimeter fields have none.

### Every 2023 hotspot attribute

Examples below are literal non-null values found while reading the complete CSV. Empty cells were treated as missing. Numeric zero is present in several fields and is not assumed to mean valid fire-season weather. Units/definitions explicitly documented for weather, fire indices and modelled behaviour come from [NRCan's hotspot attribute sheet](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/fireM3_hotspots_metadata_attributes_EN_FR.pdf). Other abbreviated fields are left unexpanded where that sheet does not define them. The current sheet lists some fields absent from this historical CSV; only the actual 35 archived columns are listed here.

| Column | Observed examples | Missing rows | Definition / qualification |
| --- | --- | --- | --- |
| rep_date | 2023-01-01 05:05:00; 2023-01-01 07:30:00; 2023-01-01 07:35:00 | 0 | Detection timestamp, UTC |
| satellite | Terra; NOAA-20; S-NPP | 204,879 | Satellite name |
| sensor | MODIS; VIIRS-I | 0 | Sensor name |
| source | NASA; USFS; HMS | 0 | Data provider |
| lat | 54.4253; 54.4075; 54.4049 | 0 | Latitude, decimal degrees |
| lon | -117.545; -117.536; -117.552 | 0 | Longitude, decimal degrees |
| frp | 36.68; 69.37; 45.04 | 2,095 | FRP field; units not verified in cited sheet |
| country | C; U; M | 0 | Country code; mapping not verified |
| agency | AB; OK; M | 0 | Province/territory or other agency code |
| ecozone | 9; 8; 14 | 1,410 | Ecozone code; mapping not verified |
| elev | 856; 874; 905 | 79,028 | Elevation field; units not verified here |
| slope | 1.14086; 3.78769; 3.61477 | 0 | Slope field; units not verified here |
| aspect | 4.3987; 359.433; 9.75548 | 0 | Aspect field; units not verified here |
| fuel | C2; M1_50; M1_65 | 346 | FBP fuel type |
| greenup | 0; 1; -1 | 15,953 | Greenup field; encoding not verified |
| pcuring | 100; 0; 55 | 0 | Curing field; encoding not verified |
| pconif | 100; 50; 65 | 0 | Conifer field; encoding not verified |
| sfl | 13.0676; 13.2173; 17.5855 | 79,374 | SFL field; definition not verified |
| cfl | 0.608459; 0.65313; 0.717342 | 2,612 | CFL field; definition not verified |
| cbh | (all missing) | 3,674,409 | CBH field; entirely missing |
| temp | 0; -4.85; -4.8 | 23 | Local-noon temperature, degrees |
| rh | 0; 74; 54 | 23 | Local-noon relative humidity, % |
| ws | 0; 12.09; 12.12 | 23 | Local-noon wind speed, km/h |
| wd | 0; 76; 210 | 23 | Local-noon wind direction, degrees |
| pcp | 0; 0.004; 0.459 | 23 | Local-noon previous 24-hour precipitation, mm |
| ffmc | 0; 83.377; 84.866 | 23 | Fine Fuel Moisture Code |
| dmc | 0; 7.98; 2.233 | 23 | Duff Moisture Code |
| dc | 0; 391.454; 80.369 | 23 | Drought Code |
| bui | 0; 15.186; 4.176 | 23 | Buildup Index |
| isi | 0; 4.317; 2.731 | 23 | Initial Spread Index |
| fwi | 0; 5.923; 1.382 | 0 | Fire Weather Index |
| sfc | 0; 3.5; 0.00463259 | 11,232 | Modelled surface fuel consumption, kg/m² |
| tfc | 0; 3.5; 0.00463259 | 11,232 | Modelled total fuel consumption, kg/m² |
| ros | 0; 0.02; 5.84 | 11,232 | Modelled spread rate, m/min |
| hfi | 0; 6132; 3046 | 11,232 | Modelled head fire intensity, kW/m |

## 2. Daily coverage and IDs

**No: these archives do not contain one polygon per fire per day.** `FIRSTDATE`/`firstdate` and `LASTDATE`/`lastdate` are the earliest and latest hotspot detections contributing to an outline; neither is a perimeter observation date. There is no snapshot date field and no NWT fire number field.

For 2023, there are **1,144 polygons and 1,144 distinct non-null `uid` values**: exactly one record per `uid`. This is a count of mapped hotspot clusters, not a verified count of distinct agency-managed fires. The latter cannot be recovered from these attributes alone.

| 2023 grouping / date interpretation | Groups | Distinct dates per group: min / median / max | Distribution |
| --- | --- | --- | --- |
| uid, distinct LASTDATE calendar day (proxy only) | 1144 | 1 / 1 / 1 | 1,144 groups with one date |
| uid, distinct FIRSTDATE value (endpoint only) | 1144 | 1 / 1 / 1 | 1,144 groups with one date |
| consis_id, distinct LASTDATE calendar day (proxy only) | 1141 | 1 / 1 / 2 | 1,139 groups with one date; 2 with two dates; one record has null ID |
| Actual dated perimeter snapshots | Not present | Not measurable | No snapshot date field |

The two repeated `consis_id` values are 52914 (LASTDATE 2023-06-04 and 2023-07-31: **57-day gap**) and 45211 (2023-04-29 and 2023-08-03: **96-day gap**). These are different polygon records, not evidence of a daily history. No `uid` has any consecutive-day perimeter pair. The 2023 first-to-last activity span is min/median/max **1/17/173 calendar days**, including both endpoints; this must not be counted as the number of observed perimeter days.

The [archived perimeter metadata](https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/fireM3_buffered_hotspots_metadata_NAP_ISO_19115_2003.pdf) describes season-to-date hotspot-derived outlines and daily maintenance. That operational update frequency does not imply that the annual ZIP retains each daily version.

## 3. Yellowknife fires and available area/distance measurements

Town reference point: **longitude −114.372°, latitude 62.454°**, as supplied. The [GNWT August 28 update](https://www.gov.nt.ca/en/public-safety/latest-updates) names ZF015 and ZF011 in the Behchokǫ̀/Yellowknife response. Four large surrounding outlines were spatially matched to the corresponding named incidents using [GNWT's official Fire History layer 81](https://www.apps.geomatics.gov.nt.ca/arcgis/rest/services/GNWT/BiologicEcologic_LCC/MapServer/81). The saved GeoJSON query selects `fireyear=2023` and fire IDs containing ZF015, ZF011, ZF012 or ZF085, with all attributes and `outSR=4326`. This is a measured spatial crosswalk, not an ID field carried in the CWFIS archive. ZF012 and ZF085 are included as nearby fires, not a claim that each independently threatened the town centre.

Area is shown two ways: provider `area / 100` and computed polygon area in the equal-area CRS EPSG:6933. The discrepancy is real in the downloaded 2023 data; its cause was not established. Minimum distance is from polygon boundary to the supplied town point in a WGS84 azimuthal-equidistant projection centred on that point. These are measurements of the one archived outline, **not measurements on LASTDATE or daily reconstructions**.

| NWT incident (spatial match) | CWFIS uid / consis_id | Outline inside NWT mapped burn | First hotspot UTC | Last hotspot UTC | Archived outlines / dated daily snapshots | Provider km² | Geometry km² | Min distance km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ZF015-23 | 242981 / 58857 | 77.7% | 2023-06-29 18:53:00+00:00 | 2023-09-13 09:24:00+00:00 | 1 / 0 | 1,384.19 | 1,780.90 | 24.909 |
| ZF011-23 | 253864 / 57493 | 80.8% | 2023-07-13 04:21:00+00:00 | 2023-09-12 20:27:00+00:00 | 1 / 0 | 408.65 | 525.78 | 30.470 |
| ZF085-23 | 253775 / 60002 | 76.2% | 2023-08-02 04:51:00+00:00 | 2023-09-24 10:09:00+00:00 | 1 / 0 | 507.51 | 652.56 | 32.919 |
| ZF012-23 | 241014 / 57494 | 76.6% | 2023-07-01 09:11:00+00:00 | 2023-09-10 21:06:00+00:00 | 1 / 0 | 153.78 | 197.88 | 46.962 |

For **every day in the activity spans above**, daily perimeter area and daily distance are unavailable. In particular, each date 2023-08-10 through 2023-08-20 has **N/A area and N/A minimum perimeter distance for all four incidents**. Repeating the final outline on each date would invent a growth history. Daily hotspot counts are available in section 5.

The nearest archived outline is consistent with ZF015-23: CWFIS uid **242981**, consis_id **58857**. Its first/last hotspot dates are June 29–September 13, with just one outline. Its **24.909 km** town distance must not be interpreted as the historical closest approach of the actual fire. The independent [GNWT wildfire review](https://www.gov.nt.ca/ecc/sites/ecc/files/resources/1._nwt_wildfire_response_review_report_and_appendices_final_compressed-9k.pdf) records ZF015 detection on June 28; an incident discovery date can precede CWFIS's first contributing detection.

## 4. Fires with at least 10 perimeter days and final area at least 10 km²

Qualification uses `uid`, distinct LASTDATE calendar days as the only available record-date proxy, and provider `AREA / 100 ≥ 10`. There is no true daily snapshot date, but the one-record-per-uid result is sufficient to rule out the threshold. The result is unchanged if FIRSTDATE or computed geometric area is used.

| Year | Unique uid / outlines | Provider final area ≥10 km² | Maximum stored record days per uid | ≥10 perimeter days AND ≥10 km² |
| --- | --- | --- | --- | --- |
| 2018 | 544 | 287 | 1 | 0 |
| 2019 | 222 | 108 | 1 | 0 |
| 2020 | 77 | 30 | 1 | 0 |
| 2021 | 557 | 309 | 1 | 0 |
| 2022 | 399 | 211 | 1 | 0 |
| 2023 | 1144 | 543 | 1 | 0 |

Thus the requested counts for **2018, 2019, 2020, 2021, 2022, 2023 are 0, 0, 0, 0, 0, 0** in these archives. This is not a count of fires that burned for at least ten days. It means none supplies ten dated perimeter records. The planned train/validation/test split supplies **zero directly observed growth-label sequences** from these files alone.

## 5. Yellowknife dated hotspots, 10–20 August 2023

**Yes.** `rep_date` is the UTC detection timestamp. All **3,674,409** rows have parseable dates, spanning **2023-01-01 05:05 UTC to 2023-12-31 22:35 UTC**. There is no `uid`, `consis_id`, agency fire ID, or other direct incident-join field in this CSV.

The following are **raw CSV row counts whose point intersects the respective final CWFIS polygon**, with no extra buffer, no temporal restriction to provider first/last endpoints, and no deduplication across sources, satellites or overpasses. They are spatial assignments, not official fire membership labels. A zero means no detected point inside that outline on that UTC day; it does not establish no burning.

| UTC date | ZF015-23 (uid 242981) | ZF011-23 (uid 253864) | ZF085-23 (uid 253775) | ZF012-23 (uid 241014) |
| --- | --- | --- | --- | --- |
| 2023-08-10 | 261 | 214 | 1809 | 12 |
| 2023-08-11 | 177 | 531 | 127 | 17 |
| 2023-08-12 | 110 | 150 | 269 | 36 |
| 2023-08-13 | 1318 | 135 | 855 | 191 |
| 2023-08-14 | 1132 | 538 | 896 | 711 |
| 2023-08-15 | 2012 | 1212 | 569 | 300 |
| 2023-08-16 | 1485 | 1090 | 1013 | 212 |
| 2023-08-17 | 96 | 15 | 2 | 0 |
| 2023-08-18 | 25 | 5 | 74 | 0 |
| 2023-08-19 | 22 | 12 | 6 | 1 |
| 2023-08-20 | 8 | 0 | 1 | 0 |

| Matched uid | All-year spatially matched rows | Detection days | First / last matched day | Intervals >1 day / largest interval | 10–20 Aug rows |
| --- | --- | --- | --- | --- | --- |
| 242981 | 32218 | 70 | 2023-06-29 / 2023-09-13 | 3 / 5 days | 6646 |
| 253864 | 13255 | 40 | 2023-07-13 / 2023-09-10 | 3 / 16 days | 3902 |
| 253775 | 19427 | 45 | 2023-08-02 / 2023-09-24 | 6 / 4 days | 5621 |
| 241014 | 4713 | 39 | 2023-07-01 / 2023-09-10 | 5 / 16 days | 1480 |

For uid 242981, **6,646 rows occur on all 11 days** in the requested window; the peak is **2,012 rows on August 15**. Across its 77-day first-to-last calendar span there are **70 detected days and 7 missing days**, with three multi-day intervals and a maximum **5-day interval between detections** (four intervening missing days). Thus even this large fire's hotspot sequence is gappy outside the requested window.

The spatial count for uid 242981 is **32,218**, versus stored `hcount=31,790`; those counts are not interchangeable. The CSV and polygons may involve different source selection, filtering or boundary inclusion. The measured result is the stated spatial rule, with the mismatch retained rather than silently forced to agree.

## 6. Verdict and project implication

**B — perimeters not daily, but daily hotspots allow rebuilding approximate daily outlines.** The data contain enough dated, spatially relevant detections for a Yellowknife reconstruction experiment. A reproducible method could accumulate detections through each UTC day, apply a documented buffer/concave-outline rule, and track growing clusters through merges. That would create estimated daily boundaries; the method and its accuracy have not been validated in this check.

For the eight-sector +1/+3-day task, build boundaries from detections available through each cutoff, retain missing-observation masks, and evaluate against held-out reconstructed targets with that limitation stated. A final-season footprint may support retrospective event grouping, but using future detections or final shapes to construct an input boundary would leak future information. Reconstructed labels share a measurement process with hotspot inputs, so skill against those labels does not establish skill against an independent physical fire edge. Satellite visibility, repeated detections, spatial resolution and incident merges remain material limitations; NRCan characterises these outlines as approximate, especially suitable for large fires.

The archive URL checks cover 2012–2023 hotspots, but only **2023 hotspot contents** were measured. Counts of fires eligible under a reconstructed ≥10-hotspot-day rule across 2018–2023 remain unmeasured; the zeros in section 4 apply only to supplied perimeter records. Proceed with B only if reconstructed growth proxies are acceptable; these ZIPs do not provide the daily perimeter ground truth assumed in A.

### Reproduction and retained evidence

Downloads and all derived raw evidence are in `data/raw/cwfis_perimeters/`; the local environment is `.venv/`. Both paths are git-ignored. No ZIP members were extracted or executed. The checker reads archives in place, uses chunked CSV reading, records SHA-256 values, and accepts input paths as arguments. Its JSON includes complete per-year attribute examples, missing counts, ID/date distributions and every matched hotspot-day count.

```bash
cd ~/Projects/qff-ember
.venv/bin/python -I scripts/check_perimeters.py \
  --perimeters data/raw/cwfis_perimeters/{2018,2019,2020,2021,2022,2023}_perimeters.zip \
  --hotspots data/raw/cwfis_perimeters/2023_hotspots.zip \
  --reference-perimeters data/raw/cwfis_perimeters/nwt-yellowknife-2023.geojson \
  --pdf data/raw/cwfis_perimeters/perimeters_metadata.pdf \
        data/raw/cwfis_perimeters/perimeters_attributes.pdf \
        data/raw/cwfis_perimeters/hotspots_attributes.pdf \
  --output data/raw/cwfis_perimeters/check-results.json
```

Dependencies: geopandas, pandas, pyogrio, shapely; pypdf is optional for the `--pdf` metadata extraction. `download-manifest.json` records the 18 archive URLs, HTTP status and ZIP sizes; `*.headers.txt` retain response headers. `archive-index.html`, metadata PDFs/XML and the GNWT layer JSON/GeoJSON retain source evidence. The checker performs no network requests and never executes downloaded content.

Access limitations: the new catalogue product pages were inaccessible during lookup; the older buffered metadata endpoint returned a page without usable field definitions. Direct archive and documentation downloads succeeded, and the report uses those instead. No requested archive was missing.

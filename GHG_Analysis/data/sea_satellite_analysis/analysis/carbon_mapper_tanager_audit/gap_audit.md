# Carbon Mapper–Tanager Southeast Asia Gap Audit

**Audit date (UTC):** 2026-07-21T08:12:05.331749Z  
**Fixed cohort:** 385 Carbon Mapper Tanager CH₄ plume observations, 2024-10-26 to 2026-06-08  
**Primary source definition:** Carbon Mapper API clustering with `eps=500` m  
**Purpose:** identify what the public Tanager record establishes, what is missing, and which gaps justify a designed 30-acquisition Southeast Asian campaign.

## Executive finding

The headline number is **385 plume observations, not 385 sites**. Carbon Mapper's 500 m source grouping maps them to **93 source clusters**. Only **229/385 (59.5%)** have a public emission rate and uncertainty; **156/385 (40.5%)** are hidden or unquantified. The API does not expose a reason code for those missing rates, so this audit cannot say whether each case failed because of wind, retrieval quality, plume morphology, policy, or another rule.

At source level, 12/93 clusters have no public rate for any cohort detection, 42/93 have a mixture of public and hidden rates, and 39/93 are quantified on every cohort detection. There are 50 sources detected on at least two dates, and every one has at least one public rate. The 12 never-quantified sources each appear on only one cohort date, so a deliberate revisit is needed to test whether their missing quantification was scene-specific or persistent.

The larger strategic gap is a denominator-detail problem. Carbon Mapper's public source summaries provide qualifying observation-date counts and persistence, using a documented default maximum cloud cover of 25%. However, the unauthenticated API does not expose a complete region-wide record of tasking attempts, rejected or unpublished acquisitions, source-level usable pixels, or scene-specific minimum detection limits; the detailed scene-coverage query returned `401 Unauthorized` during this audit. Therefore, **absence from these 385 records cannot be called a Tanager non-detection, and a country with zero records cannot be called emission-free**. This directly supports the rationale's proposed tropical-observability experiment rather than another plume portal.

## 1. Cohort and source definitions

The cohort is held fixed to the IDs saved by `01_sea_data_census.ipynb`; live global counts are not substituted. Every ID was looked up again through the Carbon Mapper annotated-plume endpoint. Source counts are reported at three clustering distances because the word *site* is not a sensor measurement-it depends on the spatial grouping rule. Carbon Mapper's `source_name` is a machine-generated identifier for an ephemeral spatial cluster, not a verified facility or operator name.

| cluster_distance_m | source_count | cohort_plumes_mapped | cohort_plumes_unmapped | interpretation |
| --- | --- | --- | --- | --- |
| 500 | 93 | 385 | 0 | Primary Carbon Mapper-style facility grouping |
| 1000 | 86 | 385 | 0 | Moderate grouping sensitivity |
| 5000 | 81 | 385 | 0 | Comparable radius to the earlier cross-sensor notebook |

At 500 m, 0 cohort plumes are unmapped and 0 map to more than one source. The earlier notebook's one-pass 5 km result should not be used as the authoritative site count because single-linkage grouping can depend on row order; the API grouping above is explicit and reproducible.

## 2. What is missing from the 385 published plume records-

| level | field_or_capability | available_count | missing_count | missing_percent |
| --- | --- | --- | --- | --- |
| Plume | Live API scene timestamp | 301 | 84 | 21.8% |
| Plume | Analysis timestamp after saved-cohort fallback | 385 | 0 | 0.0% |
| Plume | Public emission rate | 229 | 156 | 40.5% |
| Plume | Public emission uncertainty | 229 | 156 | 40.5% |
| Plume | Plume quality value | 0 | 385 | 100.0% |
| Plume | Wind speed | 385 | 0 | 0.0% |
| Plume | Wind direction | 385 | 0 | 0.0% |
| Plume | Specific sector code | 365 | 20 | 5.2% |
| Plume | Mission phase | 385 | 0 | 0.0% |
| Plume | Sensitivity mode | 385 | 0 | 0.0% |
| Plume | Ground sample distance | 385 | 0 | 0.0% |
| Plume | Off-nadir angle | 385 | 0 | 0.0% |
| Plume | Processing software version | 385 | 0 | 0.0% |
| Plume | Emission product version | 385 | 0 | 0.0% |
| Plume | Plume image | 385 | 0 | 0.0% |
| Plume | Concentration raster | 385 | 0 | 0.0% |
| Plume | Publication timestamp | 385 | 0 | 0.0% |
| Plume | Non-empty publication-source list | 0 | 385 | 100.0% |
| Plume | Carbon Mapper 500 m source match | 385 | 0 | 0.0% |
| Source | Carbon Mapper spatial-cluster identifier | 93 | 0 | 0.0% |
| Source | Observation-date denominator | 93 | 0 | 0.0% |
| Source | Detection-date count | 93 | 0 | 0.0% |
| Source | Persistence metric | 93 | 0 | 0.0% |

The live API omits `scene_timestamp` for 84/385 records even though the saved cohort contains those dates. The audit preserves this as a missingness result and uses the saved values only as an explicit analysis fallback. The `plume_quality` field contains a value for 0/385 records. A null field is not interpreted as a bad plume; it means the public response does not supply that label. A non-empty `publication_sources` list appears on 0/385 records. Carbon Mapper spatial-cluster identifiers are present for 93/93 source clusters, and an observation-date denominator is present for 93/93 clusters.

Using the live timestamp where present and the documented saved-cohort fallback otherwise, the median delay from acquisition to publication is 30.0 days and the 95th percentile is 107.0 days. This describes this fixed public cohort; it is not a guaranteed service-level commitment.

### Quantification context (descriptive, not causal)

| dimension | category | plume_count | quantified_count | hidden_or_unquantified_count | quantified_percent |
| --- | --- | --- | --- | --- | --- |
| Sensitivity mode | glint | 10 | 10 | 0 | 100.0% |
| Sensitivity mode | maximum_sensitivity | 142 | 87 | 55 | 61.3% |
| Sensitivity mode | medium_sensitivity | 38 | 24 | 14 | 63.2% |
| Sensitivity mode | standard_sensitivity | 195 | 108 | 87 | 55.4% |
| Mission phase | first_light | 41 | 23 | 18 | 56.1% |
| Mission phase | production | 344 | 206 | 138 | 59.9% |
| Absolute off-nadir angle | 0-5° | 44 | 25 | 19 | 56.8% |
| Absolute off-nadir angle | >5-15° | 122 | 74 | 48 | 60.7% |
| Absolute off-nadir angle | >15-30° | 216 | 129 | 87 | 59.7% |
| Absolute off-nadir angle | >30° | 3 | 1 | 2 | 33.3% |
| Wind speed used | 0-2 m/s | 178 | 97 | 81 | 54.5% |
| Wind speed used | >2-4 m/s | 161 | 100 | 61 | 62.1% |
| Wind speed used | >4-6 m/s | 39 | 27 | 12 | 69.2% |
| Wind speed used | >6 m/s | 7 | 5 | 2 | 71.4% |
| Plumes in same scene | 1 plume | 150 | 91 | 59 | 60.7% |
| Plumes in same scene | 2-3 plumes | 172 | 110 | 62 | 64.0% |
| Plumes in same scene | 4+ plumes | 63 | 28 | 35 | 44.4% |
| Offshore flag | Not flagged offshore | 374 | 218 | 156 | 58.3% |
| Offshore flag | Offshore | 11 | 11 | 0 | 100.0% |
| Acquisition year | 2024 | 3 | 1 | 2 | 33.3% |
| Acquisition year | 2025 | 246 | 144 | 102 | 58.5% |
| Acquisition year | 2026 | 136 | 84 | 52 | 61.8% |

The percentage differences above show where hidden/unavailable rates are concentrated in this cohort. They do **not** identify the reason a rate is missing: sensitivity mode, geometry, wind, mission phase, source type, scene complexity, and publication rules can be confounded. Carbon Mapper does not expose a per-record suppression reason, so causal language would be unjustified.

### What the public source-level denominator does provide

At the 500 m source definition, 68/93 sources have at least two qualifying observation dates and 27/93 have at least one qualifying observation date without a detection in the provider summary. The median source has 2.0 qualifying observation dates, and the median provider persistence is 1.00. These aggregates are valuable, but they do not replace scene-level tropical usability and detection-limit metadata.

| metric | value | unit |
| --- | --- | --- |
| source_count | 93.0 | sources |
| sources_with_observation_denominator | 93.0 | sources |
| sources_with_two_or_more_qualifying_observation_dates | 68.0 | sources |
| sources_with_at_least_one_qualifying_null_date | 27.0 | sources |
| median_qualifying_observation_dates | 2.0 | dates per source |
| maximum_qualifying_observation_dates | 15.0 | dates |
| median_detection_dates | 2.0 | dates per source |
| median_persistence | 1.0 | fraction |
| minimum_persistence | 0.25 | fraction |
| maximum_persistence | 1.0 | fraction |
| sources_with_persistence_equal_to_one | 66.0 | sources |

## 3. Geographic coverage-and what it does not prove

Land-country assignment uses each API point and Natural Earth boundaries. Of the 385 rectangle-filtered records, 345 are on ASEAN land and 40 fall outside ASEAN land or cannot be assigned to a land polygon. The 345 ASEAN-land detections map to 77 primary source clusters. The ASEAN countries with at least one land detection are Indonesia, Philippines, Vietnam, Malaysia, Thailand, Cambodia, and Myanmar. The ASEAN countries with no land detection in the fixed cohort are Brunei, Laos, Singapore, and Timor-Leste. The rectangular notebook frame also contains detections assigned to Hong Kong, Offshore / boundary-unassigned, and People's Republic of China; these must not be described as ASEAN detections.

| country | detections | sources_500m | quantified | hidden_or_unquantified | quantified_percent |
| --- | --- | --- | --- | --- | --- |
| Indonesia | 108 | 26 | 47 | 61 | 43.5% |
| Philippines | 72 | 8 | 47 | 25 | 65.3% |
| Vietnam | 47 | 13 | 26 | 21 | 55.3% |
| Malaysia | 46 | 11 | 28 | 18 | 60.9% |
| Thailand | 32 | 9 | 21 | 11 | 65.6% |
| Cambodia | 20 | 1 | 10 | 10 | 50.0% |
| Myanmar | 20 | 9 | 13 | 7 | 65.0% |
| Brunei | 0 | 0 | 0 | 0 |  |
| Laos | 0 | 0 | 0 | 0 |  |
| Singapore | 0 | 0 | 0 | 0 |  |
| Timor-Leste | 0 | 0 | 0 | 0 |  |

A zero in this table combines several unknowns: the satellite may not have been tasked, an acquisition may have been unusable, a plume may have been below the scene threshold, an emitter may have been inactive, or no source may have been present. The 30-image case should be framed as an experiment that separates these possibilities, not as a map of where methane does or does not exist.

## 4. Sector concentration

| sector | sector_label | detections | sources_500m | detection_share_percent | quantified_percent |
| --- | --- | --- | --- | --- | --- |
| 6A | Solid waste / landfill | 339 | 63 | 88.1% | 59.0% |
| 1B2 | Oil and gas | 18 | 16 | 4.7% | 77.8% |
| other | Other / unclassified | 18 | 8 | 4.7% | 44.4% |
| 1B1a | Coal mining | 7 | 3 | 1.8% | 57.1% |
|  | Unclassified | 2 | 2 | 0.5% | 100.0% |
| 6B | Wastewater | 1 | 1 | 0.3% | 100.0% |

The record is highly concentrated in whatever sectors Carbon Mapper has already targeted and published. Raw counts therefore describe the public acquisition-and-detection record, not the true regional sector mix. This is precisely why the new acquisition portfolio should reserve images for underrepresented POME/wastewater, coal, oil-and-gas, urban, coastal, and offshore environments instead of simply revisiting the sector with the most existing detections.

## 5. Gaps that cannot be counted from the current endpoints

| gap_category | not_available_in_public_endpoints | why_it_matters |
| --- | --- | --- |
| Observation inventory | Complete public region-wide scene inventory and per-scene source coverage detail | Public source summaries provide aggregate qualifying observation dates, but the detailed scene-coverage query required authentication in this audit. |
| Tasking history | Requested, acquired, rejected, and unpublished attempts | Needed to measure geographic and seasonal selection bias. |
| Scene usability | Per-source cloud, haze, valid-pixel fraction, and usable area | The source query uses a documented cloud threshold, but does not expose enough public detail to calculate tropical valid-observation yield. |
| Sensitivity | Scene- and source-specific minimum detection limit | Needed before a non-detection can be interpreted as a valid null. |
| Retrieval diagnostics | Local background noise, artifact flags, and dual-window agreement | Needed to compare reliability across vegetation, dark ponds, cities, coasts, and glint. |
| Quantification provenance | Reason code when emission_auto is hidden or unavailable | Needed to separate wind failure, plume-shape failure, QC suppression, and policy suppression. |
| Illumination | Sun elevation and azimuth in the public plume record | Needed to diagnose radiance and surface-dependent retrieval performance. |
| Wind provenance | Wind model, analysis time, spatial resolution, and alternative-wind spread | Needed to reproduce and interpret emission-rate uncertainty. |
| Surface context | Land cover and retrieval-background class | Needed to quantify performance by tropical source environment. |
| Action readiness | Verified facility/operator, stakeholder, intervention, response, and mitigation status | Needed to turn detection evidence into a defensible action pathway. |

These are not criticisms of the quality of Carbon Mapper's published plume products. They are variables required for the narrower research question in `SEA_PROJECT_RATIONALE_AND_NEXT_STEPS.md`: how tropical observing conditions affect valid-observation yield and how scarce Tanager tasking should be allocated.

## 6. Evidence-led case for additional Tanager acquisitions

The audit supports a strong but bounded argument:

1. **Tanager is already uniquely productive in the SEA study frame.** The fixed rectangle-filtered cohort contains 385 facility-scale CH₄ detections across 93 Carbon Mapper source clusters; 345 detections and 77 source clusters are on ASEAN land. This demonstrates that the sensor can find actionable-scale emitters in a difficult region while correcting the earlier geographic shorthand.
2. **Detection is not yet equivalent to quantification.** 156 plume records lack a public rate and uncertainty, and 12 source clusters are never publicly quantified in this cohort. Repeat acquisition under different wind, cloud, illumination, and surface conditions can test which gaps are recoverable.
3. **Country and sector coverage is opportunity-driven.** The public positives cannot identify true absence because the full observation denominator is unavailable. New acquisitions should deliberately fill geographic and sector cells, with failed and usable observations recorded as outcomes.
4. **Tropical reliability is the novel contribution.** Pair operational products with native-radiance diagnostics, explicit valid-null criteria, and scene-specific limits across landfills, dark POME ponds, vegetation, cities, coasts, and offshore glint.
5. **The 30 images should be a portfolio, not a top-30 emitter list.** Use some for first looks in missing country/sector cells, some for repeat-detected but never-quantified sources, some for wet/dry seasonal pairs, and some as quantified benchmark controls. The exact split should follow target feasibility and stakeholder readiness.

### Provisional 30-image design envelope

This allocation is a testable starting constraint derived from the audit, not the final target list. Facility coordinates still require the separate actionable-source registry described in the project rationale. Final selections should combine objectives where possible, but each image must have one primary purpose so the portfolio can be evaluated honestly.

| portfolio_component | image_count | evidence_from_audit | selection_rule | intended_learning |
| --- | --- | --- | --- | --- |
| Missing-country first looks | 8 | Brunei, Laos, Singapore, and Timor-Leste have zero ASEAN-land detections in the fixed cohort. | Two facility candidates per missing country, chosen from an external actionable-source registry. | Separate geographic tasking gaps from source absence and test transfer to new tropical settings. |
| Quantification-recovery revisits | 8 | 12 source clusters have no public rate and each appears on only one cohort date. | Revisit eight Tier-2 sources balanced across countries, sectors, wind regimes, and surfaces. | Test whether a second acquisition converts a location-only detection into a quantified result. |
| Underrepresented sector and surface stress tests | 6 | 88.1% of detections are landfill-coded; wastewater, coal, oil/gas, POME-like dark ponds, coasts, and offshore settings are sparse. | Choose distinct actionable source classes and retrieval backgrounds; do not add more landfill scenes by default. | Measure transferability and artifact behavior outside the dominant landfill sample. |
| Wet/dry seasonal pairs | 4 | The public positive catalogue cannot isolate monsoon effects or valid-observation yield. | Select two high-priority sites and acquire one wet-season and one dry-season observation at each. | Test cloud, moisture, surface, and source-seasonality effects using paired sites. |
| Quantified controls and cloud contingency | 4 | 39 source clusters are quantified on every cohort detection, while tropical acquisitions remain weather-risky. | Reserve benchmark revisits and allow failed cloudy acquisitions to be retried without breaking the design. | Anchor retrieval comparisons and protect the experiment from predictable tropical data loss. |

The generated `revisit_candidates.csv` is therefore a triage list, not a final allegation or tasking plan. Tier 1 contains sources detected on multiple dates but never publicly quantified; Tier 4 contains quantified sources that can serve as controls. New-location candidates still require a separate facility registry because a detection-only catalogue cannot list facilities Tanager never observed.

## 7. Data-version and interpretation notes

Compared with the saved census CSV, 0 rates became available, 0 became unavailable, 0 published numeric rates changed, and 0 hidden-rate flags changed in the live API snapshot. However, 84 currently published records no longer expose `scene_timestamp` even though the saved extract contains it. This confirms that the snapshot date and fallback rule belong beside every statistic.

Guardrails:

- `hide_emission=true` or a null rate is reported as *hidden/unavailable*, not as a failed retrieval unless Carbon Mapper provides a reason.
- `plume_quality=null` is reported as a missing public value, not as bad quality.
- Carbon Mapper persistence and observation counts are provider source-level summaries; the cohort-only detection counts are kept separate.
- Country assignment is a derived land-polygon join. Offshore points remain `Offshore / boundary-unassigned` and are not forced into the nearest jurisdiction.
- Instantaneous plume rates are not annualized, and source recurrence is not claimed to prove continuous emissions.
- The API cluster identifier is not a facility/operator name. Any later operator attribution requires separate evidence and is not a legal or compliance finding.

## 8. Reproducibility and sources

- Carbon Mapper Data API documentation: https://api.carbonmapper.org/api/v1/docs
- Carbon Mapper product guide: https://carbonmapper.org/articles/product-guide
- Fixed cohort: `data/sea_satellite_analysis/raw_extracts/tanager_plumes_sea.csv`
- Exact plume API snapshot: `data/sea_satellite_analysis/raw_extracts/carbon_mapper_tanager_sea_plumes_api_2026-07-21.json`
- Source API snapshot: `data/sea_satellite_analysis/raw_extracts/carbon_mapper_tanager_sea_sources_api_2026-07-21.json`
- Script: `scripts/analyze_carbon_mapper_tanager_sea.py`
- Country polygons: Natural Earth 1:10m Admin 0 Countries (public domain)

All Carbon Mapper statistics in this report are computed from the public API snapshot identified above. Carbon Mapper should be cited as the data provider and its current Terms of Use should be checked before redistribution.

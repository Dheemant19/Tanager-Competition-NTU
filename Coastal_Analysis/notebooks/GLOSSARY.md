# Coastal Analysis Glossary

A plain-language reference for the terms, units, and data products used in the notebooks. Tags identify the notebook where a term is introduced: N01 through N07, followed by the CoastCheck capstone. The sequence is `01_explore_scene.ipynb`, `02_spectral_exploration.ipynb`, `03_water_quality.ipynb`, `04_quantitative_turbidity.ipynb`, `05_ai_water_quality.ipynb`, `06_sentinel2_acolite_validation.ipynb`, `07_uncertainty.ipynb`, and `coastcheck_sangatta.ipynb`.

## Sensor and data model

**Hyperspectral imaging spectroscopy** `[N01]`  
A measurement with many narrow, contiguous spectral bands, so each pixel has a reflectance spectrum rather than a few broad colours.

**Tanager-1** `[N01]`  
Planet imaging spectrometer used here. The surface-reflectance cube has 426 bands from about 380 to 2500 nm with about 5 nm sampling.

**Band** `[N01]`  
One spectral channel, measured over a small wavelength interval.

**Wavelength (lambda), nanometre (nm)** `[N01]`  
Position in the spectrum. Visible light spans roughly 400 to 700 nm.

**VNIR and SWIR** `[N02]`  
VNIR is visible and near-infrared light, roughly 400 to 1000 nm. SWIR is shortwave infrared, roughly 1000 to 2500 nm. Most water-quality signal is in the VNIR because water absorbs strongly beyond about 750 nm.

**FWHM** `[N02]`  
Full width at half maximum, the width of a band response at half of its peak sensitivity. It describes spectral sharpness.

**good_wavelengths** `[N02]`  
A per-band cube flag: 1 is usable and 0 is unreliable, usually because of atmospheric water-vapour absorption.

**GSD** `[N01]`  
Ground sample distance, the ground width represented by one pixel. The Tanager scene is about 33 m per pixel.

## Geospatial and file concepts

**STAC** `[N01]`  
SpatioTemporal Asset Catalog, a JSON catalogue standard with catalog, collection, item, and asset levels.

**Collection, item, and asset** `[N01]`  
A collection groups scenes. An item is one satellite acquisition. An asset is a downloadable file attached to that item.

**Inventory** `[N01]`  
The local scene table at `data/inventory/tanager_inventory.csv`, containing metadata and asset URLs.

**bbox and centroid** `[N01]`  
A bounding box gives west, south, east, and north coordinates. A centroid is its geographic centre.

**Orthorectified** `[N01]`  
Geometrically corrected so pixels align to map coordinates. Tanager `ortho_*` assets are orthorectified; `basic_*` assets are not.

**CRS, EPSG, and UTM 50N** `[N01]`  
A coordinate reference system defines map coordinates. This scene uses WGS84 / UTM zone 50N, EPSG:32650.

**GeoTIFF** `[N01]`  
A TIFF image carrying geospatial metadata such as CRS and pixel-to-map transform.

**dtype** `[N01]`  
The numeric type stored per pixel. `uint8` holds 0 to 255 integers; `float32` and `float64` hold decimal values.

**nodata** `[N01]`  
A reserved value for pixels without a valid measurement. Exclude it from statistics and image stretches.

**geolocation_array** `[N01]`  
A two-band GeoTIFF of longitude and latitude for the basic grid, used to georeference non-orthorectified products.

## Image processing

**Alpha band** `[N01]`  
A visual-image validity channel. Zero marks the off-swath border; positive values mark real image data.

**Percentile contrast stretch** `[N01]`  
A display rescaling that maps selected low and high percentiles to 0 and 1. It reveals dark water and plume structure but must not be used as retrieval input.

**Sun elevation, sun azimuth, and off-nadir angle** `[N01]`  
Acquisition geometry that affects illumination, distortion, and sun-glint risk.

**UDM** `[N01]`  
Usable-Data Mask. It flags cloud, haze, shadow, and nodata pixels so retrievals use only clear water.

## Radiometry

**Radiance** `[N02]`  
Measured light energy per area, solid angle, and wavelength. Tanager `ortho_radiance_hdf5` is top-of-atmosphere radiance and still includes atmospheric effects.

**Reflectance (rho)** `[N02]`  
The fraction of incoming light reflected by a surface. It is dimensionless and is the input expected by water-quality algorithms.

**Surface reflectance** `[N02]`  
Reflectance after atmospheric correction. Tanager `ortho_sr_hdf5` is the primary product for the analysis.

**Atmospheric correction** `[N02]`  
Removal of atmospheric scattering and absorption from satellite measurements. Small residual errors matter over dark water.

**ISOFIT** `[N04]`
Imaging Spectrometer Optimal FITting, the optimal-estimation atmospheric correction used for Tanager surface reflectance. Its standard product treatment is land-oriented rather than water-specific.

**surface_reflectance_uncertainty** `[N04]`  
The ISOFIT per-pixel, per-wavelength uncertainty layer. It represents a measurement-noise floor, not total retrieval error.

**Steradian (sr)** `[N02]`  
The SI unit of solid angle, used in radiance and remote-sensing reflectance.

## Water optics

**Water surface reflectance (rho_w)** `[N02]`  
The water reflectance supplied by the surface-reflectance cube. Its Lambertian treatment is an approximation for water.

**Rrs** `[N05]`  
Remote-sensing reflectance, defined as water-leaving radiance divided by downwelling irradiance. This analysis uses the approximation `Rrs ≈ rho_w / pi` when preparing MDN inputs.

**Water-leaving radiance and downwelling irradiance** `[N02]`  
Water-leaving radiance is light exiting water with constituent information. Downwelling irradiance is the light arriving at the surface from above.

**Case-1 and Case-2 water** `[N02]`  
Case-1 water is mainly controlled by phytoplankton. Case-2 coastal or estuarine water is affected by chlorophyll, suspended sediment, and CDOM that vary independently. The Sangatta scene is Case-2.

**Red edge** `[N02]`  
The sharp red-to-NIR reflectance transition. A small peak near 700 to 710 nm can indicate chlorophyll in water.

**Black-pixel assumption** `[N02]`  
Clear water is nearly black in the NIR, so NIR brightness over water can indicate glint or an atmospheric-correction residual.

**O2-A absorption band** `[N02]`  
An atmospheric oxygen feature near 760 nm that appears in top-of-atmosphere radiance and is removed from surface reflectance.

**Streaming read** `[N02]`  
Reading one two-dimensional band at a time rather than loading the full hyperspectral cube into memory.

## Water-quality parameters and methods

**Turbidity, TSM, and TSS** `[N03]`  
Turbidity describes water cloudiness from suspended particles and is reported in FNU or NTU. Total suspended matter and total suspended solids are mass concentrations, commonly g m^-3 or mg L^-1. Their optical signal rises in the red and NIR.

**Chlorophyll-a** `[N03]`  
A photosynthetic pigment used as a phytoplankton proxy, commonly reported in mg m^-3. It has absorption near 443 and 665 nm and a possible red-edge feature.

**CDOM** `[N03]`  
Coloured dissolved organic matter, often river-derived dissolved material. It is commonly reported as `a_g(440)` in m^-1 and absorbs more strongly toward blue wavelengths.

**K_d(490) and Secchi depth** `[N03]`  
`K_d(490)` is a light-attenuation coefficient in m^-1. Secchi depth is a field clarity measure in metres. Higher attenuation usually means lower clarity.

**Transect** `[N03]`  
A line of sampled pixels, here running from the coast toward offshore water to show a plume gradient.

**Relative index** `[N03]`  
An internally consistent pattern or ranking without a local absolute calibration.

**NDWI** `[N03]`  
Normalized Difference Water Index: `(Green - NIR) / (Green + NIR)`. It is used to retain water pixels and exclude land.

**NDCI** `[N03]`  
Normalized Difference Chlorophyll Index: `(Rrs_708 - Rrs_665) / (Rrs_708 + Rrs_665)`. It is used as a chlorophyll screening index, not converted to a site-specific concentration here.

**Nechad turbidity algorithm** `[N03]`  
A single-band form, `T = A rho_w(lambda) / (1 - rho_w(lambda) / C) + B`, with band-specific calibration constants.

**FNU and NTU** `[N04]`  
Formazin and nephelometric turbidity units. They describe how strongly a water sample scatters light. Notebook 04 reports estimated turbidity in FNU.

**Dogliotti switching retrieval** `[N04]`  
A red-to-NIR turbidity method. It uses red light at low to moderate turbidity, NIR when red reflectance saturates, and blends results for red reflectance from 0.05 to 0.07.

**Spectral resampling** `[N04]`  
Averaging narrow Tanager bands into a target sensor bandpass so published broad-band coefficients can be applied.

**SWIR de-glinting** `[N04]`  
A simple residual correction that subtracts SWIR reflectance from visible and NIR bands because water should be nearly black near 1600 nm.

**Formal and total uncertainty** `[N04]`  
Formal uncertainty propagates the supplied reflectance uncertainty through a retrieval. Total uncertainty also includes generic coefficients, atmospheric correction, and missing local validation.

**MDN** `[N05]`  
Mixture Density Network. The model predicts a probability distribution for chlorophyll-a, TSS, and CDOM after Tanager reflectance is resampled to Sentinel-3 OLCI bands.

**GLORIA** `[N05]`  
A global collection of in-situ water spectra and measured water-quality values used to train the MDN models.

**Same-day validation** `[N06]`  
A comparison of Tanager and Sentinel-2C ACOLITE turbidity products acquired about 22 minutes apart. It tests agreement in pattern, not local ground truth.

**Confidence map and error budget** `[N07]`  
A spatial summary that combines sensitivity tests, measurement uncertainty, glint, adjacency, and noise-floor flags. It rates plume pattern more strongly than faint water or absolute constituent estimates.

## Confounds

**Sun glint** `[N02]`  
Specular reflection from the water surface that increases brightness and can mimic turbidity.

**Adjacency effect** `[N07]`  
Atmospheric scattering of bright land light into nearby water pixels, causing nearshore bias.

**Bottom reflectance** `[N02]`  
Seabed light contribution in optically shallow water that can resemble suspended sediment.

**SWIR sand-water test** `[N02]`  
A check that bright red pixels are water rather than exposed sand. Turbid water stays dark in SWIR while exposed land and sand remain bright. It cannot by itself exclude shallow bottom influence.

**Constituent cross-talk** `[N03]`  
Overlapping optical effects of chlorophyll, suspended sediment, and CDOM in Case-2 water.

## Software and formats

**HDF5** `[N02]`  
A hierarchical container for large arrays and metadata, used for Tanager spectral cubes.

**rasterio** `[N01]`  
A Python library for reading and writing georeferenced raster data.

**h5py** `[N02]`  
A Python library for reading HDF5 files.

**ACOLITE** `[N06]`  
Aquatic remote-sensing software used here to process the Sentinel-2C comparison product and retrieve Dogliotti turbidity.

## CoastCheck capstone

**CoastCheck** `[coastcheck_sangatta]`  
The capstone combines published river context, satellite plume evidence, the same-day Sentinel-2 comparison, and ranked shoreline sections to guide repeat monitoring. It does not identify pollution sources or prove ecological impact.

/* ════════════════════════════════════════════════════════════════════
   Tanager-1 Hyperspectral Scene Explorer — application logic
   Requires: Leaflet, Plotly, data/inventory_data.js (TANAGER_INVENTORY)
   ════════════════════════════════════════════════════════════════════ */

/* ── Collection palette (map identity colors, fixed order) ──────────── */
const COLORS = {
    'natural-lands':        '#4f8259',
    'urban':                '#66737d',
    'agriculture':          '#a87924',
    'coastal-water-bodies': '#3f7895',
    'energy-mining':        '#a95b55',
    'ROCX2025':             '#755f91',
    'fire':                 '#b56930',
    'snow-ice':             '#5f8799',
    'GHG-plumes':           '#9a5d78',
};

/* Chart series colors — validated categorical slots on the dark surface */
const SERIES = {
    radiance:    '#356f9b',
    reflectance: '#397454',
};
const SPECTRUM_PRODUCTS = {
    basic_radiance: { label: 'Basic TOA radiance', color: SERIES.radiance, units: 'W/(m² sr µm)', available: false },
    ortho_radiance: { label: 'Ortho TOA radiance', color: SERIES.radiance, units: 'W/(m² sr µm)', available: true },
    basic_sr: { label: 'Basic surface reflectance', color: SERIES.reflectance, units: 'Reflectance (unitless)', available: false },
    ortho_sr: { label: 'Ortho surface reflectance', color: SERIES.reflectance, units: 'Reflectance (unitless)', available: true },
};
const COMPARE_COLORS = ['#356f9b', '#397454', '#a46f1d', '#755f91', '#a95b55'];
const THUMBNAIL_REVISION = 'fullcolor-20260804';
const COASTAL_PALETTES = {
    relative_turbidity: ['#30123b', '#28bbec', '#a4fc3c', '#fb8022', '#7a0403'],
    turbidity_fnu: ['#30123b', '#28bbec', '#a4fc3c', '#fb8022', '#7a0403'],
    relative_cdom: ['#000004', '#51127c', '#b73779', '#fc8961', '#fcfdbf'],
    ndci: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
};

function thumbnailUrl(itemId) {
    return `images/thumbnail/${encodeURIComponent(itemId)}.png?v=${THUMBNAIL_REVISION}`;
}

/* Plot chrome tokens (must match styles.css) */
const PLOT = {
    font:   '"Source Sans 3", "Segoe UI", sans-serif',
    ink:    '#20272d',
    ink2:   '#56636d',
    grid:   '#e2e6e9',
    axis:   '#aeb8bf',
};

/* ── Deduplicated scene list ────────────────────────────────────────────
   The raw inventory has one row per (collection, scene); scenes that belong
   to several collections appear multiple times. Collapse to one record per
   scene with a `collections` array (first-seen collection stays primary). */
const SCENES = (() => {
    const byId = new Map();
    TANAGER_INVENTORY.forEach(row => {
        const existing = byId.get(row.item_id);
        if (existing) {
            if (!existing.collections.includes(row.collection)) {
                existing.collections.push(row.collection);
            }
        } else {
            byId.set(row.item_id, Object.assign({}, row, { collections: [row.collection] }));
        }
    });
    return [...byId.values()];
})();

/* ── State ──────────────────────────────────────────────────────────── */
const activeColls = new Set(Object.keys(COLORS));
let currentOverlay = null;
let currentBorder  = null;
let currentScienceOverlay = null;
let currentOverlayKey = null;
let currentAlignedOverlayUrl = null;
let currentCoastalOverlay = null;
let currentCoastalOverlayUrl = null;
let currentCoastalOverlayKey = null;
let currentGhgOverlay = null;
let currentGhgOverlayKey = null;
let currentGhgLayer = null;
let currentGhgData = null;
const ghgLayerCache = new Map();
let currentCompositeOverlay = null;
let currentCompositeMapUrl = null;
let selectedScene = null;
let selectedBounds = null;
let selectedThumbPath = null;
let activeSampleMarker = null;
let activeSampleArea = null;
let lastSampleLatLng = null;
let lastSpectrumData = null;
const sampleRadius = 0;
let spectrumRequestId = 0;
let roiRequestId = 0;
let coastalRequestId = 0;
let coastalOverlayRequestId = 0;
let ghgRequestId = 0;
let roiLayer = null;
let roiGeometry = null;
let roiDrawing = false;
let roiDrawTool = null;
let roiPlacementShape = null;
let roiShapeKind = null;
let roiDragStart = null;
let roiDragPreview = null;
let roiDragTooltip = null;
let roiDragGeometry = null;
let roiDragging = false;
let suppressMapClickUntil = 0;
let activeMapTool = 'browse';
let activeWorkspaceTab = 'overview';
let compareMode = null; // null, "point", or "area"
let compareSamples = [];
let sceneSort = 'date';
let timelineMonth = null;                 // 'YYYY-MM' or null
let sceneScience = { scenes: {}, overlay_definitions: {} };
let compositeObjectUrl = null;
let lastCompositeBlob = null;
let lastCompositeMeta = null;
let lastCompositeSceneId = null;
const reviewState = JSON.parse(localStorage.getItem('tanagerReviewState') || '{}');

/* Wavelength window for the spectrum plots (Tanager range 376–2500 nm) */
const WL_FLOOR = 376;
const WL_CEIL = 2500;
const WL_MIN_GAP = 30;                     // keep handles at least this far apart
let wlMin = WL_FLOOR;
let wlMax = WL_CEIL;

/* ── Map init ───────────────────────────────────────────────────────── */
const map = L.map('map', {
    center: [10, 20], zoom: 3, minZoom: 2,
    maxBounds: [[-85, -180], [85, 180]],
    maxBoundsViscosity: 1.0,
    zoomControl: true,
    attributionControl: false,
});

const BASEMAPS = {
    imagery: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        maxZoom: 19
    }),
    dark: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
        maxZoom: 16
    }),
};
let activeBasemap = 'imagery';
BASEMAPS.imagery.addTo(map);

const MapCreditsControl = L.Control.extend({
    options: { position: 'bottomright' },
    onAdd() {
        const container = L.DomUtil.create('div', 'map-credits-control');
        container.innerHTML = `
            <button type="button" aria-expanded="false">Map credits</button>
            <div hidden>Imagery: Esri, Maxar, Earthstar Geographics and GIS User Community.<br>Reference: Esri, HERE, Garmin and OpenStreetMap contributors.</div>
        `;
        L.DomEvent.disableClickPropagation(container);
        const button = container.querySelector('button');
        const details = container.querySelector('div');
        button.addEventListener('click', () => {
            const open = details.hidden;
            details.hidden = !open;
            button.setAttribute('aria-expanded', String(open));
        });
        return container;
    }
});
map.addControl(new MapCreditsControl());

const markers = L.layerGroup().addTo(map);
const sampleMarkers = L.layerGroup().addTo(map);
const compareAreaLayers = L.layerGroup().addTo(map);

/* Give Leaflet a nudge after docked panels change the map width */
function queueMapResize() {
    setTimeout(() => map.invalidateSize({ pan: false }), 220);
    setTimeout(() => map.invalidateSize({ pan: false }), 450);
}

/* ── Region overlays ────────────────────────────────────────────────── */
const regions = {
    eq:   L.rectangle([[-15,-180],[15,180]],  {color:'#c98500', weight:1.2, fillOpacity:0.03, dashArray:'4 4', interactive:false}),
    trop: L.rectangle([[-30,-180],[30,180]],  {color:'#8a5a00', weight:1,   fillOpacity:0.015,dashArray:'6 6', interactive:false}),
    sea:  L.rectangle([[-10,95],[25,145]],    {color:'#4cb3e6', weight:1.5, fillOpacity:0.04, dashArray:'4 4', interactive:false}),
};

if (document.getElementById('r-eq').checked)   regions.eq.addTo(map);
if (document.getElementById('r-trop').checked) regions.trop.addTo(map);
if (document.getElementById('r-sea').checked)  regions.sea.addTo(map);

function wireRegion(id, layer) {
    document.getElementById(id).addEventListener('change', e => {
        e.target.checked ? layer.addTo(map) : map.removeLayer(layer);
    });
}
wireRegion('r-eq',   regions.eq);
wireRegion('r-trop', regions.trop);
wireRegion('r-sea',  regions.sea);

/* ── Topbar stats + UTC clock ───────────────────────────────────────── */
document.getElementById('s-total').innerText = SCENES.length;

const utcClock = document.getElementById('utc-clock');
function tickClock() {
    utcClock.textContent = new Date().toISOString().slice(11, 19);
}
tickClock();
setInterval(tickClock, 1000);

/* ── Status bar ─────────────────────────────────────────────────────── */
const stCursor = document.getElementById('st-cursor');

function setLinkState(state='ok', text='Catalogue ready') {
    const status = document.getElementById('st-system');
    const label = document.getElementById('st-system-text');
    if (!status || !label) return;
    status.dataset.state = state;
    label.textContent = String(text)
        .replaceAll(' Â· ', ' · ')
        .replaceAll('Â·', '·')
        .toLowerCase()
        .replace(/(^| · )\w/g, match => match.toUpperCase());
}

function setModeState() {
    document.getElementById('map').classList.toggle('sampling', Boolean(selectedScene));
    document.getElementById('map-badge-text').textContent = selectedScene
        ? 'Sampling armed · click inside the footprint to extract a spectrum'
        : 'Select a scene · then click inside its footprint to sample spectra';
}

function setModeState() {
    const mapElement = document.getElementById('map');
    mapElement.classList.toggle('sampling', Boolean(selectedScene));
    mapElement.classList.toggle('point-tool', activeMapTool === 'point');
    mapElement.classList.toggle('roi-armed', Boolean(roiPlacementShape));
    mapElement.classList.toggle('roi-dragging', roiDragging);
    const messages = {
        browse: selectedScene ? 'Scene selected · use Point, Area or Compose to analyse' : 'Select a scene from the catalogue to begin',
        point: selectedScene ? 'Point sampling · click inside the selected scene' : 'Select a scene before sampling a point',
        area: selectedScene ? 'Area sampling · press inside the scene, drag to size, then release' : 'Select a scene before drawing an area',
        compose: selectedScene ? 'Band composer · choose a recipe in the right dock' : 'Select a scene before composing bands'
    };
    document.getElementById('map-badge-text').textContent = messages[activeMapTool] || messages.browse;
    const labels = {
        browse: 'Browse scenes',
        point: 'Point sampling',
        area: roiPlacementShape ? `${roiPlacementShape[0].toUpperCase()}${roiPlacementShape.slice(1)} area` : 'Area sampling',
        compose: 'Band composer'
    };
    const label = document.getElementById('active-tool-label');
    if (label) label.textContent = labels[activeMapTool] || labels.browse;
    document.getElementById('tool-browse')?.classList.toggle('active', activeMapTool === 'browse');
    document.getElementById('tool-point')?.classList.toggle('active', activeMapTool === 'point');
}

map.on('mousemove', e => {
    stCursor.innerHTML = `LAT ${e.latlng.lat.toFixed(5)}&nbsp; LON ${e.latlng.lng.toFixed(5)}`;
});
map.on('mouseout', () => {
    stCursor.innerHTML = 'LAT &mdash; &nbsp;LON &mdash;';
});

/* ── Basemap toggle ─────────────────────────────────────────────────── */
document.querySelectorAll('#basemap-toggle button').forEach(btn => {
    btn.addEventListener('click', () => {
        const name = btn.dataset.base;
        if (name === activeBasemap) return;
        map.removeLayer(BASEMAPS[activeBasemap]);
        BASEMAPS[name].addTo(map);
        BASEMAPS[name].bringToBack();
        activeBasemap = name;
        document.querySelectorAll('#basemap-toggle button').forEach(b =>
            b.classList.toggle('active', b === btn));
    });
});

/* ── Collapsible columns ────────────────────────────────────────────── */
document.querySelectorAll('.col-collapse, .col-rail').forEach(btn => {
    btn.addEventListener('click', () => {
        document.getElementById(btn.dataset.col).classList.toggle('collapsed');
        queueMapResize();
    });
});

/* ── Collection checkboxes (with proportional count bars) ───────────── */
const collCounts = {};
TANAGER_INVENTORY.forEach(r => collCounts[r.collection] = (collCounts[r.collection]||0) + 1);
const maxCollCount = Math.max(...Object.values(collCounts), 1);

const collListEl = document.getElementById('coll-list');

Object.keys(COLORS).forEach(name => {
    const color = COLORS[name];
    const count = collCounts[name] || 0;
    const pct = Math.max(3, Math.round((count / maxCollCount) * 100));

    const row = document.createElement('div');
    row.className = 'coll-row';
    row.innerHTML = `
        <div class="top">
            <label class="left chk">
                <input type="checkbox" checked data-coll="${name}">
                <span class="chk-box"></span>
                <span class="coll-swatch" style="background:${color}"></span>
                <span class="coll-name">${name}</span>
            </label>
            <span class="coll-count">${count}</span>
        </div>
        <div class="coll-bar"><i style="width:${pct}%; background:${color}"></i></div>
    `;

    const cb = row.querySelector('input');
    cb.addEventListener('change', () => {
        cb.checked ? activeColls.add(name) : activeColls.delete(name);
        refresh();
    });

    collListEl.appendChild(row);
});

/* ── Catalog filters + reset ────────────────────────────────────────── */
const searchInput = document.getElementById('f-search');
const cloudSlider = document.getElementById('f-cloud');
const cloudLabel  = document.getElementById('cloud-val');
const dateFromInput = document.getElementById('f-date-from');
const dateToInput = document.getElementById('f-date-to');
const modeSelect = document.getElementById('f-mode');
const sunSlider = document.getElementById('f-sun');
const sunLabel = document.getElementById('sun-val');
const offNadirSlider = document.getElementById('f-off-nadir');
const offNadirLabel = document.getElementById('off-nadir-val');
const hazeSlider = document.getElementById('f-haze');
const hazeLabel = document.getElementById('haze-val');

searchInput.addEventListener('input', refresh);
searchInput.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    const match = filteredRows()[0];
    if (!match) return;
    showScene(match, COLORS[match.collection] || '#888');
});
cloudSlider.addEventListener('input', () => {
    cloudLabel.innerText = cloudSlider.value + '%';
    refresh();
});
dateFromInput.addEventListener('change', refresh);
dateToInput.addEventListener('change', refresh);
modeSelect.addEventListener('change', refresh);
sunSlider.addEventListener('input', () => {
    sunLabel.innerText = sunSlider.value + '°';
    refresh();
});
offNadirSlider.addEventListener('input', () => {
    offNadirLabel.innerText = offNadirSlider.value + '°';
    refresh();
});
hazeSlider.addEventListener('input', () => {
    hazeLabel.innerText = hazeSlider.value + '%';
    refresh();
});

document.getElementById('btn-reset').addEventListener('click', () => {
    searchInput.value = '';
    cloudSlider.value = 100;
    cloudLabel.innerText = '100%';
    dateFromInput.value = '';
    dateToInput.value = '';
    modeSelect.value = '';
    sunSlider.value = 0;
    sunLabel.innerText = '0°';
    offNadirSlider.value = 31;
    offNadirLabel.innerText = '31°';
    hazeSlider.value = 100;
    hazeLabel.innerText = '100%';
    timelineMonth = null;
    sceneSort = 'date';
    Object.keys(COLORS).forEach(name => activeColls.add(name));
    document.querySelectorAll('#coll-list input[data-coll]').forEach(cb => cb.checked = true);
    refresh();
});

/* ── Keyboard shortcuts ─────────────────────────────────────────────── */
document.addEventListener('keydown', e => {
    const inInput = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if (e.key === '/' && !inInput) {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
    } else if (e.key === 'Escape') {
        if (roiPlacementShape || roiDragging || roiDrawing) {
            cancelPresetRoiDrag(true);
            if (roiDrawTool) roiDrawTool.disable();
            roiDrawing = false;
            activeMapTool = 'browse';
            setModeState();
        } else if (document.querySelector('#export-tray.open, #help-popover.open')) {
            setTransientPanel('export-tray', false);
            setTransientPanel('help-popover', false);
        } else if (selectedScene) {
            closePanel();
        }
    }
});

/* ── DOM handles ────────────────────────────────────────────────────── */
const inspector = document.getElementById('inspector');
const spectrumStatus = document.getElementById('spectrum-status');
const spectrumQa = document.getElementById('spectrum-qa');
const roiSummary = document.getElementById('roi-summary');
const sceneHealth = document.getElementById('scene-health');
const sceneCautions = document.getElementById('scene-cautions');
const overlayControls = document.getElementById('overlay-controls');
const compareList = document.getElementById('compare-list');
const indexGrid = document.getElementById('index-grid');
const bandGrid = document.getElementById('band-grid');
const coverageList = document.getElementById('coverage-list');
const coverageCount = document.getElementById('coverage-count');
const sceneTable = document.getElementById('scene-table');
const filterSummary = document.getElementById('filter-summary');
const resultsCount = document.getElementById('results-count');
const compareButton = document.getElementById('btn-compare');
const compareAreaButton = document.getElementById('btn-compare-area');
const clearSamplesButton = document.getElementById('btn-clear-samples');
const clearRoiButton = document.getElementById('btn-clear-roi');
const undoSampleButton = document.getElementById('btn-undo-sample');
const sampleDiagnostics = document.getElementById('sample-diagnostics');
const coastalAnalysisPanel = document.getElementById('coastal-analysis-panel');
const coastalWorkflowButtons = [...document.querySelectorAll('[data-coastal-workflow]')];
const coastalStatus = document.getElementById('coastal-status');
const coastalResults = document.getElementById('coastal-results');
const coastalMapLegend = document.getElementById('coastal-map-legend');
const ghgMethanePanel = document.getElementById('ghg-methane-panel');
const ghgLayerButtons = [...document.querySelectorAll('[data-ghg-layer]')];
const ghgReferenceButton = document.getElementById('ghg-reference-button');
const ghgStatus = document.getElementById('ghg-status');
const ghgLoadingState = document.getElementById('ghg-loading-state');
const ghgLoadingLabel = document.getElementById('ghg-loading-label');
const ghgResult = document.getElementById('ghg-result');
const ghgResultTitle = document.getElementById('ghg-result-title');
const ghgResultImage = document.getElementById('ghg-result-image');
const ghgResultLegend = document.getElementById('ghg-result-legend');
const ghgResultMetrics = document.getElementById('ghg-result-metrics');
const ghgOverlayButton = document.getElementById('ghg-overlay-button');
const ghgComparison = document.getElementById('ghg-comparison');
const ghgComparisonLabel = document.getElementById('ghg-comparison-label');
const ghgCompareCwmf = document.getElementById('ghg-compare-cwmf');
const ghgCompareReference = document.getElementById('ghg-compare-reference');
const ghgComparisonMetrics = document.getElementById('ghg-comparison-metrics');
const ghgMapLegend = document.getElementById('ghg-map-legend');
const figureDialog = document.getElementById('figure-dialog');
const figureDialogTitle = document.getElementById('figure-dialog-title');
const figureDialogImage = document.getElementById('figure-dialog-image');
const importSampleInput = document.getElementById('import-sample');
const starButton = document.getElementById('btn-star');
const reviewedButton = document.getElementById('btn-reviewed');
const sortDateButton = document.getElementById('sort-date');
const sortCloudButton = document.getElementById('sort-cloud');
const compositeProduct = document.getElementById('composite-product');
const compositePreset = document.getElementById('composite-preset');
const compositeR = document.getElementById('composite-r');
const compositeG = document.getElementById('composite-g');
const compositeB = document.getElementById('composite-b');
const compositeLow = document.getElementById('composite-low');
const compositeHigh = document.getElementById('composite-high');
const compositeButton = document.getElementById('render-composite');
const compositeStatus = document.getElementById('composite-status');
const compositePreview = document.getElementById('composite-preview');
const compositeShell = compositePreview.closest('.composite-shell');
const compositeCaption = document.getElementById('composite-caption');
const compositeBands = document.getElementById('composite-bands');
const compositeRecipeNote = document.getElementById('composite-recipe-note');
const roiShapeButtons = document.querySelectorAll('.roi-shape-btn');
const composerSummary = document.getElementById('composer-summary');
const samplingSummary = document.getElementById('sampling-summary');
const indicesSummary = document.getElementById('indices-summary');
const bandsSummary = document.getElementById('bands-summary');
const inspectorSections = document.querySelectorAll('.ins-disclosure');
const sampleState = document.getElementById('sample-state');
const drawerContext = document.getElementById('drawer-context');
const spectrumProduct = document.getElementById('spectrum-product');
const sceneRequiredDialog = document.getElementById('scene-required-dialog');

function openInspectorSection(id) {
    const inspectorMap = {
        'section-composer': 'compose',
        'section-overlays': 'overview',
        'section-coverage': 'overview'
    };
    const drawerMap = {
        'section-sampling': 'spectrum',
        'section-indices': 'indices',
        'section-bands': 'indices'
    };
    if (inspectorMap[id]) activateInspectorTab(inspectorMap[id]);
    if (drawerMap[id]) activateDrawerTab(drawerMap[id]);
}

inspectorSections.forEach(section => {
    section.addEventListener('toggle', () => {
        if (!section.open) return;
        inspectorSections.forEach(other => {
            if (other !== section) other.open = false;
        });
        queueMapResize();
    });
});

/* Workspace docks, task tabs and transient trays */
function activateCatalogTab(name) {
    document.querySelectorAll('[data-catalog-tab]').forEach(button => {
        const active = button.dataset.catalogTab === name;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('[data-catalog-panel]').forEach(panel => {
        panel.classList.toggle('active', panel.dataset.catalogPanel === name);
    });
}

function activateInspectorTab(name) {
    document.querySelectorAll('[data-ins-tab]').forEach(button => {
        const active = button.dataset.insTab === name;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('[data-ins-panel]').forEach(panel => {
        panel.classList.toggle('active', panel.dataset.insPanel === name);
    });
}

function syncDrawerToggle() {
    // Analysis now uses native disclosure panels instead of a bottom drawer.
}

function setDrawerOpen(open) {
    const panel = document.querySelector('[data-analysis-panel="spectrum"]');
    if (panel) panel.open = open;
}

function activateDrawerTab(name, expand=true) {
    const panel = document.querySelector(`[data-analysis-panel="${name}"]`);
    if (panel && expand) panel.open = true;
    setTimeout(() => {
        if (window.Plotly) {
            const plot = document.getElementById('spectrum-plot');
            if (plot?._fullLayout) Plotly.Plots.resize(plot);
        }
    }, 240);
}

function setWorkspaceTab(name, { promptForScene=true }={}) {
    if (name === 'analysis' && !selectedScene) {
        if (promptForScene && sceneRequiredDialog && !sceneRequiredDialog.open) {
            sceneRequiredDialog.showModal();
        }
        return false;
    }
    activeWorkspaceTab = name;
    document.body.dataset.workspaceTab = name;
    document.querySelectorAll('.workspace-tabs button').forEach(button => {
        const active = button.id === `tab-${name}`;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', String(active));
    });
    if (name === 'analysis') inspector.classList.add('open');
    queueMapResize();
    return true;
}

function setCatalogOpen(open) {
    const dock = document.getElementById('catalog-dock');
    dock.classList.toggle('collapsed', !open);
    dock.classList.toggle('scenes-collapsed', !open);
    document.getElementById('catalog-rail').style.display = open ? 'none' : 'grid';
    const toggle = document.getElementById('catalog-close');
    toggle.title = open ? 'Collapse scenes' : 'Expand scenes';
    toggle.setAttribute('aria-label', toggle.title);
    toggle.setAttribute('aria-expanded', String(open));
    queueMapResize();
}

function setScenesOpen(open) {
    const dock = document.getElementById('catalog-dock');
    const filtersOpen = !document.getElementById('filter-subdock').classList.contains('collapsed');
    dock.classList.toggle('scenes-collapsed', !open);
    dock.classList.toggle('collapsed', !open && !filtersOpen);
    document.getElementById('catalog-rail').style.display = !open && !filtersOpen ? 'grid' : 'none';
    const toggle = document.getElementById('catalog-close');
    toggle.title = open ? 'Collapse scenes' : 'Expand scenes';
    toggle.setAttribute('aria-label', toggle.title);
    toggle.setAttribute('aria-expanded', String(open));
    queueMapResize();
}

function setTransientPanel(id, open) {
    const panel = document.getElementById(id);
    if (!panel) return;
    panel.classList.toggle('open', open);
    panel.setAttribute('aria-hidden', String(!open));
}

document.querySelectorAll('[data-catalog-tab]').forEach(button =>
    button.addEventListener('click', () => activateCatalogTab(button.dataset.catalogTab))
);
document.querySelectorAll('[data-ins-tab]').forEach(button =>
    button.addEventListener('click', () => activateInspectorTab(button.dataset.insTab))
);
document.getElementById('tab-overview').addEventListener('click', () => setWorkspaceTab('overview'));
document.getElementById('tab-analysis').addEventListener('click', () => setWorkspaceTab('analysis'));
document.getElementById('dialog-overview').addEventListener('click', () => {
    sceneRequiredDialog.close();
    setWorkspaceTab('overview', { promptForScene: false });
});
document.getElementById('catalog-close').addEventListener('click', () => {
    const dock = document.getElementById('catalog-dock');
    setScenesOpen(dock.classList.contains('scenes-collapsed'));
});
document.getElementById('catalog-rail').addEventListener('click', () => setScenesOpen(true));
function setFilterOpen(open) {
    const dock = document.getElementById('catalog-dock');
    document.getElementById('filter-subdock').classList.toggle('collapsed', !open);
    dock.classList.toggle('filters-collapsed', !open);
    if (!open && dock.classList.contains('scenes-collapsed')) {
        dock.classList.add('collapsed');
        document.getElementById('catalog-rail').style.display = 'grid';
    } else if (open) {
        dock.classList.remove('collapsed');
        document.getElementById('catalog-rail').style.display = 'none';
    }
    queueMapResize();
}
document.getElementById('filter-collapse').addEventListener('click', () => setFilterOpen(false));
document.getElementById('filter-rail').addEventListener('click', () => setFilterOpen(true));
document.getElementById('tool-export')?.addEventListener('click', () => {
    inspector.classList.add('open');
    activateInspectorTab('export');
    queueMapResize();
});
document.getElementById('tool-browse').addEventListener('click', () => {
    cancelPresetRoiDrag(true);
    compareMode = null;
    syncCompareButtons();
    activeMapTool = 'browse';
    setModeState();
});
document.getElementById('tool-point').addEventListener('click', () => {
    cancelPresetRoiDrag(true);
    compareMode = null;
    syncCompareButtons();
    activeMapTool = 'point';
    activateDrawerTab('spectrum');
    setModeState();
});
spectrumProduct.addEventListener('change', () => {
    redrawSpectra();
    if (lastSpectrumData) renderBandReadout(lastSpectrumData);
});

function renderSortButtons() {
    sortDateButton.classList.toggle('active', sceneSort === 'date');
    sortCloudButton.classList.toggle('active', sceneSort === 'cloud');
}

function renderSummary(rows) {
    const maxCloud = Number(cloudSlider.value);
    const minSun = Number(sunSlider.value);
    const maxOffNadir = Number(offNadirSlider.value);
    const maxHaze = Number(hazeSlider.value);
    const query = searchInput.value.trim();
    const filters = [];
    if (query) filters.push(`q="${query}"`);
    if (maxCloud < 100) filters.push(`cloud≤${maxCloud}%`);
    if (dateFromInput.value) filters.push(`from ${dateFromInput.value}`);
    if (dateToInput.value) filters.push(`to ${dateToInput.value}`);
    if (modeSelect.value) filters.push(modeSelect.options[modeSelect.selectedIndex].text);
    if (minSun > 0) filters.push(`sun≥${minSun}°`);
    if (maxOffNadir < 31) filters.push(`off-nadir≤${maxOffNadir}°`);
    if (maxHaze < 100) filters.push(`haze≤${maxHaze}%`);
    if (activeColls.size < Object.keys(COLORS).length) {
        filters.push(`${activeColls.size}/${Object.keys(COLORS).length} collections`);
    }
    if (timelineMonth) filters.push(`month=${timelineMonth}`);
    filterSummary.textContent = filters.length ? filters.join(' · ') : 'no constraints — full catalog';
    resultsCount.textContent = rows.length;
    document.getElementById('s-filt').innerText = rows.length;
}

sortDateButton.addEventListener('click', () => { sceneSort = 'date'; refresh(); });
sortCloudButton.addEventListener('click', () => { sceneSort = 'cloud'; refresh(); });

function syncCompareButtons() {
    compareButton.classList.toggle('active', compareMode === 'point');
    compareAreaButton.classList.toggle('active', compareMode === 'area');
}

function resetComparedSamples() {
    compareSamples = [];
    sampleMarkers.clearLayers();
    compareAreaLayers.clearLayers();
    renderCompareList();
    updateUndoSampleButton();
}

function setCompareMode(mode) {
    const nextMode = compareMode === mode ? null : mode;
    cancelPresetRoiDrag(true);
    compareMode = nextMode;
    resetComparedSamples();
    syncCompareButtons();
    activateDrawerTab('spectrum');
    if (nextMode === 'point') {
        activeMapTool = 'point';
        setSpectrumStatus('Compare Points: select two points inside the scene.');
    } else if (nextMode === 'area') {
        activeMapTool = 'area';
        setSpectrumStatus('Compare Areas: choose a shape, then select two areas.');
    } else {
        activeMapTool = 'browse';
        setSpectrumStatus('Comparison off.');
    }
    setModeState();
}

compareButton.addEventListener('click', () => setCompareMode('point'));
compareAreaButton.addEventListener('click', () => setCompareMode('area'));

clearSamplesButton.addEventListener('click', () => {
    compareSamples = [];
    lastSpectrumData = null;
    sampleMarkers.clearLayers();
    compareAreaLayers.clearLayers();
    clearActiveSampleMarker();
    clearRoiAnalysis(false);
    renderCompareList();
    clearSpectrum();
});

undoSampleButton.addEventListener('click', undoLatestPoint);
clearRoiButton.addEventListener('click', () => {
    clearRoiAnalysis(true);
    if (compareSamples.some(sample => sample.data?.sample_type === 'roi')) resetComparedSamples();
});

starButton?.addEventListener('click', () => toggleReviewFlag('starred'));
reviewedButton?.addEventListener('click', () => toggleReviewFlag('reviewed'));
document.getElementById('export-json').addEventListener('click', exportJsonReport);
document.getElementById('export-csv').addEventListener('click', exportSpectraCsv);
document.getElementById('export-png').addEventListener('click', exportPlotPng);
document.getElementById('export-evidence').addEventListener('click', exportEvidencePackage);
document.getElementById('copy-link').addEventListener('click', copySceneLink);
importSampleInput.addEventListener('change', event => importSampleDocument(event.target.files?.[0]));
compositePreset.addEventListener('change', applyCompositePreset);
[compositeR, compositeG, compositeB].forEach(input => {
    input.addEventListener('input', () => {
        compositePreset.value = 'custom';
        applyCompositePreset();
    });
});
compositeButton.addEventListener('click', renderComposite);
document.getElementById('download-composite').addEventListener('click', downloadCompositePng);
document.getElementById('clear-composite').addEventListener('click', () => {
    clearCompositePreview();
});

roiShapeButtons.forEach(button => {
    button.addEventListener('click', () => {
        if (button.dataset.roiShape === 'custom') armRoiDraw();
        else armPresetRoi(button.dataset.roiShape);
    });
});

/* ── Science manifest ───────────────────────────────────────────────── */
async function loadScienceManifest() {
    try {
        const response = await fetch('data/scene_science_manifest.json', { cache: 'no-store' });
        if (!response.ok) {
            setLinkState('ok', 'CATALOG READY · NO SCIENCE MANIFEST');
            return;
        }
        const data = await response.json();
        sceneScience = data && data.scenes ? data : sceneScience;
        setLinkState('ok', `CATALOG READY · ${Object.keys(sceneScience.scenes || {}).length} SCENES WITH KERCHUNK REFS`);
    } catch {
        sceneScience = { scenes: {}, overlay_definitions: {} };
        setLinkState('error', 'SCIENCE MANIFEST FAILED TO LOAD');
    }
}

function sceneScienceFor(row) {
    return sceneScience.scenes?.[row.item_id] || null;
}

/* ── Review flags (localStorage) ────────────────────────────────────── */
function reviewFor(sceneId) {
    if (!reviewState[sceneId]) reviewState[sceneId] = { starred: false, reviewed: false };
    return reviewState[sceneId];
}

function saveReviewState() {
    localStorage.setItem('tanagerReviewState', JSON.stringify(reviewState));
}

function toggleReviewFlag(flag) {
    if (!selectedScene) return;
    const state = reviewFor(selectedScene.item_id);
    state[flag] = !state[flag];
    saveReviewState();
    renderReviewButtons();
    renderSceneTable(filteredRows());
}

function renderReviewButtons() {
    if (!selectedScene || !starButton || !reviewedButton) return;
    const state = reviewFor(selectedScene.item_id);
    starButton.classList.toggle('active', state.starred);
    reviewedButton.classList.toggle('active', state.reviewed);
    starButton.textContent = state.starred ? 'Starred' : 'Star';
    reviewedButton.textContent = state.reviewed ? 'Reviewed' : 'Mark reviewed';
}

/* ── Scene diagnostics ──────────────────────────────────────────────── */
function formatValue(value, suffix='') {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
    return `${Number(value).toFixed(Number(value) % 1 ? 1 : 0)}${suffix}`;
}

function productPresent(row, key) {
    const value = row[`asset_${key}`];
    return Boolean(value && value === value);
}

function cautionBadges(row) {
    const science = sceneScienceFor(row);
    if (science?.cautions?.length) return science.cautions;
    const cloud = Number(row.cloud_percent || 0);
    const haze = Number(row.light_haze_percent || 0);
    const sun = Number(row.sun_elevation || 0);
    const badges = [];
    if (cloud >= 50) badges.push({ label: 'Cloudy scene', severity: 'high' });
    else if (cloud >= 20) badges.push({ label: 'Partial cloud', severity: 'medium' });
    if (haze >= 50) badges.push({ label: 'High light haze', severity: 'medium' });
    if (sun > 0 && sun < 20) badges.push({ label: 'Low sun elevation', severity: 'medium' });
    if (!productPresent(row, 'ortho_sr_hdf5')) badges.push({ label: 'Surface reflectance unavailable', severity: 'medium' });
    return badges;
}

function renderSceneHealth(row) {
    const cells = [
        ['Cloud cover', formatValue(row.cloud_percent, '%')],
        ['Light haze', formatValue(row.light_haze_percent, '%')],
        ['Quality', row.quality_category || '-'],
        ['Sun elevation', formatValue(row.sun_elevation, '°')],
        ['Off-nadir', formatValue(row.off_nadir, '°')],
        ['GSD', formatValue(row.gsd, ' m')]
    ];
    sceneHealth.innerHTML = cells.map(([key, value]) => `
        <div class="health-cell">
            <div class="health-k">${escapeHtml(key)}</div>
            <div class="health-v" title="${escapeHtml(value)}">${escapeHtml(value)}</div>
        </div>
    `).join('');
    const badges = cautionBadges(row);
    sceneCautions.innerHTML = badges.map(b =>
        `<span class="qa-chip ${b.severity === 'high' ? 'bad' : 'warn'}">${escapeHtml(b.label)}</span>`
    ).join('');
}

/* ── Science overlays ───────────────────────────────────────────────── */
function loadRasterImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`Could not load raster ${src}`));
        image.src = src;
    });
}

function canvasObjectUrl(canvas) {
    return new Promise((resolve, reject) => {
        canvas.toBlob(blob => blob ? resolve(URL.createObjectURL(blob)) : reject(new Error('Could not encode aligned raster.')), 'image/png');
    });
}

function estimateQaTranslation(source, thumb, cloudPercent=0) {
    if (Number(cloudPercent) <= 0) return { x: 0, y: 0 };

    const analysisSize = 256;
    const sourceCanvas = document.createElement('canvas');
    const thumbCanvas = document.createElement('canvas');
    sourceCanvas.width = thumbCanvas.width = analysisSize;
    sourceCanvas.height = thumbCanvas.height = analysisSize;
    const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
    const thumbContext = thumbCanvas.getContext('2d', { willReadFrequently: true });
    sourceContext.imageSmoothingEnabled = false;
    sourceContext.drawImage(source, 0, 0, analysisSize, analysisSize);
    thumbContext.drawImage(thumb, 0, 0, analysisSize, analysisSize);

    const sourcePixels = sourceContext.getImageData(0, 0, analysisSize, analysisSize).data;
    const thumbPixels = thumbContext.getImageData(0, 0, analysisSize, analysisSize).data;
    const maskPixels = [];
    const cloudLikelihood = new Float32Array(analysisSize * analysisSize);

    for (let index = 0; index < analysisSize * analysisSize; index += 1) {
        const offset = index * 4;
        const sr = sourcePixels[offset];
        const sg = sourcePixels[offset + 1];
        const sb = sourcePixels[offset + 2];
        if (
            sourcePixels[offset + 3] > 20
            && sr > 150
            && sr > sg + 30
            && sr > sb + 20
        ) {
            maskPixels.push([index % analysisSize, Math.floor(index / analysisSize)]);
        }

        if (thumbPixels[offset + 3] <= 40) continue;
        const red = thumbPixels[offset] / 255;
        const green = thumbPixels[offset + 1] / 255;
        const blue = thumbPixels[offset + 2] / 255;
        const maximum = Math.max(red, green, blue);
        const minimum = Math.min(red, green, blue);
        const saturation = (maximum - minimum) / (maximum + 1e-6);
        const brightness = Math.max(0, Math.min(1, (maximum - 0.42) / 0.58));
        const neutrality = Math.max(0, Math.min(1, (0.55 - saturation) / 0.55));
        cloudLikelihood[index] = brightness * neutrality;
    }

    // Very small masks do not carry enough image evidence for safe registration.
    if (maskPixels.length < 160) return { x: 0, y: 0 };
    const stride = Math.max(1, Math.ceil(maskPixels.length / 5000));
    const sampledMask = maskPixels.filter((_, index) => index % stride === 0);

    const scoreAt = (dx, dy) => {
        let score = 0;
        let count = 0;
        for (const [x, y] of sampledMask) {
            const targetX = x + dx;
            const targetY = y + dy;
            if (targetX < 0 || targetX >= analysisSize || targetY < 0 || targetY >= analysisSize) continue;
            score += cloudLikelihood[targetY * analysisSize + targetX];
            count += 1;
        }
        if (count < sampledMask.length * 0.88) return -Infinity;
        return score / Math.max(1, count);
    };

    const baseline = scoreAt(0, 0);
    let best = { x: 0, y: 0, rawScore: baseline, adjustedScore: baseline };
    for (let dy = -5; dy <= 5; dy += 1) {
        for (let dx = -5; dx <= 5; dx += 1) {
            const rawScore = scoreAt(dx, dy);
            const adjustedScore = rawScore - Math.hypot(dx, dy) * 0.002;
            if (adjustedScore > best.adjustedScore) {
                best = { x: dx, y: dy, rawScore, adjustedScore };
            }
        }
    }

    return best.rawScore >= 0.12 && best.rawScore - baseline >= 0.035
        ? { x: best.x, y: best.y }
        : { x: 0, y: 0 };
}

async function buildAlignedQaUrl(sourcePath, thumbPath, cloudPercent=0) {
    const [source, thumb] = await Promise.all([loadRasterImage(sourcePath), loadRasterImage(thumbPath)]);
    const canvas = document.createElement('canvas');
    canvas.width = thumb.naturalWidth || thumb.width;
    canvas.height = thumb.naturalHeight || thumb.height;
    const context = canvas.getContext('2d');
    const translation = estimateQaTranslation(source, thumb, cloudPercent);
    const offsetX = translation.x * canvas.width / 256;
    const offsetY = translation.y * canvas.height / 256;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, offsetX, offsetY, canvas.width, canvas.height);

    // Keep mask pixels within the scene's visible raster footprint.
    context.globalCompositeOperation = 'destination-in';
    context.drawImage(thumb, 0, 0, canvas.width, canvas.height);
    context.globalCompositeOperation = 'source-over';
    return canvasObjectUrl(canvas);
}

function rasterFootprintTransform(template, alphaThreshold=12) {
    const canvas = document.createElement('canvas');
    canvas.width = template.naturalWidth || template.width;
    canvas.height = template.naturalHeight || template.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(template, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    let meanX = 0;
    let meanY = 0;
    for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
            if (pixels[(y * canvas.width + x) * 4 + 3] <= alphaThreshold) continue;
            count += 1;
            meanX += x;
            meanY += y;
        }
    }
    if (!count) return { centerX: canvas.width / 2, centerY: canvas.height / 2, angle: 0, width: canvas.width, height: canvas.height };
    meanX /= count;
    meanY /= count;
    let xx = 0;
    let yy = 0;
    let xy = 0;
    for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
            if (pixels[(y * canvas.width + x) * 4 + 3] <= alphaThreshold) continue;
            const dx = x - meanX;
            const dy = y - meanY;
            xx += dx * dx;
            yy += dy * dy;
            xy += dx * dy;
        }
    }
    const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
    const ux = Math.cos(angle);
    const uy = Math.sin(angle);
    const vx = -uy;
    const vy = ux;
    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
            if (pixels[(y * canvas.width + x) * 4 + 3] <= alphaThreshold) continue;
            const dx = x - meanX;
            const dy = y - meanY;
            const u = dx * ux + dy * uy;
            const v = dx * vx + dy * vy;
            minU = Math.min(minU, u);
            maxU = Math.max(maxU, u);
            minV = Math.min(minV, v);
            maxV = Math.max(maxV, v);
        }
    }
    const centerU = (minU + maxU) / 2;
    const centerV = (minV + maxV) / 2;
    return {
        centerX: meanX + centerU * ux + centerV * vx,
        centerY: meanY + centerU * uy + centerV * vy,
        angle,
        width: maxU - minU + 1,
        height: maxV - minV + 1
    };
}

async function buildAlignedCompositeUrl(sourceUrl, templatePath) {
    const [source, template] = await Promise.all([loadRasterImage(sourceUrl), loadRasterImage(templatePath)]);
    const transform = rasterFootprintTransform(template);
    const canvas = document.createElement('canvas');
    canvas.width = template.naturalWidth || template.width;
    canvas.height = template.naturalHeight || template.height;
    const context = canvas.getContext('2d');
    let drawWidth = transform.width;
    let drawHeight = transform.height;
    let angle = transform.angle;
    const sourceLandscape = source.width >= source.height;
    const footprintLandscape = transform.width >= transform.height;
    if (sourceLandscape !== footprintLandscape) {
        [drawWidth, drawHeight] = [drawHeight, drawWidth];
        angle += Math.PI / 2;
    }
    context.save();
    context.translate(transform.centerX, transform.centerY);
    context.rotate(angle);
    context.drawImage(source, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
    context.restore();
    context.globalCompositeOperation = 'destination-in';
    context.drawImage(template, 0, 0, canvas.width, canvas.height);
    context.globalCompositeOperation = 'source-over';
    return canvasObjectUrl(canvas);
}

function overlayLabel(overlay) {
    return sceneScience.overlay_definitions?.[overlay.key]?.label || overlay.key.replaceAll('_', ' ');
}

function renderOverlayControls(row, bounds) {
    if (!row) {
        overlayControls.innerHTML = '<div class="empty-layer-note">Scene layers appear after selection.</div>';
        return;
    }
    const science = sceneScienceFor(row);
    const overlays = (science?.overlays || []).filter(overlay => !['ndvi', 'ndwi'].includes(overlay.key));
    if (currentOverlayKey && !overlays.some(overlay => overlay.key === currentOverlayKey)) currentOverlayKey = null;
    const qaOverlay = overlays.find(overlay => overlay.key === 'qa_mask') || overlays[0] || null;
    const qaActive = Boolean(qaOverlay && currentOverlayKey === qaOverlay.key && currentScienceOverlay);
    const sceneVisible = Boolean(currentOverlay);
    overlayControls.innerHTML = `
        <div class="layer-row static">
            <button class="layer-visibility ${sceneVisible ? '' : 'off'}" type="button" data-scene-visibility aria-pressed="${sceneVisible}" title="Toggle selected RGB scene"><svg><use href="#i-eye"></use></svg></button>
            <div class="layer-main"><span class="layer-name">Scene</span><span class="layer-meta">${sceneVisible ? 'Visible' : 'Hidden'}</span></div>
        </div>
        ${qaOverlay ? `
        <div class="layer-row">
            <button class="layer-visibility ${qaActive ? '' : 'off'}" type="button" data-overlay="${escapeHtml(qaOverlay.key)}" aria-pressed="${qaActive}" title="Toggle cloud, cirrus and nodata mask"><svg><use href="#i-eye"></use></svg></button>
            <div class="layer-main"><span class="layer-name">Cloud / cirrus / nodata</span><span class="layer-meta">${qaActive ? 'Visible' : 'Hidden'}</span></div>
        </div>` : ''}
    `;
    overlayControls.querySelector('[data-scene-visibility]')?.addEventListener('click', () => {
        if (currentOverlay) {
            map.removeLayer(currentOverlay);
            currentOverlay = null;
        } else if (selectedThumbPath && bounds) {
            currentOverlay = L.imageOverlay(selectedThumbPath, bounds, {
                opacity: 0.88,
                interactive: false,
                className: 'scene-overlay-img'
            }).addTo(map);
        }
        renderOverlayControls(row, bounds);
    });
    overlayControls.querySelector('[data-overlay]')?.addEventListener('click', async () => {
        currentOverlayKey = qaActive ? null : qaOverlay.key;
        await setScienceOverlay(overlays, bounds);
        renderOverlayControls(row, bounds);
    });
}

let scienceOverlayRequestId = 0;
async function setScienceOverlay(overlays, bounds) {
    const requestId = ++scienceOverlayRequestId;
    if (currentScienceOverlay) {
        map.removeLayer(currentScienceOverlay);
        currentScienceOverlay = null;
    }
    if (currentAlignedOverlayUrl) {
        URL.revokeObjectURL(currentAlignedOverlayUrl);
        currentAlignedOverlayUrl = null;
    }
    document.getElementById('qa-map-key').hidden = true;
    if (!currentOverlayKey) return;
    const overlay = overlays.find(item => item.key === currentOverlayKey);
    if (!overlay?.path) return;
    try {
        const alignedUrl = selectedThumbPath
            ? await buildAlignedQaUrl(overlay.path, selectedThumbPath, Number(selectedScene?.cloud_percent || 0))
            : overlay.path;
        if (requestId !== scienceOverlayRequestId) {
            if (alignedUrl.startsWith('blob:')) URL.revokeObjectURL(alignedUrl);
            return;
        }
        currentAlignedOverlayUrl = alignedUrl.startsWith('blob:') ? alignedUrl : null;
        currentScienceOverlay = L.imageOverlay(alignedUrl, bounds, {
            opacity: 0.72,
            interactive: false,
            className: 'science-overlay-img'
        }).addTo(map);
        currentScienceOverlay.bringToFront();
        document.getElementById('qa-map-key').hidden = false;
    } catch (error) {
        currentOverlayKey = null;
        setLinkState('error', 'QA alignment failed');
        console.error(error);
    }
}

/* ── Arbitrary-band image composer ──────────────────────────────────── */
const COMPOSITE_PRESETS = {
    'true-color': {
        bands: [665, 560, 490],
        note: 'Natural visible-light rendering for general scene context.'
    },
    vegetation: {
        bands: [865, 665, 560],
        note: 'Near-infrared appears red, making vigorous vegetation stand out.'
    },
    agriculture: {
        bands: [1610, 865, 490],
        note: 'Separates crop condition, canopy moisture and exposed soil.'
    },
    'burn-scar': {
        bands: [2200, 1610, 865],
        note: 'Highlights dry or burned surfaces and can expose active-fire contrast.'
    },
    urban: {
        bands: [2200, 1610, 665],
        note: 'Separates built-up and bare surfaces from vegetation and water.'
    },
    geology: {
        bands: [2200, 1610, 560],
        note: 'Uses SWIR contrast to reveal mineral, rock and soil differences.'
    },
    'snow-water': {
        bands: [1610, 865, 560],
        note: 'Helps distinguish snow and ice from cloud, vegetation and open water.'
    },
    ndvi: {
        index: true,
        note: 'Calculated (NIR − red) / (NIR + red); greener colours indicate stronger vegetation response.'
    },
    ndwi: {
        index: true,
        note: 'Calculated (green − NIR) / (green + NIR); blue colours emphasize open water.'
    },
    mndwi: {
        index: true,
        note: 'Calculated (green − SWIR1) / (green + SWIR1); improves water contrast in built-up land.'
    },
    nbr: {
        index: true,
        note: 'Calculated (NIR − SWIR2) / (NIR + SWIR2); useful for burned-area and vegetation contrast.'
    }
};

function applyCompositePreset() {
    const preset = COMPOSITE_PRESETS[compositePreset.value];
    const isIndex = Boolean(preset?.index);
    compositeBands.classList.toggle('index-mode', isIndex);
    compositeProduct.disabled = isIndex;
    if (isIndex) compositeProduct.value = 'ortho_sr';
    if (preset?.bands) {
        [compositeR.value, compositeG.value, compositeB.value] = preset.bands;
    }
    compositeRecipeNote.textContent = preset?.note || 'Enter any three wavelengths and assign them to the display channels.';
    compositeButton.textContent = isIndex ? 'Render calculated index' : 'Render selected bands';
    if (composerSummary) {
        const label = compositePreset.options[compositePreset.selectedIndex]?.text || 'Custom bands';
        composerSummary.textContent = label.split('·')[0].trim();
    }
}

function clearCompositePreview() {
    if (currentCompositeOverlay) {
        map.removeLayer(currentCompositeOverlay);
        currentCompositeOverlay = null;
    }
    if (currentCompositeMapUrl) {
        URL.revokeObjectURL(currentCompositeMapUrl);
        currentCompositeMapUrl = null;
    }
    if (compositeObjectUrl) {
        URL.revokeObjectURL(compositeObjectUrl);
        compositeObjectUrl = null;
    }
    lastCompositeBlob = null;
    lastCompositeMeta = null;
    lastCompositeSceneId = null;
    compositePreview.removeAttribute('src');
    compositeShell.classList.remove('ready');
    compositeButton.closest('.composer-form')?.classList.remove('loading');
    compositeCaption.textContent = '';
    compositeStatus.textContent = '';
    compositeStatus.className = 'spectrum-status';
    compositeButton.disabled = false;
    document.getElementById('download-composite').disabled = true;
    document.getElementById('clear-composite').disabled = true;
    document.getElementById('composite-map-key').hidden = true;
    if (composerSummary) composerSummary.textContent = 'True colour';
}

function downloadCompositePng() {
    if (!lastCompositeBlob || !compositeObjectUrl || !lastCompositeSceneId) return;
    const link = document.createElement('a');
    link.href = compositeObjectUrl;
    link.download = `${lastCompositeSceneId}_${lastCompositeMeta?.recipe || compositePreset.value || 'composite'}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

async function renderComposite() {
    if (!selectedScene) {
        compositeStatus.textContent = 'Select a scene before rendering bands.';
        compositeStatus.className = 'spectrum-status error';
        return;
    }
    const values = [compositeR, compositeG, compositeB].map(input => Number(input.value));
    const low = Number(compositeLow.value);
    const high = Number(compositeHigh.value);
    const preset = COMPOSITE_PRESETS[compositePreset.value];
    const isIndex = Boolean(preset?.index);
    if (!isIndex && values.some(value => !Number.isFinite(value) || value < 376 || value > 2500)) {
        compositeStatus.textContent = 'R, G and B must each be between 376 and 2500 nm.';
        compositeStatus.className = 'spectrum-status error';
        return;
    }
    if (!Number.isFinite(low) || !Number.isFinite(high) || low < 0 || high > 100 || low >= high) {
        compositeStatus.textContent = 'Stretch must satisfy 0 ≤ low < high ≤ 100.';
        compositeStatus.className = 'spectrum-status error';
        return;
    }
    const params = new URLSearchParams({
        scene_id: selectedScene.item_id,
        product: compositeProduct.value,
        r: values[0],
        g: values[1],
        b: values[2],
        low,
        high,
        max_size: 320,
        recipe: isIndex ? compositePreset.value : 'rgb'
    });
    compositeButton.disabled = true;
    compositeButton.closest('.composer-form')?.classList.add('loading');
    if (composerSummary) composerSummary.textContent = 'Rendering…';
    compositeStatus.textContent = isIndex
        ? 'Reading index bands and building preview…'
        : 'Reading three bands and building preview…';
    compositeStatus.className = 'spectrum-status';
    try {
        const response = await fetch(`/api/composite?${params.toString()}`);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || `Composite request failed (${response.status})`);
        }
        const blob = await response.blob();
        const metadata = JSON.parse(response.headers.get('X-Tanager-Composite') || '{}');
        if (compositeObjectUrl) URL.revokeObjectURL(compositeObjectUrl);
        compositeObjectUrl = URL.createObjectURL(blob);
        lastCompositeBlob = blob;
        lastCompositeMeta = metadata;
        lastCompositeSceneId = selectedScene.item_id;
        compositePreview.src = compositeObjectUrl;
        compositeShell.classList.add('ready');
        const matched = metadata.matched_wavelengths_nm || values;
        const strideText = metadata.stride > 1 ? ` · preview stride ${metadata.stride}` : '';
        const matchText = metadata.render_mode === 'index'
            ? `${metadata.recipe_label || compositePreset.value.toUpperCase()} · matched ${matched.map(value => `${Number(value).toFixed(1)} nm`).join(' / ')}`
            : `Matched R ${Number(matched[0]).toFixed(1)}, G ${Number(matched[1]).toFixed(1)}, B ${Number(matched[2]).toFixed(1)} nm`;
        compositeCaption.textContent =
            `${matchText} · ${low}–${high} percentile stretch${strideText}.`;
        compositeStatus.textContent = '';
        compositeStatus.className = 'spectrum-status';
        document.getElementById('download-composite').disabled = false;
        document.getElementById('clear-composite').disabled = false;
        if (composerSummary) {
            composerSummary.textContent = metadata.recipe_label || 'Custom RGB render';
        }
    } catch (error) {
        compositeShell.classList.remove('ready');
        compositeCaption.textContent = '';
        compositeStatus.textContent = error.message;
        compositeStatus.className = 'spectrum-status error';
        if (composerSummary) composerSummary.textContent = 'Rendering issue';
    } finally {
        compositeButton.closest('.composer-form')?.classList.remove('loading');
        compositeButton.disabled = false;
    }
}

applyCompositePreset();

/* ── Sample markers ─────────────────────────────────────────────────── */
function renderCompareList() {
    compareList.innerHTML = compareSamples.map((sample, index) => `
        <div class="compare-row">
            <span class="compare-swatch" style="background:${sample.color}"></span>
            <span>${escapeHtml(sample.label)} ${sample.data?.sample_type === 'roi'
                ? `${Number(sample.pixelCount || 0).toLocaleString()} pixels`
                : `${sample.lat.toFixed(5)}, ${sample.lon.toFixed(5)}`}</span>
            <span>${index + 1}</span>
        </div>
    `).join('');
}

function sampleIcon(label, color, isArea, pending=false) {
    return L.divIcon({
        html: `<div class="sample-marker ${isArea ? 'area' : ''} ${pending ? 'pending' : ''}" style="border-color:${color};">${escapeHtml(label)}</div>`,
        className: '',
        iconSize: isArea ? [32, 32] : [22, 22],
        iconAnchor: isArea ? [16, 16] : [11, 11]
    });
}

function sampleAreaRadiusMeters(data, radiusValue=sampleRadius) {
    const product = data?.products?.ortho_sr?.available ? data.products.ortho_sr : data?.products?.ortho_radiance;
    const pixelSize = product?.grid?.pixel_size_m;
    if (!Array.isArray(pixelSize) || pixelSize.length < 2) return null;
    const meters = (Math.abs(Number(pixelSize[0])) + Math.abs(Number(pixelSize[1]))) / 2;
    return Number.isFinite(meters) ? meters * (radiusValue ? 1.5 : 0.5) : null;
}

function addSampleMarker(latlng, label, color, data, layerGroup, radiusValue=sampleRadius) {
    L.marker(latlng, {
        icon: sampleIcon(label, color, Boolean(radiusValue)),
        interactive: false
    }).addTo(layerGroup);
    const radius = sampleAreaRadiusMeters(data, radiusValue);
    if (radius) {
        L.circle(latlng, {
            radius,
            color,
            weight: 1.5,
            dashArray: radiusValue ? '4 4' : null,
            fillColor: color,
            fillOpacity: radiusValue ? 0.14 : 0.08,
            interactive: false,
            className: 'sample-area-ring'
        }).addTo(layerGroup);
    }
}

function clearActiveSampleMarker() {
    if (activeSampleMarker) {
        map.removeLayer(activeSampleMarker);
        activeSampleMarker = null;
    }
    if (activeSampleArea) {
        map.removeLayer(activeSampleArea);
        activeSampleArea = null;
    }
    updateUndoSampleButton();
}

function setActiveSampleMarker(latlng, data, pending=false) {
    clearActiveSampleMarker();
    const color = pending ? '#356d57' : '#ffffff';
    activeSampleMarker = L.marker(latlng, {
        icon: sampleIcon('P', color, Boolean(sampleRadius), pending),
        interactive: false
    }).addTo(map);
    updateUndoSampleButton();
    if (pending) return;
    const radius = sampleAreaRadiusMeters(data);
    if (radius) {
        activeSampleArea = L.circle(latlng, {
            radius,
            color,
            weight: 1.5,
            dashArray: sampleRadius ? '4 4' : null,
            fillColor: color,
            fillOpacity: sampleRadius ? 0.14 : 0.08,
            interactive: false,
            className: 'sample-area-ring'
        }).addTo(map);
    }
}

function renderCompareMarkers() {
    sampleMarkers.clearLayers();
    compareSamples.filter(sample => sample.data?.sample_type !== 'roi').forEach(sample => {
        addSampleMarker([sample.lat, sample.lon], sample.label, sample.color, sample.data, sampleMarkers, sample.radius);
    });
    updateUndoSampleButton();
}

function renderCompareAreas() {
    compareAreaLayers.clearLayers();
    compareSamples.filter(sample => sample.data?.sample_type === 'roi').forEach(sample => {
        const latlngs = sample.geometry.coordinates[0].map(([lon, lat]) => [lat, lon]);
        L.polygon(latlngs, {
            ...ROI_STYLE,
            color: sample.color,
            fillColor: sample.color,
        }).addTo(compareAreaLayers);
        L.marker([sample.lat, sample.lon], {
            icon: sampleIcon(sample.label, sample.color, true),
            interactive: false,
        }).addTo(compareAreaLayers);
    });
    updateUndoSampleButton();
}

function updateUndoSampleButton() {
    if (!undoSampleButton) return;
    const hasPoint = compareSamples.length > 0 ||
        Boolean(activeSampleMarker && lastSpectrumData?.sample_type !== 'roi');
    undoSampleButton.disabled = !hasPoint;
}

function renderRestoredPointResult(data) {
    lastSpectrumData = data;
    lastSampleLatLng = L.latLng(data.clicked.lat, data.clicked.lon);
    if (data.sample_type === 'roi') renderRoiResult(data);
    else renderQa(data);
    redrawSpectra();
    updateWlUI();
    renderIndices(data);
    renderBandReadout(data);
    renderCoveringScenes();
    setSpectrumStatus(data.sample_type === 'roi'
        ? 'Compared area restored.'
        : `Spectrum restored at ${data.clicked.lat.toFixed(5)}, ${data.clicked.lon.toFixed(5)}.`, 'ready');
}

function undoLatestPoint() {
    if (compareSamples.length) {
        compareSamples.pop();
        compareSamples.forEach((sample, index) => {
            sample.label = `${sample.data?.sample_type === 'roi' ? 'A' : 'P'}${index + 1}`;
        });
        renderCompareMarkers();
        renderCompareAreas();
        renderCompareList();
        if (compareSamples.length) {
            const restored = compareSamples[compareSamples.length - 1];
            if (restored.data?.sample_type === 'roi') {
                roiGeometry = JSON.parse(JSON.stringify(restored.geometry));
                roiShapeKind = restored.shape || 'custom';
            }
            renderRestoredPointResult(restored.data);
        } else {
            lastSpectrumData = null;
            lastSampleLatLng = null;
            roiGeometry = null;
            roiShapeKind = null;
            roiSummary.innerHTML = '';
            sampleMarkers.clearLayers();
            clearSpectrum();
        }
        updateUndoSampleButton();
        return;
    }
    if (!activeSampleMarker || lastSpectrumData?.sample_type === 'roi') return;
    clearActiveSampleMarker();
    lastSpectrumData = null;
    lastSampleLatLng = null;
    clearSpectrum();
    updateUndoSampleButton();
}

/* ── Derived indices + band readout ─────────────────────────────────── */
function renderIndices(data) {
    const entries = Object.entries(data?.indices || {});
    const validCount = entries.filter(([, index]) => !index.invalid).length;
    indexGrid.innerHTML = entries.length
        ? entries.map(([name, index]) => `
            <div class="metric-row">
                <span class="metric-copy"><strong class="metric-name">${escapeHtml(name)}</strong><span class="metric-formula">${escapeHtml(index.formula || '')}</span></span>
                <span class="metric-value ${index.invalid ? 'na' : ''}">${index.invalid ? '—' : Number(index.value).toFixed(4)}</span>
            </div>
        `).join('')
        : '<div class="grid-note">Surface-reflectance indices appear after a successful sample.</div>';
    if (indicesSummary) {
        indicesSummary.textContent = entries.length
            ? `${validCount} of ${entries.length} available`
            : 'Awaiting sample';
    }
}

function nearestValue(product, target) {
    const wavelengths = product?.wavelengths || [];
    const values = product?.values || [];
    if (!wavelengths.length || !values.length) return null;
    let best = 0;
    let bestDelta = Infinity;
    wavelengths.forEach((w, i) => {
        const delta = Math.abs(Number(w) - target);
        if (delta < bestDelta) {
            best = i;
            bestDelta = delta;
        }
    });
    return { wavelength: wavelengths[best], value: values[best], delta: bestDelta };
}

function renderBandReadout(data) {
    const selected = data?.products?.[spectrumProduct.value];
    const product = selected?.available
        ? selected
        : (data?.products?.ortho_sr?.available ? data.products.ortho_sr : data?.products?.ortho_radiance);
    const bands = [
        ['Blue', 490],
        ['Green', 560],
        ['Red', 665],
        ['Red edge', 705],
        ['NIR', 865],
        ['SWIR1', 1610],
        ['SWIR2', 2200]
    ];
    bandGrid.innerHTML = product?.available
        ? bands.map(([name, target]) => {
            const match = nearestValue(product, target);
            const missing = match?.value === null || match?.value === undefined;
            const val = missing ? '—' : Number(match.value).toPrecision(4);
            return `<div class="band-row"><span class="band-name">${name} · ${target} nm</span><span class="band-value ${missing ? 'na' : ''}">${val}</span></div>`;
        }).join('')
        : '<div class="grid-note">Band readout appears after a successful sample.</div>';
    if (bandsSummary) {
        bandsSummary.textContent = product?.available ? '7 reference bands' : 'Awaiting sample';
    }
}

/* ── Exports ────────────────────────────────────────────────────────── */
function downloadText(filename, mime, text) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportJsonReport() {
    if (!selectedScene || !lastSpectrumData) return setSpectrumStatus('Sample a point or area before exporting JSON.', 'error');
    const report = {
        exported_at: new Date().toISOString(),
        scene: selectedScene,
        scene_health: sceneScienceFor(selectedScene),
        active_overlay: currentOverlayKey || 'scene_thumbnail_only',
        samples: compareSamples.length ? compareSamples.map(s => s.data) : [lastSpectrumData]
    };
    downloadText(`${selectedScene.item_id}_scene_report.json`, 'application/json', JSON.stringify(report, null, 2));
}

function spectraCsvText() {
    const rows = ['sample,product,statistic,wavelength_nm,value'];
    const defaultLabel = lastSpectrumData?.sample_type === 'roi' ? 'ROI' : 'P1';
    const samples = compareSamples.length ? compareSamples : [{ label: defaultLabel, data: lastSpectrumData }];
    samples.forEach(sample => {
        Object.entries(sample.data.products || {}).forEach(([productName, product]) => {
            if (!product?.available) return;
            const series = product.statistics
                ? [
                    ['median', product.statistics.median],
                    ['average', product.statistics.mean]
                ]
                : [['value', product.values]];
            series.forEach(([statistic, values]) => {
                (product.wavelengths || []).forEach((wavelength, i) => {
                    rows.push([sample.label, productName, statistic, wavelength, values?.[i] ?? ''].join(','));
                });
            });
        });
    });
    return rows.join('\n');
}

function exportSpectraCsv() {
    if (!selectedScene || !lastSpectrumData) return setSpectrumStatus('Sample a point or area before exporting CSV.', 'error');
    downloadText(`${selectedScene.item_id}_scene_spectra.csv`, 'text/csv', spectraCsvText());
}

function canvasBlob(canvas, type='image/png') {
    return new Promise((resolve, reject) => {
        canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not encode the PNG canvas.')), type);
    });
}

function exportedSampleDocument() {
    if (lastSpectrumData?.sample_type === 'roi' && lastSpectrumData.geometry) {
        return {
            type: 'FeatureCollection',
            properties: {
                schema: 'tanager-workbench-sample/v1',
                scene_id: selectedScene.item_id,
                exported_at: new Date().toISOString()
            },
            features: [currentSampleFeature()]
        };
    }
    const pointSamples = compareSamples.length
        ? compareSamples
        : [{
            label: 'P1',
            lat: lastSpectrumData.clicked.lat,
            lon: lastSpectrumData.clicked.lon,
            radius: sampleRadius
        }];
    return {
        type: 'FeatureCollection',
        properties: {
            schema: 'tanager-workbench-sample/v1',
            scene_id: selectedScene.item_id,
            exported_at: new Date().toISOString()
        },
        features: pointSamples.map((sample, index) => ({
            type: 'Feature',
            properties: {
                scene_id: selectedScene.item_id,
                sample_type: 'point',
                label: sample.label || `P${index + 1}`,
                radius_pixels: Number(sample.radius ?? sampleRadius)
            },
            geometry: {
                type: 'Point',
                coordinates: [Number(sample.lon), Number(sample.lat)]
            }
        }))
    };
}

function snapshotSeries(productName, fallbackColor) {
    if (compareSamples.length) {
        return compareSamples.map(sample => {
            const product = sample.data.products?.[productName];
            if (!product?.available) return null;
            return {
                ...filteredBandSeries(product, product.values || []),
                label: sample.label,
                color: sample.color
            };
        }).filter(Boolean);
    }
    const product = lastSpectrumData.products?.[productName];
    if (!product?.available) return [];
    return [
        {
            ...filteredBandSeries(product, product.values || []),
            label: product.statistics ? 'Area median' : 'Point',
            color: fallbackColor
        },
        ...(product.statistics?.mean
            ? [{
                ...filteredBandSeries(product, product.statistics.mean),
                label: 'Area average',
                color: fallbackColor,
                dashed: true
            }]
            : [])
    ];
}

function drawSnapshotChart(ctx, bounds, title, yTitle, series) {
    const { x, y, width, height } = bounds;
    const margin = { left: 92, right: 28, top: 60, bottom: 62 };
    const plot = {
        x: x + margin.left,
        y: y + margin.top,
        width: width - margin.left - margin.right,
        height: height - margin.top - margin.bottom
    };
    ctx.fillStyle = '#e7e9ea';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = '#aeb5ba';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
    ctx.fillStyle = '#20262b';
    ctx.font = '600 22px Arial, sans-serif';
    ctx.fillText(title, x + 22, y + 34);

    const values = series.flatMap(item => item.y.filter(value => Number.isFinite(Number(value))).map(Number));
    const yMin = Math.min(0, ...values);
    const rawMax = Math.max(...values, 1);
    const yMax = rawMax === yMin ? yMin + 1 : rawMax * 1.04;

    ctx.font = '16px Arial, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 5; i++) {
        const ratio = i / 5;
        const py = plot.y + plot.height - ratio * plot.height;
        const value = yMin + ratio * (yMax - yMin);
        ctx.strokeStyle = '#c5cacc';
        ctx.beginPath();
        ctx.moveTo(plot.x, py);
        ctx.lineTo(plot.x + plot.width, py);
        ctx.stroke();
        ctx.fillStyle = '#4d5961';
        ctx.fillText(Number(value).toPrecision(3), plot.x - 10, py);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
        const ratio = i / 5;
        const px = plot.x + ratio * plot.width;
        const wavelength = wlMin + ratio * (wlMax - wlMin);
        ctx.strokeStyle = '#c5cacc';
        ctx.beginPath();
        ctx.moveTo(px, plot.y);
        ctx.lineTo(px, plot.y + plot.height);
        ctx.stroke();
        ctx.fillStyle = '#4d5961';
        ctx.fillText(Math.round(wavelength), px, plot.y + plot.height + 10);
    }

    ctx.fillStyle = '#303a41';
    ctx.font = '17px Arial, sans-serif';
    ctx.fillText('Wavelength (nm)', plot.x + plot.width / 2, y + height - 25);
    ctx.save();
    ctx.translate(x + 23, plot.y + plot.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yTitle, 0, 0);
    ctx.restore();

    series.forEach(item => {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2.5;
        ctx.setLineDash(item.dashed ? [10, 7] : []);
        ctx.beginPath();
        let drawing = false;
        item.x.forEach((wavelength, index) => {
            const value = Number(item.y[index]);
            if (!Number.isFinite(value)) {
                drawing = false;
                return;
            }
            const px = plot.x + ((Number(wavelength) - wlMin) / (wlMax - wlMin)) * plot.width;
            const py = plot.y + plot.height - ((value - yMin) / (yMax - yMin)) * plot.height;
            if (!drawing) {
                ctx.moveTo(px, py);
                drawing = true;
            } else {
                ctx.lineTo(px, py);
            }
        });
        ctx.stroke();
    });
    ctx.setLineDash([]);

    if (series.length > 1) {
        let legendX = plot.x;
        const legendY = y + 42;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.font = '15px Arial, sans-serif';
        series.forEach(item => {
            ctx.strokeStyle = item.color;
            ctx.lineWidth = 3;
            ctx.setLineDash(item.dashed ? [8, 5] : []);
            ctx.beginPath();
            ctx.moveTo(legendX, legendY);
            ctx.lineTo(legendX + 28, legendY);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#303a41';
            ctx.fillText(item.label, legendX + 36, legendY);
            legendX += 46 + ctx.measureText(item.label).width;
        });
    }
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
}

async function buildSpectrumSnapshotBlob() {
    const width = 1500;
    const chartHeight = 500;
    const padding = 32;
    const header = 92;
    const gap = 20;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = header + chartHeight * 2 + gap + padding;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#20262b';
    ctx.font = '600 28px Arial, sans-serif';
    ctx.fillText(`Tanager-1 scene evidence: ${selectedScene.item_id}`, padding, 36);
    ctx.fillStyle = '#56636d';
    ctx.font = '17px Arial, sans-serif';
    const sampleDescription = lastSpectrumData.sample_type === 'roi'
        ? 'polygon area · median and average'
        : `point ${lastSpectrumData.clicked.lat.toFixed(5)}, ${lastSpectrumData.clicked.lon.toFixed(5)} · single pixel`;
    ctx.fillText(`${sampleDescription} | ${new Date().toISOString()}`, padding, 65);
    drawSnapshotChart(
        ctx,
        { x: padding, y: header, width: width - padding * 2, height: chartHeight },
        'TOA radiance',
        'W/(m² sr µm)',
        snapshotSeries('ortho_radiance', SERIES.radiance)
    );
    drawSnapshotChart(
        ctx,
        { x: padding, y: header + chartHeight + gap, width: width - padding * 2, height: chartHeight },
        'Surface reflectance',
        'Reflectance (unitless)',
        snapshotSeries('ortho_sr', SERIES.reflectance)
    );
    return canvasBlob(canvas);
}

async function exportPlotPng() {
    if (!window.Plotly || !lastSpectrumData) return setSpectrumStatus('Sample a point or area before exporting PNG.', 'error');
    const exportButton = document.getElementById('export-png');
    let exportStage = 'preparing plots';
    try {
        exportButton.disabled = true;
        setSpectrumStatus('Building PNG snapshot and sample file…');
        exportStage = 'encoding the PNG';
        const pngBlob = await buildSpectrumSnapshotBlob();
        exportStage = 'starting downloads';
        const baseName = `${selectedScene.item_id}_spectrum`;
        downloadBlob(`${baseName}.png`, pngBlob);
        window.setTimeout(() => {
            downloadText(
                `${baseName}_coordinates.geojson`,
                'application/geo+json',
                JSON.stringify(exportedSampleDocument(), null, 2)
            );
        }, 180);
        setSpectrumStatus('PNG and re-loadable sample coordinates exported.', 'ready');
    } catch (error) {
        const detail = error?.message || String(error || 'unknown browser error');
        setSpectrumStatus(`PNG export failed while ${exportStage}: ${detail}`, 'error');
    } finally {
        exportButton.disabled = false;
    }
}

async function importSampleDocument(file) {
    if (!file) return;
    try {
        const documentData = JSON.parse(await file.text());
        const features = documentData.type === 'FeatureCollection'
            ? documentData.features
            : documentData.type === 'Feature'
                ? [documentData]
                : [];
        if (!features.length) throw new Error('No GeoJSON sample features were found.');
        const supported = features.filter(feature => ['Point', 'Polygon'].includes(feature?.geometry?.type));
        if (!supported.length) throw new Error('The file must contain Point or Polygon features.');
        const sceneId = documentData.properties?.scene_id ||
            supported.find(feature => feature.properties?.scene_id)?.properties.scene_id;
        if (sceneId && selectedScene?.item_id !== sceneId) {
            const row = SCENES.find(scene => scene.item_id === sceneId);
            if (!row) throw new Error(`Scene ${sceneId} is not present in this catalogue.`);
            showScene(row, COLORS[row.collection] || '#888');
        }
        if (!selectedScene) throw new Error('Select a scene, or load a sample file containing scene_id.');

        const polygon = supported.find(feature => feature.geometry.type === 'Polygon');
        if (polygon) {
            compareMode = null;
            syncCompareButtons();
            const geometry = polygon.geometry;
            clearRoiLayer();
            roiGeometry = geometry;
            roiShapeKind = 'custom';
            roiLayer = L.polygon(
                geometry.coordinates[0].map(([lon, lat]) => [lat, lon]),
                ROI_STYLE
            ).addTo(map);
            setActiveRoiShape('custom');
            await sampleRoi(geometry);
        } else {
            const points = supported.filter(feature => feature.geometry.type === 'Point');
            compareSamples = [];
            sampleMarkers.clearLayers();
            clearActiveSampleMarker();
            compareMode = points.length > 1 ? 'point' : null;
            syncCompareButtons();
            for (const feature of points) {
                const [lon, lat] = feature.geometry.coordinates;
                if (!Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) {
                    throw new Error('A point contains invalid coordinates.');
                }
                await sampleSpectrum(L.latLng(Number(lat), Number(lon)));
            }
        }
        activateDrawerTab('spectrum');
        setSpectrumStatus(`Loaded ${supported.length} sample ${supported.length === 1 ? 'location' : 'locations'} from ${file.name}.`, 'ready');
    } catch (error) {
        setSpectrumStatus(`Sample import failed: ${error.message}`, 'error');
    } finally {
        importSampleInput.value = '';
    }
}

function currentFilterState() {
    return {
        search: searchInput.value,
        collections: [...activeColls],
        max_cloud_percent: Number(cloudSlider.value),
        acquired_from: dateFromInput.value || null,
        acquired_to: dateToInput.value || null,
        collection_mode: modeSelect.value || null,
        min_sun_elevation_deg: Number(sunSlider.value),
        max_off_nadir_deg: Number(offNadirSlider.value),
        max_light_haze_percent: Number(hazeSlider.value)
    };
}

function currentSampleFeature() {
    if (lastSpectrumData?.sample_type === 'roi' && lastSpectrumData.geometry) {
        return {
            type: 'Feature',
            properties: { scene_id: selectedScene.item_id, sample_type: 'roi' },
            geometry: lastSpectrumData.geometry
        };
    }
    return {
        type: 'Feature',
        properties: {
            scene_id: selectedScene.item_id,
            sample_type: 'point',
            radius_pixels: sampleRadius
        },
        geometry: {
            type: 'Point',
            coordinates: [lastSpectrumData.clicked.lon, lastSpectrumData.clicked.lat]
        }
    };
}

function reproductionScript(feature) {
    const header = `"""Reproduce this Tanager workbench API extraction.

1. Start the workbench locally: python serve.py
2. Install requests if needed: python -m pip install requests
3. Run this file: python reproduce.py
"""

import json
import requests

BASE_URL = "http://127.0.0.1:3000"
SCENE_ID = ${JSON.stringify(selectedScene.item_id)}
`;
    if (feature.geometry.type === 'Polygon') {
        return header + `
payload = {
    "scene_id": SCENE_ID,
    "geometry": ${JSON.stringify(feature.geometry, null, 4)},
    "products": ["ortho_radiance", "ortho_sr"],
}
response = requests.post(f"{BASE_URL}/api/roi", json=payload, timeout=180)
response.raise_for_status()
result = response.json()
`;
    }
    return header + `
params = {
    "scene_id": SCENE_ID,
    "lat": ${Number(lastSpectrumData.clicked.lat)},
    "lon": ${Number(lastSpectrumData.clicked.lon)},
    "radius": ${Number(sampleRadius)},
    "products": "ortho_radiance,ortho_sr",
}
response = requests.get(f"{BASE_URL}/api/spectrum", params=params, timeout=180)
response.raise_for_status()
result = response.json()
`;
}

async function exportEvidencePackage() {
    if (!selectedScene || !lastSpectrumData) {
        return setSpectrumStatus('Sample a point or area before creating an evidence package.', 'error');
    }
    if (!window.JSZip || !window.Plotly) {
        return setSpectrumStatus('Evidence export libraries did not load. Refresh and try again.', 'error');
    }
    setSpectrumStatus('Building evidence package…');
    try {
        const feature = currentSampleFeature();
        const report = {
            exported_at: new Date().toISOString(),
            workbench_url: window.location.href,
            scene: selectedScene,
            active_filters: currentFilterState(),
            method: {
                point: 'single pixel',
                roi: lastSpectrumData.sample_type === 'roi'
                    ? 'pixel-centre polygon inclusion; median and arithmetic average'
                    : null,
                bad_band_policy: 'surface-reflectance bands flagged good_wavelengths=false are excluded'
            },
            api_result: lastSpectrumData
        };
        const readme = `TANAGER WORKBENCH EVIDENCE PACKAGE
Scene: ${selectedScene.item_id}
Created: ${report.exported_at}
Sample: ${lastSpectrumData.sample_type === 'roi' ? 'polygon area' : 'point'}

FILES
- analysis.json: scene metadata, filters, methodology and complete API result
- spectra.csv: plotted point values or area median and average by wavelength
- sample.geojson: the exact point or polygon used
- reproduce.py: a small script that repeats the API request against a local server
- plots/: exported radiance and surface-reflectance figures

IMPORTANT LIMITATIONS
- Scene matching uses published bounding boxes; inspect each preview before a multi-date comparison.
- Polygon pixels are selected by projected pixel-centre inclusion.
- Large accepted polygons may use the analysis stride recorded in analysis.json for spectral statistics.
- Surface-reflectance wavelengths flagged as bad by the product are removed.
- Cloud and cirrus fractions describe the selected area; they are reported, not silently removed.
`;
        const spectrumSnapshotBlob = await buildSpectrumSnapshotBlob();
        const zip = new JSZip();
        zip.file('README.txt', readme);
        zip.file('analysis.json', JSON.stringify(report, null, 2));
        zip.file('spectra.csv', spectraCsvText());
        zip.file('sample.geojson', JSON.stringify(feature, null, 2));
        zip.file('reproduce.py', reproductionScript(feature));
        zip.file('plots/spectrum_snapshot.png', spectrumSnapshotBlob);
        if (lastCompositeBlob && lastCompositeSceneId === selectedScene.item_id) {
            zip.file('composite/composite.png', lastCompositeBlob);
            zip.file('composite/composite_metadata.json', JSON.stringify(lastCompositeMeta, null, 2));
        }
        const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${selectedScene.item_id}_evidence_package.zip`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setSpectrumStatus('Evidence package created.', 'ready');
    } catch (error) {
        setSpectrumStatus(`Evidence package failed: ${error.message}`, 'error');
    }
}

async function copySceneLink() {
    if (!selectedScene) return;
    updateUrlState(selectedScene.item_id);
    const url = new URL(window.location.href);
    try {
        await navigator.clipboard.writeText(url.toString());
        setSpectrumStatus('Scene link copied.', 'ready');
    } catch {
        setSpectrumStatus(url.toString(), 'ready');
    }
}

/* ── Polygon area sampling ──────────────────────────────────────────── */
const ROI_STYLE = {
    color: '#356d57',
    weight: 2,
    fillColor: '#356d57',
    fillOpacity: 0.12,
    className: 'roi-polygon'
};

function setActiveRoiShape(shape) {
    roiShapeButtons.forEach(button => {
        button.classList.toggle('active', button.dataset.roiShape === shape);
    });
}

function clearRoiLayer() {
    if (roiLayer) {
        map.removeLayer(roiLayer);
        roiLayer = null;
    }
    roiGeometry = null;
    roiShapeKind = null;
    roiSummary.innerHTML = '';
    setActiveRoiShape(null);
}

function clearRoiAnalysis(clearResults=true) {
    roiRequestId++;
    clearPresetRoiPreview();
    if (roiDrawTool) {
        roiDrawTool.disable();
        roiDrawTool = null;
    }
    roiDrawing = false;
    roiPlacementShape = null;
    const wasRoiResult = lastSpectrumData?.sample_type === 'roi';
    clearRoiLayer();
    if (clearResults && wasRoiResult) {
        lastSpectrumData = null;
        clearSpectrum();
    }
}

function armPresetRoi(shape) {
    if (!selectedScene) {
        setSpectrumStatus('Select a scene before placing an analysis area.', 'error');
        return;
    }
    if (roiDrawTool) roiDrawTool.disable();
    const comparingAreas = compareMode === 'area';
    clearRoiAnalysis(!comparingAreas && lastSpectrumData?.sample_type === 'roi');
    if (!comparingAreas) {
        compareMode = null;
        resetComparedSamples();
        syncCompareButtons();
    }
    roiPlacementShape = shape;
    activeMapTool = 'area';
    setActiveRoiShape(shape);
    activateDrawerTab('spectrum');
    setSpectrumStatus(`Press inside the scene and drag to size the ${shape}. Release to analyse.`);
    setModeState();
}

function clearPresetRoiPreview() {
    if (roiDragPreview) {
        map.removeLayer(roiDragPreview);
        roiDragPreview = null;
    }
    if (roiDragTooltip) {
        map.removeLayer(roiDragTooltip);
        roiDragTooltip = null;
    }
    roiDragStart = null;
    roiDragGeometry = null;
    roiDragging = false;
    if (!map.dragging.enabled()) map.dragging.enable();
}

function cancelPresetRoiDrag(disarm=false) {
    clearPresetRoiPreview();
    if (disarm) {
        roiPlacementShape = null;
        setActiveRoiShape(null);
    }
    setModeState();
}

function presetGeometryFromPointer(origin, pointer, shape) {
    const originPoint = map.latLngToContainerPoint(origin);
    const pointerPoint = map.latLngToContainerPoint(pointer);
    const dx = pointerPoint.x - originPoint.x;
    const dy = pointerPoint.y - originPoint.y;
    let offsets = [];
    if (shape === 'square') {
        const side = Math.max(Math.abs(dx), Math.abs(dy));
        const edgeX = (dx < 0 ? -1 : 1) * side;
        const edgeY = (dy < 0 ? -1 : 1) * side;
        offsets = [[0, 0], [edgeX, 0], [edgeX, edgeY], [0, edgeY]];
    } else if (shape === 'rectangle') {
        offsets = [[0, 0], [dx, 0], [dx, dy], [0, dy]];
    } else {
        const left = Math.min(0, dx);
        const right = Math.max(0, dx);
        const top = Math.min(0, dy);
        const bottom = Math.max(0, dy);
        const width = right - left;
        const height = bottom - top;
        offsets = [
            [left + width * 0.5, top],
            [right, top + height * 0.38],
            [right - width * 0.19, bottom],
            [left + width * 0.19, bottom],
            [left, top + height * 0.38]
        ];
    }
    const latlngs = offsets.map(([x, y]) => {
        const point = L.point(originPoint.x + x, originPoint.y + y);
        return map.containerPointToLatLng(point);
    });
    const ring = latlngs.map(latlng => [latlng.lng, latlng.lat]);
    ring.push([...ring[0]]);
    const xs = offsets.map(([x]) => x);
    const ys = offsets.map(([, y]) => y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const horizontalStart = map.containerPointToLatLng([originPoint.x + minX, originPoint.y]);
    const horizontalEnd = map.containerPointToLatLng([originPoint.x + maxX, originPoint.y]);
    const verticalStart = map.containerPointToLatLng([originPoint.x, originPoint.y + minY]);
    const verticalEnd = map.containerPointToLatLng([originPoint.x, originPoint.y + maxY]);
    return {
        geometry: { type: 'Polygon', coordinates: [ring] },
        latlngs,
        widthM: map.distance(horizontalStart, horizontalEnd),
        heightM: map.distance(verticalStart, verticalEnd),
        screenSize: Math.hypot(dx, dy)
    };
}

function formatRoiDimension(metres) {
    return metres >= 1000 ? `${(metres / 1000).toFixed(metres >= 10000 ? 0 : 1)} km` : `${Math.round(metres)} m`;
}

function beginPresetRoiDrag(event) {
    if (!roiPlacementShape || !selectedBounds?.contains(event.latlng)) return;
    if (event.originalEvent?.button !== undefined && event.originalEvent.button !== 0) return;
    L.DomEvent.stop(event.originalEvent);
    roiDragStart = event.latlng;
    roiDragging = true;
    roiDragGeometry = null;
    map.dragging.disable();
    setModeState();
}

function updatePresetRoiDrag(event) {
    if (!roiDragging || !roiDragStart || !roiPlacementShape) return;
    const result = presetGeometryFromPointer(roiDragStart, event.latlng, roiPlacementShape);
    roiDragGeometry = result;
    const insideScene = result.latlngs.every(latlng => selectedBounds?.contains(latlng));
    if (!roiDragPreview) {
        roiDragPreview = L.polygon(result.latlngs, {
            color: insideScene ? '#f7f7f7' : '#bc5a53',
            weight: 2,
            dashArray: '6 4',
            fillColor: insideScene ? '#2f6c58' : '#bc5a53',
            fillOpacity: 0.18,
            interactive: false,
            className: `roi-preview ${insideScene ? '' : 'invalid'}`
        }).addTo(map);
    } else {
        roiDragPreview.setLatLngs(result.latlngs);
        roiDragPreview.setStyle({
            color: insideScene ? '#f7f7f7' : '#bc5a53',
            fillColor: insideScene ? '#2f6c58' : '#bc5a53'
        });
    }
    const dimensionText = roiPlacementShape === 'pentagon'
        ? `Ø ${formatRoiDimension(result.widthM)}`
        : `${formatRoiDimension(result.widthM)} × ${formatRoiDimension(result.heightM)}`;
    if (!roiDragTooltip) {
        roiDragTooltip = L.tooltip({
            permanent: true,
            direction: 'top',
            className: 'roi-size-tooltip',
            offset: [0, -8],
            interactive: false
        }).setLatLng(event.latlng).setContent(`${dimensionText}${insideScene ? ' · release to analyse' : ' · outside scene'}`).addTo(map);
    } else {
        roiDragTooltip.setLatLng(event.latlng).setContent(`${dimensionText}${insideScene ? ' · release to analyse' : ' · outside scene'}`);
    }
}

function finishPresetRoiDrag() {
    if (!roiDragging) return;
    const shape = roiPlacementShape;
    const result = roiDragGeometry;
    const valid = result
        && result.screenSize >= 7
        && result.latlngs.every(latlng => selectedBounds?.contains(latlng));
    clearPresetRoiPreview();
    if (!valid) {
        setSpectrumStatus(result?.screenSize >= 7
            ? 'Keep the full area inside the selected scene and release again.'
            : 'Drag farther to create a measurable analysis area.', 'error');
        setModeState();
        return;
    }
    suppressMapClickUntil = Date.now() + 350;
    roiPlacementShape = null;
    clearRoiLayer();
    roiShapeKind = shape;
    roiGeometry = result.geometry;
    roiLayer = L.polygon(
        result.geometry.coordinates[0].map(([lon, lat]) => [lat, lon]),
        ROI_STYLE
    ).addTo(map);
    setActiveRoiShape(shape);
    setModeState();
    sampleRoi(result.geometry);
}

map.on('mousedown', beginPresetRoiDrag);
map.on('mousemove', updatePresetRoiDrag);
map.on('mouseup', finishPresetRoiDrag);
document.addEventListener('mouseup', finishPresetRoiDrag);

function armRoiDraw() {
    if (!selectedScene) {
        setSpectrumStatus('Select a scene before drawing an analysis area.', 'error');
        return;
    }
    if (!window.L?.Draw?.Polygon) {
        setSpectrumStatus('Area drawing library did not load. Refresh the page and try again.', 'error');
        return;
    }
    if (roiDrawTool) roiDrawTool.disable();
    const comparingAreas = compareMode === 'area';
    clearRoiAnalysis(!comparingAreas && lastSpectrumData?.sample_type === 'roi');
    if (!comparingAreas) {
        compareMode = null;
        resetComparedSamples();
        syncCompareButtons();
    }
    roiPlacementShape = null;
    roiDrawing = true;
    activeMapTool = 'area';
    setActiveRoiShape('custom');
    activateDrawerTab('spectrum');
    setSpectrumStatus('Draw a polygon on the scene; click the first vertex to finish.');
    setModeState();
    roiDrawTool = new L.Draw.Polygon(map, {
        allowIntersection: false,
        showArea: true,
        shapeOptions: ROI_STYLE
    });
    roiDrawTool.enable();
}

map.on('draw:created', event => {
    roiDrawing = false;
    suppressMapClickUntil = Date.now() + 500;
    roiDrawTool = null;
    clearRoiLayer();
    roiLayer = event.layer;
    roiLayer.addTo(map);
    roiGeometry = roiLayer.toGeoJSON().geometry;
    roiShapeKind = 'custom';
    setActiveRoiShape('custom');
    sampleRoi(roiGeometry);
});

map.on('draw:drawstop', () => {
    roiDrawing = false;
    roiDrawTool = null;
    if (!roiGeometry) setActiveRoiShape(null);
    setModeState();
});

/* ── Scene panel lifecycle ──────────────────────────────────────────── */
function closePanel() {
    const hadScene = Boolean(selectedScene);
    inspector.classList.remove('open');
    clearCompositePreview();
    if (currentOverlay) { map.removeLayer(currentOverlay); currentOverlay = null; }
    if (currentBorder)  { map.removeLayer(currentBorder);  currentBorder  = null; }
    if (currentScienceOverlay) { map.removeLayer(currentScienceOverlay); currentScienceOverlay = null; }
    if (currentAlignedOverlayUrl) { URL.revokeObjectURL(currentAlignedOverlayUrl); currentAlignedOverlayUrl = null; }
    sampleMarkers.clearLayers();
    clearActiveSampleMarker();
    selectedScene = null;
    syncCoastalAnalysis();
    syncGhgAnalysis();
    document.getElementById('analysis-scene-id').textContent = 'No scene';
    selectedBounds = null;
    selectedThumbPath = null;
    lastSampleLatLng = null;
    lastSpectrumData = null;
    compareSamples = [];
    compareAreaLayers.clearLayers();
    compareMode = null;
    syncCompareButtons();
    currentOverlayKey = null;
    activeMapTool = 'browse';
    spectrumRequestId++;
    clearRoiAnalysis(false);
    clearSpectrum(false);
    renderOverlayControls(null, null);
    activateInspectorTab('overview');
    setModeState();
    if (hadScene) {
        const url = new URL(window.location.href);
        url.searchParams.delete('scene');
        window.history.replaceState(null, '', url);
        renderSceneTable(filteredRows());
        renderMapLayers(filteredRows());
        queueMapResize();
    }
    if (hadScene) setWorkspaceTab('overview', { promptForScene: false });
}

document.getElementById('sp-close').addEventListener('click', closePanel);

// Sample selected scenes, or close the panel when clicking outside the footprint.
map.on('click', e => {
    if (roiPlacementShape) {
        if (!selectedBounds?.contains(e.latlng)) setSpectrumStatus('Press inside the selected scene, then drag to size the area.', 'error');
        return;
    }
    if (roiDrawing || Date.now() < suppressMapClickUntil) return;
    if (e.originalEvent._fromMarker) return;
    if (selectedScene && selectedBounds && selectedBounds.contains(e.latlng) && activeMapTool === 'point') {
        clearRoiAnalysis(lastSpectrumData?.sample_type === 'roi');
        sampleSpectrum(e.latlng);
        return;
    }
    if (!selectedBounds?.contains(e.latlng)) closePanel();
});

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

function setSpectrumStatus(text, state='') {
    const loading = /loading|fetching|building/i.test(text);
    spectrumStatus.textContent = text;
    spectrumStatus.className = `spectrum-status ${state} ${loading ? 'loading' : ''}`.trim();
    document.getElementById('analysis-drawer').classList.toggle('loading', loading);
    if (sampleState) {
        sampleState.textContent = text;
        sampleState.classList.toggle('ready', state === 'ready');
        sampleState.classList.toggle('error', state === 'error');
    }
    if (drawerContext) {
        drawerContext.textContent = state === 'ready'
            ? (lastSpectrumData?.sample_type === 'roi' ? 'Area spectrum ready' : 'Point spectrum ready')
            : text.replace(/\s*·.*$/, '').slice(0, 32);
    }
    if (sampleDiagnostics && state === 'ready' && lastSpectrumData) sampleDiagnostics.hidden = false;
    if (state === 'ready') activateDrawerTab('spectrum');
    if (!samplingSummary) return;
    if (state === 'error') samplingSummary.textContent = 'Needs attention';
    else if (state === 'ready') {
        samplingSummary.textContent = lastSpectrumData?.sample_type === 'roi'
            ? 'Area spectrum ready'
            : 'Point spectrum ready';
    } else if (/loading|fetching|building/i.test(text)) samplingSummary.textContent = 'Working…';
    else if (compareMode) samplingSummary.textContent = 'Compare mode';
    else samplingSummary.textContent = 'Point or area';
}

function clearSpectrum(collapse=true) {
    setSpectrumStatus(selectedScene ? 'Choose Point or an Area tool to sample the selected scene.' : 'Select a scene, then choose a sampling tool.');
    spectrumQa.innerHTML = '';
    roiSummary.innerHTML = '';
    if (sampleDiagnostics) sampleDiagnostics.hidden = true;
    compareList.innerHTML = '';
    coverageCount.textContent = 'Awaiting sample';
    coverageList.innerHTML = '<div class="grid-note">Sample a point or area to find other catalogue dates.</div>';
    indexGrid.innerHTML = '<div class="grid-note">Surface-reflectance indices appear after a successful sample.</div>';
    bandGrid.innerHTML = '<div class="grid-note">Band readout appears after a successful sample.</div>';
    if (indicesSummary) indicesSummary.textContent = 'Awaiting sample';
    if (bandsSummary) bandsSummary.textContent = 'Awaiting sample';
    if (window.Plotly) {
        Plotly.purge('spectrum-plot');
    }
    document.getElementById('analysis-drawer').classList.remove('loading');
    updateUndoSampleButton();
    if (collapse) setDrawerOpen(false);
}

/* ── Spectrum plots ─────────────────────────────────────────────────── */
function basePlotLayout(title, yTitle) {
    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: PLOT.font, color: PLOT.ink2, size: 12 },
        margin: { l: 48, r: 10, t: 26, b: 34 },
        title: { text: title, font: { size: 13, color: PLOT.ink }, x: 0.02, xanchor: 'left' },
        xaxis: {
            title: { text: 'Wavelength (nm)', font: { size: 11 } },
            gridcolor: PLOT.grid,
            zerolinecolor: PLOT.grid,
            linecolor: PLOT.axis,
            ticks: 'outside',
            tickcolor: PLOT.axis,
        },
        yaxis: {
            title: { text: yTitle || '', font: { size: 11 } },
            gridcolor: PLOT.grid,
            zerolinecolor: PLOT.grid,
            linecolor: PLOT.axis,
            ticks: 'outside',
            tickcolor: PLOT.axis,
            rangemode: 'tozero',
        },
    };
}

function renderUnavailable(plotId, title, reason) {
    const layout = basePlotLayout(title, '');
    layout.xaxis = { visible: false };
    layout.yaxis = { visible: false };
    layout.annotations = [{
        text: escapeHtml(reason || 'Unavailable'),
        x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
        showarrow: false, font: { color: PLOT.ink2, size: 11 }
    }];
    Plotly.react(plotId, [], layout, { displayModeBar: false, responsive: true });
}

function filteredBandSeries(product, values) {
    const wavelengths = product.wavelengths || [];
    const good = product.good_wavelengths;
    const x = [];
    const y = [];
    wavelengths.forEach((w, i) => {
        const wl = Number(w);
        if (wl < wlMin || wl > wlMax) return;
        x.push(w);
        let v = values[i];
        if (v === null || v === undefined) v = null;
        else if (Array.isArray(good) && good[i] === false) v = null;
        y.push(v);
    });
    return { x, y };
}

function productTrace(product, title, color, values=product.values || [], dash='solid') {
    const series = filteredBandSeries(product, values);
    return {
        x: series.x,
        y: series.y,
        name: title,
        type: 'scatter',
        mode: 'lines',
        line: { color, width: 1.6, dash },
        hovertemplate: `${escapeHtml(title)}<br>%{x:.1f} nm<br>%{y:.5g}<extra></extra>`,
        connectgaps: false
    };
}

function renderProduct(plotId, product, title, color, yTitle, compareProductName=null) {
    if (!window.Plotly) return;
    if (!product || !product.available) {
        renderUnavailable(plotId, title, product?.reason || 'Product unavailable');
        return;
    }

    const traces = compareProductName && compareSamples.length
        ? compareSamples
            .flatMap(sample => {
                const sampleProduct = sample.data.products?.[compareProductName];
                if (!sampleProduct?.available) return [];
                if (!sampleProduct.statistics) {
                    return [productTrace(sampleProduct, sample.label, sample.color)];
                }
                return [
                    productTrace(
                        sampleProduct,
                        `${sample.label} median`,
                        sample.color,
                        sampleProduct.statistics.median || sampleProduct.values
                    ),
                    ...(Array.isArray(sampleProduct.statistics.mean)
                        ? [productTrace(sampleProduct, `${sample.label} mean`, sample.color, sampleProduct.statistics.mean, 'dash')]
                        : [])
                ];
            })
        : [
            productTrace(product, product.statistics ? 'Area median' : title, color, product.statistics?.median || product.values),
            ...(Array.isArray(product.statistics?.mean)
                ? [productTrace(product, 'Area mean', color, product.statistics.mean, 'dash')]
                : [])
        ];

    const layout = basePlotLayout(title, yTitle || product.units || '');
    layout.showlegend = traces.length > 1;
    layout.legend = { orientation: 'h', y: 1.14, x: 0, font: { size: 11 } };
    Plotly.react(plotId, traces, layout, { displayModeBar: false, responsive: true });
}

// Re-render the selected spectrum product from the last loaded sample, applying the
// current wavelength window. Safe to call any time (no-op without data).
function redrawSpectra() {
    if (!lastSpectrumData) return;
    const productName = spectrumProduct.value;
    const config = SPECTRUM_PRODUCTS[productName];
    const product = lastSpectrumData.products?.[productName];
    const unavailableProduct = config?.available === false
        ? { available: false, reason: 'Kerchunk reference unavailable for this product.' }
        : product;
    renderProduct(
        'spectrum-plot',
        unavailableProduct,
        config?.label || 'Spectrum',
        config?.color || SERIES.reflectance,
        config?.units || '',
        productName
    );
}

/* ── Wavelength window control (dual slider over the spectral bar) ───── */
const wlMinInput = document.getElementById('wl-min');
const wlMaxInput = document.getElementById('wl-max');
const wlBubbleMin = document.getElementById('wl-bubble-min');
const wlBubbleMax = document.getElementById('wl-bubble-max');
const wlMaskLeft = document.getElementById('sr-mask-left');
const wlMaskRight = document.getElementById('sr-mask-right');
const wlSelected = document.getElementById('sr-selected');
const wlReadout = document.getElementById('wl-readout');
const wlResetBtn = document.getElementById('wl-reset');

function wlPercent(w) {
    return ((w - WL_FLOOR) / (WL_CEIL - WL_FLOOR)) * 100;
}

// Count how many bands of the current sample fall inside the window
function bandsInWindow() {
    const selected = lastSpectrumData?.products?.[spectrumProduct.value];
    const product = selected?.available
        ? selected
        : (lastSpectrumData?.products?.ortho_sr?.available
            ? lastSpectrumData.products.ortho_sr
            : lastSpectrumData?.products?.ortho_radiance);
    const wl = product?.wavelengths;
    if (!Array.isArray(wl)) return null;
    const good = product.good_wavelengths;
    return wl.reduce((n, w, index) => n + (
        Number(w) >= wlMin &&
        Number(w) <= wlMax &&
        (!Array.isArray(good) || good[index] !== false)
            ? 1
            : 0
    ), 0);
}

function updateWlUI() {
    const pMin = wlPercent(wlMin);
    const pMax = wlPercent(wlMax);
    wlMaskLeft.style.left = '0';
    wlMaskLeft.style.width = pMin + '%';
    wlMaskRight.style.left = pMax + '%';
    wlMaskRight.style.width = (100 - pMax) + '%';
    wlSelected.style.left = pMin + '%';
    wlSelected.style.width = (pMax - pMin) + '%';
    wlBubbleMin.style.left = pMin + '%';
    wlBubbleMax.style.left = pMax + '%';
    wlBubbleMin.textContent = Math.round(wlMin);
    wlBubbleMax.textContent = Math.round(wlMax);

    const full = wlMin <= WL_FLOOR && wlMax >= WL_CEIL;
    const bands = bandsInWindow();
    const bandTxt = bands !== null ? ` · <span class="accent">${bands} bands</span>` : '';
    wlReadout.innerHTML = full
        ? `Full range · ${WL_FLOOR}&ndash;${WL_CEIL} nm${bandTxt}`
        : `<span class="accent">${Math.round(wlMin)}&ndash;${Math.round(wlMax)} nm</span> window${bandTxt}`;
    wlResetBtn.disabled = full;
}

function onWlInput() {
    let lo = Number(wlMinInput.value);
    let hi = Number(wlMaxInput.value);
    if (lo > hi - WL_MIN_GAP) {
        // Push the other handle so they never cross
        if (document.activeElement === wlMinInput) {
            hi = Math.min(WL_CEIL, lo + WL_MIN_GAP);
            wlMaxInput.value = hi;
        } else {
            lo = Math.max(WL_FLOOR, hi - WL_MIN_GAP);
            wlMinInput.value = lo;
        }
    }
    wlMin = lo;
    wlMax = hi;
    updateWlUI();
    redrawSpectra();
}

wlMinInput.addEventListener('input', onWlInput);
wlMaxInput.addEventListener('input', onWlInput);
wlResetBtn.addEventListener('click', () => {
    wlMin = WL_FLOOR;
    wlMax = WL_CEIL;
    wlMinInput.value = WL_FLOOR;
    wlMaxInput.value = WL_CEIL;
    updateWlUI();
    redrawSpectra();
});
updateWlUI();

function renderQa(data) {
    if (sampleDiagnostics) sampleDiagnostics.hidden = false;
    const available = Object.values(data.products || {}).find(p => p && p.available);
    const sr = data.products?.ortho_sr;
    const qa = available?.qa || {};
    const badBandCount = Array.isArray(sr?.good_wavelengths)
        ? sr.good_wavelengths.filter(value => value === false).length
        : 0;
    const totalBandCount = Array.isArray(sr?.wavelengths) ? sr.wavelengths.length : 0;
    const pixelStatus = qa.is_nodata ? 'Nodata' : qa.is_cloud ? 'Cloud' : qa.is_cirrus ? 'Cirrus' : 'Usable';
    const cells = [
        ['Product', available?.label || '—'],
        ['Usable bands', totalBandCount ? totalBandCount - badBandCount : '—'],
        ['Pixel status', pixelStatus]
    ];
    roiSummary.innerHTML = cells.map(([key, value]) => `
        <div class="health-cell">
            <div class="health-k">${escapeHtml(key)}</div>
            <div class="health-v" title="${escapeHtml(value)}">${escapeHtml(value)}</div>
        </div>
    `).join('');
    const chips = [
        ...(qa.is_cloud ? [{ text: 'Cloud in sample', bad: true }] : []),
        ...(qa.is_cirrus ? [{ text: 'Cirrus in sample', bad: true }] : []),
        ...(qa.is_nodata ? [{ text: 'Nodata in sample', bad: true }] : []),
        ...(badBandCount ? [{ text: `${badBandCount} bad SR bands removed`, bad: false }] : [])
    ];
    spectrumQa.innerHTML = chips.map(chip =>
        `<span class="qa-chip ${chip.bad ? 'bad' : ''}">${escapeHtml(chip.text)}</span>`
    ).join('');
}

function formatArea(areaM2) {
    if (!Number.isFinite(Number(areaM2))) return '—';
    return areaM2 >= 1_000_000
        ? `${(areaM2 / 1_000_000).toFixed(2)} km²`
        : `${(areaM2 / 10_000).toFixed(2)} ha`;
}

function formatFraction(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderRoiResult(data) {
    if (sampleDiagnostics) sampleDiagnostics.hidden = false;
    const available = data.products?.ortho_sr?.available
        ? data.products.ortho_sr
        : Object.values(data.products || {}).find(product => product?.available);
    const roi = available?.roi || {};
    const qa = available?.qa || {};
    const cells = [
        ['Area', formatArea(roi.area_m2)],
        ['Selected pixels', roi.selected_pixel_count ?? '—'],
        ['Usable pixels', roi.data_pixel_count ?? '—'],
        ['Cloud', formatFraction(qa.cloud_fraction)],
        ['Nodata', formatFraction(qa.nodata_fraction)],
        ...(Number(qa.cirrus_fraction) > 0 ? [['Cirrus', formatFraction(qa.cirrus_fraction)]] : [])
    ];
    roiSummary.innerHTML = cells.map(([key, value]) => `
        <div class="health-cell">
            <div class="health-k">${escapeHtml(key)}</div>
            <div class="health-v" title="${escapeHtml(value)}">${escapeHtml(value)}</div>
        </div>
    `).join('');

    const sr = data.products?.ortho_sr;
    const badBandCount = Array.isArray(sr?.good_wavelengths)
        ? sr.good_wavelengths.filter(value => value === false).length
        : 0;
    const chips = [
        ...(badBandCount ? [{ text: `${badBandCount} bad SR bands removed`, bad: false }] : []),
        ...(Number(qa.cloud_fraction) > 0 ? [{ text: `${formatFraction(qa.cloud_fraction)} cloud`, bad: true }] : []),
        ...(Number(qa.cirrus_fraction) > 0 ? [{ text: `${formatFraction(qa.cirrus_fraction)} cirrus`, bad: true }] : []),
        ...(Number(qa.nodata_fraction) > 0 ? [{ text: `${formatFraction(qa.nodata_fraction)} nodata`, bad: true }] : [])
    ];
    spectrumQa.innerHTML = chips.map(chip =>
        `<span class="qa-chip ${chip.bad ? 'bad' : ''}">${escapeHtml(chip.text)}</span>`
    ).join('');
}

function isCoastalScene(row) {
    return Boolean(row?.collections?.includes('coastal-water-bodies'));
}

function clearCoastalMapOverlay() {
    coastalOverlayRequestId++;
    if (currentCoastalOverlay) {
        map.removeLayer(currentCoastalOverlay);
        currentCoastalOverlay = null;
    }
    if (currentCoastalOverlayUrl) {
        URL.revokeObjectURL(currentCoastalOverlayUrl);
        currentCoastalOverlayUrl = null;
    }
    currentCoastalOverlayKey = null;
    coastalMapLegend.hidden = true;
    coastalMapLegend.innerHTML = '';
    coastalResults.querySelectorAll('[data-coastal-overlay]').forEach(button => {
        button.classList.remove('active');
        button.setAttribute('aria-pressed', 'false');
        const label = button.querySelector('span');
        if (label) label.textContent = 'Overlay';
    });
}

function syncCoastalAnalysis(row=null) {
    const eligible = isCoastalScene(row);
    coastalRequestId++;
    clearCoastalMapOverlay();
    coastalAnalysisPanel.hidden = !eligible;
    coastalAnalysisPanel.open = false;
    coastalStatus.textContent = '';
    coastalStatus.className = 'coastal-status';
    coastalResults.innerHTML = '';
    coastalWorkflowButtons.forEach(button => {
        button.disabled = !eligible;
        button.classList.remove('active');
    });
}

function formatCoastalValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    const magnitude = Math.abs(number);
    const digits = magnitude >= 100 ? 1 : magnitude >= 10 ? 2 : magnitude >= 1 ? 3 : 4;
    return number.toFixed(digits).replace(/\.?0+$/, '');
}

function coastalLegendColors(product) {
    const colors = product?.legend?.colors || COASTAL_PALETTES[product?.key] || [];
    return colors.filter(color => /^#[0-9a-f]{6}$/i.test(String(color)));
}

function coastalLegendHtml(product, placement='card', georeferencing=null) {
    const colors = coastalLegendColors(product);
    const range = (product?.range || []).map(Number);
    const fallbackTicks = range.length === 2
        ? [range[0], (range[0] + range[1]) / 2, range[1]]
        : [];
    const ticks = (product?.legend?.ticks || fallbackTicks).map(Number);
    if (colors.length < 2 || ticks.length !== 3 || ticks.some(value => !Number.isFinite(value))) return '';
    const units = product.units && product.units !== 'unitless' ? product.units : '';
    const values = ticks.map(formatCoastalValue);
    const gradient = `linear-gradient(90deg, ${colors.join(', ')})`;
    const title = placement === 'map' ? `<strong>${escapeHtml(product.label)}</strong>` : '';
    const unitAlreadyNamed = units && String(product.label || '').toLowerCase().includes(units.toLowerCase());
    const unitLabel = units && !unitAlreadyNamed && product?.key !== 'relative_cdom'
        ? `<span class="coastal-legend-unit">${escapeHtml(units)}</span>`
        : '';
    const stretch = product?.legend?.stretch || (product?.key === 'turbidity_fnu' ? '0 to 98th percentile' : '2nd to 98th percentile');
    const registration = placement === 'map'
        ? `<span class="coastal-legend-registration">${georeferencing?.epsg ? `HDF grid · EPSG:${escapeHtml(georeferencing.epsg)}` : 'STAC ortho extent'}</span>`
        : '';
    return `
        <div class="coastal-legend coastal-legend-${placement}" aria-label="${escapeHtml(product.label)} colour scale">
            <div class="coastal-legend-heading">
                <span class="coastal-legend-title">${title}${unitLabel}</span>
                <span>${escapeHtml(stretch)}</span>
            </div>
            <div class="coastal-legend-ramp" style="background:${gradient}"></div>
            <div class="coastal-legend-ticks"><span>${escapeHtml(values[0])}</span><span>${escapeHtml(values[1])}</span><span>${escapeHtml(values[2])}</span></div>
            ${registration}
        </div>
    `;
}

function coastalOverlayBounds(georeferencing) {
    const values = (georeferencing?.bounds || []).map(Number);
    if ((values.length !== 4 || values.some(value => !Number.isFinite(value))) && selectedBounds) return selectedBounds;
    if (values.length !== 4 || values.some(value => !Number.isFinite(value))) throw new Error('Coastal overlay bounds are unavailable.');
    const [west, south, east, north] = values;
    if (west >= east || south >= north) throw new Error('HDF grid bounds are invalid.');
    const gridBounds = L.latLngBounds([[south, west], [north, east]]);
    if (selectedBounds) {
        const overlapWidth = Math.max(0, Math.min(east, selectedBounds.getEast()) - Math.max(west, selectedBounds.getWest()));
        const overlapHeight = Math.max(0, Math.min(north, selectedBounds.getNorth()) - Math.max(south, selectedBounds.getSouth()));
        const gridArea = (east - west) * (north - south);
        const sceneArea = (selectedBounds.getEast() - selectedBounds.getWest()) * (selectedBounds.getNorth() - selectedBounds.getSouth());
        const overlapRatio = (overlapWidth * overlapHeight) / Math.max(Number.EPSILON, Math.min(gridArea, sceneArea));
        if (overlapRatio < 0.8) throw new Error('HDF grid does not match the selected scene footprint.');
    }
    return gridBounds;
}

function renderCoastalProducts(data) {
    const productCards = (data.products || []).map(product => {
        const isFnu = product.key === 'turbidity_fnu';
        const units = product.units && product.units !== 'unitless' ? ` ${product.units}` : '';
        const metrics = isFnu
            ? [
                ['Median', `${formatCoastalValue(product.median)}${units}`],
                ['95th percentile', `${formatCoastalValue(product.p95)}${units}`]
            ]
            : [
                ['Range', `${formatCoastalValue(product.range?.[0])}–${formatCoastalValue(product.range?.[1])}`],
                ['Median', formatCoastalValue(product.median)]
            ];
        return `
            <article class="coastal-product-card">
                <header><h3>${escapeHtml(product.label)}</h3></header>
                <div class="coastal-image-shell">
                    <img src="${product.image}" alt="${escapeHtml(product.label)} water-masked map">
                    <button class="coastal-overlay-button" type="button" data-coastal-overlay="${escapeHtml(product.key)}" aria-pressed="false">
                        <svg><use href="#i-eye"></use></svg><span>Overlay</span>
                    </button>
                </div>
                ${coastalLegendHtml(product)}
                <dl>${metrics.map(([label, value]) => `
                    <div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>
                `).join('')}</dl>
                ${product.note ? `<p>${escapeHtml(product.note)}</p>` : ''}
            </article>
        `;
    }).join('');
    coastalResults.innerHTML = productCards;
    coastalStatus.textContent = '';
    coastalStatus.className = 'coastal-status ready';
    const products = new Map((data.products || []).map(product => [product.key, product]));
    coastalResults.querySelectorAll('[data-coastal-overlay]').forEach(button => {
        button.addEventListener('click', () => toggleCoastalMapOverlay(
            products.get(button.dataset.coastalOverlay),
            button,
            data.georeferencing
        ));
    });
}

async function toggleCoastalMapOverlay(product, button, georeferencing) {
    if (!product || !selectedBounds) return;
    if (currentCoastalOverlayKey === product.key && currentCoastalOverlay) {
        clearCoastalMapOverlay();
        return;
    }
    clearCoastalMapOverlay();
    const requestId = ++coastalOverlayRequestId;
    const sceneId = selectedScene?.item_id;
    button.disabled = true;
    try {
        const overlayBounds = coastalOverlayBounds(georeferencing);
        if (requestId !== coastalOverlayRequestId || selectedScene?.item_id !== sceneId) return;
        currentCoastalOverlayKey = product.key;
        currentCoastalOverlay = L.imageOverlay(product.image, overlayBounds, {
            opacity: 0.78,
            interactive: false,
            className: 'coastal-map-overlay'
        }).addTo(map);
        currentCoastalOverlay.bringToFront();
        button.classList.add('active');
        button.setAttribute('aria-pressed', 'true');
        button.querySelector('span').textContent = 'Remove';
        coastalMapLegend.innerHTML = coastalLegendHtml(product, 'map', georeferencing);
        coastalMapLegend.hidden = false;
    } catch (error) {
        coastalStatus.textContent = error.message;
        coastalStatus.className = 'coastal-status error';
    } finally {
        button.disabled = false;
    }
}

async function runCoastalAnalysis(workflow) {
    if (!selectedScene || !isCoastalScene(selectedScene)) return;
    const requestId = ++coastalRequestId;
    const sceneId = selectedScene.item_id;
    clearCoastalMapOverlay();
    coastalWorkflowButtons.forEach(button => {
        button.disabled = true;
        button.classList.toggle('active', button.dataset.coastalWorkflow === workflow);
    });
    coastalStatus.textContent = '';
    coastalStatus.className = 'coastal-status';
    coastalResults.innerHTML = `
        <div class="analysis-loading-state" role="status">
            <div><span class="loading-spinner"></span><strong>${workflow === 'fnu' ? 'Loading quantitative turbidity scene' : 'Loading coastal indicator scene'}</strong></div>
            <span class="loading-progress"><i></i></span>
            <span class="loading-skeleton-block"></span>
        </div>
    `;
    try {
        const params = new URLSearchParams({
            scene_id: sceneId,
            workflow,
            max_size: '320'
        });
        const response = await fetch(`/api/coastal?${params}`);
        const data = await response.json();
        if (requestId !== coastalRequestId || selectedScene?.item_id !== sceneId) return;
        if (!response.ok) throw new Error(data.error || `Coastal analysis failed (${response.status})`);
        renderCoastalProducts(data);
    } catch (error) {
        if (requestId !== coastalRequestId) return;
        coastalStatus.textContent = error.message;
        coastalStatus.className = 'coastal-status error';
    } finally {
        if (requestId === coastalRequestId) {
            coastalWorkflowButtons.forEach(button => { button.disabled = false; });
        }
    }
}

coastalWorkflowButtons.forEach(button => {
    button.addEventListener('click', () => runCoastalAnalysis(button.dataset.coastalWorkflow));
});

const HCMC_GHG_SCENE_ID = '20250407_035509_25_4001';

function isGhgMethaneScene(row) {
    const collections = row?.collections || [row?.collection].filter(Boolean);
    return row?.item_id === HCMC_GHG_SCENE_ID || collections.includes('GHG-plumes');
}

function hasPublishedGhgReference(row) {
    return row?.item_id === HCMC_GHG_SCENE_ID || Boolean(row?.asset_ortho_ql_ch4);
}

function clearGhgMapOverlay() {
    if (currentGhgOverlay) {
        map.removeLayer(currentGhgOverlay);
        currentGhgOverlay = null;
    }
    currentGhgOverlayKey = null;
    ghgMapLegend.hidden = true;
    ghgMapLegend.innerHTML = '';
    [ghgOverlayButton, ...document.querySelectorAll('[data-ghg-overlay-side]')].forEach(button => {
        button.classList.remove('active');
        button.setAttribute('aria-pressed', 'false');
        const label = button.querySelector('span');
        if (label) label.textContent = 'Overlay';
    });
}

function setGhgLoading(loading, label='Loading methane scene') {
    ghgLoadingState.hidden = !loading;
    ghgLoadingLabel.textContent = label;
}

function syncGhgAnalysis(row=null) {
    const methaneEligible = isGhgMethaneScene(row);
    ghgRequestId++;
    clearGhgMapOverlay();
    currentGhgLayer = null;
    currentGhgData = null;
    ghgMethanePanel.hidden = !methaneEligible;
    ghgMethanePanel.open = false;
    setGhgLoading(false);
    ghgStatus.textContent = '';
    ghgStatus.className = 'ghg-status';
    ghgResult.hidden = true;
    ghgComparison.hidden = true;
    const referenceAvailable = methaneEligible && hasPublishedGhgReference(row);
    ghgReferenceButton.disabled = !referenceAvailable;
    ghgLayerButtons.forEach(button => {
        if (button.dataset.ghgLayer !== 'comparison') button.disabled = !methaneEligible;
        button.classList.remove('active');
    });
}

function formatGhgNumber(value, digits=2) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function ghgLegendHtml(product) {
    const range = product?.range || [];
    const palette = product?.palette || ['#111', '#eee'];
    const gradient = `linear-gradient(90deg, ${palette.join(', ')})`;
    return `
        <div class="ghg-legend-ramp" style="background:${gradient}"></div>
        <div class="ghg-legend-labels"><span>${escapeHtml(formatGhgNumber(range[0]))}</span><span>${escapeHtml(product.units || '')}</span><span>${escapeHtml(formatGhgNumber(range[1]))}</span></div>
    `;
}

function ghgMetricRows(data) {
    const product = data.product || {};
    const metrics = product.metrics || {};
    if (data.layer === 'reference') {
        const reviewed = metrics.reviewed_comparison?.cwmf_inside_cm_b;
        return [
            ...(reviewed?.correlation != null ? [
                ['CWMF correlation', `r = ${formatGhgNumber(reviewed.correlation, 3)}`],
                ['Compared pixels', formatGhgNumber(reviewed.n, 0)],
                ['CWMF p95', `${formatGhgNumber(reviewed.ours_p95_ppm_m)} ppm·m`],
                ['Reference p95', `${formatGhgNumber(reviewed.reference_p95_ppm_m)} ppm·m`]
            ] : [])
        ];
    }
    const rows = [
        ['Median', `${formatGhgNumber(metrics.median_ppm_m)} ppm·m`],
        ['95th percentile', `${formatGhgNumber(metrics.p95_ppm_m)} ppm·m`]
    ];
    if (metrics.peak_significance_sigma != null) rows.push(['Peak significance', `${formatGhgNumber(metrics.peak_significance_sigma)}σ`]);
    if (metrics.median_robust_noise_ppm_m != null) rows.push(['Robust noise', `${formatGhgNumber(metrics.median_robust_noise_ppm_m)} ppm·m`]);
    return rows;
}

function renderGhgResult(data) {
    currentGhgData = data;
    currentGhgLayer = data.layer;
    const product = data.product;
    setGhgLoading(false);
    ghgResult.hidden = false;
    ghgResultTitle.textContent = product.label;
    ghgResultImage.src = product.image;
    ghgResultImage.alt = `${product.label} for ${data.scene_id}`;
    ghgResultLegend.innerHTML = ghgLegendHtml(product);
    ghgResultMetrics.innerHTML = ghgMetricRows(data).map(([label, value]) => `
        <div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>
    `).join('');
    ghgComparison.hidden = true;
    ghgLayerButtons.forEach(button => button.classList.toggle('active', button.dataset.ghgLayer === data.layer));
    ghgStatus.textContent = '';
    ghgStatus.className = 'ghg-status ready';
}

async function fetchGhgLayer(layer) {
    const sceneId = selectedScene?.item_id;
    if (!sceneId) throw new Error('Select a scene first.');
    const cacheKey = `${sceneId}:${layer}`;
    if (ghgLayerCache.has(cacheKey)) return ghgLayerCache.get(cacheKey);
    const params = new URLSearchParams({ scene_id: sceneId, workflow: 'methane', layer, max_size: '480' });
    const response = await fetch(`/api/ghg?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Methane analysis failed (${response.status})`);
    ghgLayerCache.set(cacheKey, data);
    return data;
}

async function runGhgLayer(layer) {
    if (!selectedScene || !isGhgMethaneScene(selectedScene)) return;
    if (layer === 'comparison') {
        await showGhgComparison();
        return;
    }
    const requestId = ++ghgRequestId;
    const sceneId = selectedScene.item_id;
    clearGhgMapOverlay();
    ghgResult.hidden = true;
    ghgComparison.hidden = true;
    ghgLayerButtons.forEach(button => {
        button.disabled = true;
        button.classList.toggle('active', button.dataset.ghgLayer === layer);
    });
    ghgStatus.textContent = layer === 'artifact' ? 'Suppressing scene artifacts…' : 'Calculating methane response…';
    ghgStatus.className = 'ghg-status loading';
    setGhgLoading(true, layer === 'artifact' ? 'Loading artifact-suppressed scene' : 'Loading CWMF methane scene');
    try {
        const data = await fetchGhgLayer(layer);
        if (requestId !== ghgRequestId || selectedScene?.item_id !== sceneId) return;
        renderGhgResult(data);
    } catch (error) {
        if (requestId !== ghgRequestId) return;
        setGhgLoading(false);
        ghgStatus.textContent = error.message;
        ghgStatus.className = 'ghg-status error';
    } finally {
        if (requestId === ghgRequestId) {
            ghgLayerButtons.forEach(button => {
                button.disabled = button.dataset.ghgLayer === 'comparison' && !hasPublishedGhgReference(selectedScene);
            });
        }
    }
}

function toggleGhgOverlay(data, button) {
    const product = data?.product;
    if (!product?.image || !product?.bounds) return;
    const overlayKey = `${data.scene_id}:${data.layer}`;
    if (currentGhgOverlay && currentGhgOverlayKey === overlayKey) {
        clearGhgMapOverlay();
        return;
    }
    clearCoastalMapOverlay();
    const [west, south, east, north] = product.bounds;
    currentGhgOverlay = L.imageOverlay(product.image, [[south, west], [north, east]], {
        opacity: 0.8,
        interactive: false,
        className: 'ghg-map-overlay'
    }).addTo(map);
    currentGhgOverlayKey = overlayKey;
    currentGhgOverlay.bringToFront();
    button.classList.add('active');
    button.setAttribute('aria-pressed', 'true');
    const label = button.querySelector('span');
    if (label) label.textContent = 'Remove';
    ghgMapLegend.innerHTML = `<strong>${escapeHtml(product.label)}</strong>${ghgLegendHtml(product)}`;
    ghgMapLegend.hidden = false;
}

async function showGhgComparison() {
    if (!selectedScene || !hasPublishedGhgReference(selectedScene)) return;
    const requestId = ++ghgRequestId;
    clearGhgMapOverlay();
    ghgResult.hidden = true;
    ghgComparison.hidden = true;
    ghgLayerButtons.forEach(button => {
        button.disabled = true;
        button.classList.toggle('active', button.dataset.ghgLayer === 'comparison');
    });
    ghgStatus.textContent = 'Preparing matched-scene comparison…';
    ghgStatus.className = 'ghg-status loading';
    setGhgLoading(true, 'Loading CWMF and published reference');
    try {
        const [cwmf, reference] = await Promise.all([fetchGhgLayer('cwmf'), fetchGhgLayer('reference')]);
        if (requestId !== ghgRequestId || selectedScene?.item_id !== cwmf.scene_id) return;
        ghgCompareCwmf.src = cwmf.product.image;
        ghgCompareReference.src = reference.product.image;
        ghgComparisonLabel.textContent = reference.product.label;
        ghgComparisonMetrics.innerHTML = ghgMetricRows(reference).map(([label, value]) => `
            <div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>
        `).join('');
        setGhgLoading(false);
        ghgComparison.hidden = false;
        ghgStatus.textContent = '';
        ghgStatus.className = 'ghg-status ready';
        ghgComparison.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } catch (error) {
        if (requestId === ghgRequestId) {
            setGhgLoading(false);
            ghgStatus.textContent = error.message;
            ghgStatus.className = 'ghg-status error';
        }
    } finally {
        ghgLayerButtons.forEach(button => {
            button.disabled = button.dataset.ghgLayer === 'comparison' && !hasPublishedGhgReference(selectedScene);
        });
    }
}

function openFigureImage(title, imageUrl) {
    if (!imageUrl || !figureDialog) return;
    figureDialogTitle.textContent = title;
    figureDialogImage.src = imageUrl;
    figureDialogImage.hidden = false;
    if (!figureDialog.open) figureDialog.showModal();
}

function closeFigureDialog() {
    if (!figureDialog?.open) return;
    figureDialog.close();
}

ghgLayerButtons.forEach(button => button.addEventListener('click', () => runGhgLayer(button.dataset.ghgLayer)));
ghgOverlayButton.addEventListener('click', () => toggleGhgOverlay(currentGhgData, ghgOverlayButton));
ghgResultImage.addEventListener('click', () => openFigureImage(currentGhgData?.product?.label || 'Methane result', currentGhgData?.product?.image));
document.getElementById('ghg-expand-image').addEventListener('click', () => openFigureImage(currentGhgData?.product?.label || 'Methane result', currentGhgData?.product?.image));
document.querySelectorAll('[data-ghg-expand-side]').forEach(button => button.addEventListener('click', () => {
    const side = button.dataset.ghgExpandSide;
    const data = ghgLayerCache.get(`${selectedScene?.item_id}:${side}`);
    openFigureImage(data?.product?.label || 'Methane comparison', data?.product?.image);
}));
document.querySelectorAll('[data-ghg-overlay-side]').forEach(button => button.addEventListener('click', () => {
    const side = button.dataset.ghgOverlaySide;
    const data = ghgLayerCache.get(`${selectedScene?.item_id}:${side}`);
    toggleGhgOverlay(data, button);
}));
document.getElementById('figure-dialog-close').addEventListener('click', closeFigureDialog);
figureDialog.addEventListener('click', event => { if (event.target === figureDialog) closeFigureDialog(); });

async function sampleRoi(geometry) {
    if (!selectedScene || !geometry) return;
    const comparingAreas = compareMode === 'area';
    const completedShape = roiShapeKind;
    openInspectorSection('section-sampling');
    const requestId = ++roiRequestId;
    spectrumRequestId++;
    if (!comparingAreas) {
        compareSamples = [];
        sampleMarkers.clearLayers();
        compareAreaLayers.clearLayers();
    }
    clearActiveSampleMarker();
    renderCompareList();
    setSpectrumStatus('Loading the area median and average…');
    setLinkState('busy', 'FETCHING AREA SPECTRA · BOUNDED RANGED READS');
    spectrumQa.innerHTML = '';
    roiSummary.innerHTML = '';
    coverageCount.textContent = '';
    coverageList.innerHTML = '<div class="grid-note">Checking catalogue coverage…</div>';

    try {
        const response = await fetch('/api/roi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scene_id: selectedScene.item_id,
                geometry,
                products: ['ortho_radiance', 'ortho_sr']
            })
        });
        const data = await response.json();
        if (requestId !== roiRequestId) return;
        if (!response.ok) throw new Error(data.error || `Area request failed (${response.status})`);

        lastSpectrumData = data;
        lastSampleLatLng = L.latLng(data.clicked.lat, data.clicked.lon);
        renderRoiResult(data);
        const product = data.products?.ortho_sr?.available
            ? data.products.ortho_sr
            : Object.values(data.products || {}).find(item => item?.available);
        const selectedPixels = product?.roi?.selected_pixel_count ?? 0;
        const analysedPixels = product?.roi?.spectral_pixel_count ?? selectedPixels;
        if (comparingAreas) {
            const color = COMPARE_COLORS[compareSamples.length % COMPARE_COLORS.length];
            compareSamples.push({
                label: `A${compareSamples.length + 1}`,
                color,
                lat: data.clicked.lat,
                lon: data.clicked.lon,
                pixelCount: selectedPixels,
                geometry: JSON.parse(JSON.stringify(geometry)),
                shape: completedShape,
                data,
            });
            if (compareSamples.length > 2) compareSamples.shift();
            compareSamples.forEach((sample, index) => sample.label = `A${index + 1}`);
            renderCompareAreas();
            renderCompareList();
        }
        redrawSpectra();
        updateWlUI();
        renderIndices(data);
        renderBandReadout(data);
        renderCoveringScenes();
        const samplingNote = analysedPixels < selectedPixels
            ? ` (${analysedPixels.toLocaleString()} used for spectral statistics)`
            : '';
        if (comparingAreas && compareSamples.length < 2) {
            if (roiLayer) map.removeLayer(roiLayer);
            roiLayer = null;
            roiGeometry = geometry;
            roiShapeKind = completedShape;
            if (completedShape && completedShape !== 'custom') {
                roiPlacementShape = completedShape;
                setActiveRoiShape(completedShape);
            }
            setModeState();
            setSpectrumStatus('Area A1 loaded. Select the second area.', 'ready');
        } else if (comparingAreas) {
            if (roiLayer) map.removeLayer(roiLayer);
            roiLayer = null;
            roiGeometry = geometry;
            roiShapeKind = completedShape;
            activeMapTool = 'browse';
            setModeState();
            setSpectrumStatus('Area comparison ready.', 'ready');
        } else {
            setSpectrumStatus(`Area spectrum loaded from ${selectedPixels.toLocaleString()} selected pixels${samplingNote}.`, 'ready');
        }
        setLinkState('ok', `AREA SPECTRUM OK · ${selectedPixels.toLocaleString()} PIXELS`);
    } catch (error) {
        if (requestId !== roiRequestId) return;
        lastSpectrumData = null;
        renderUnavailable('spectrum-plot', SPECTRUM_PRODUCTS[spectrumProduct.value]?.label || 'Spectrum', error.message);
        setSpectrumStatus(error.message, 'error');
        setLinkState('error', 'AREA SPECTRUM REQUEST FAILED');
    }
}

async function sampleSpectrum(latlng) {
    if (!selectedScene) return;
    openInspectorSection('section-sampling');
    lastSampleLatLng = latlng;
    const requestId = ++spectrumRequestId;
    if (compareMode !== 'point') {
        compareSamples = [];
        sampleMarkers.clearLayers();
        setActiveSampleMarker(latlng, null, true);
    }
    setSpectrumStatus(`Loading spectra at ${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}…`);
    setLinkState('busy', 'FETCHING SPECTRUM · RANGED READS VIA KERCHUNK');
    spectrumQa.innerHTML = '';
    roiSummary.innerHTML = '';
    coverageCount.textContent = '';
    coverageList.innerHTML = '<div class="grid-note">Checking catalogue coverage…</div>';

    const params = new URLSearchParams({
        scene_id: selectedScene.item_id,
        lat: latlng.lat,
        lon: latlng.lng,
        radius: sampleRadius,
        products: 'ortho_radiance,ortho_sr'
    });

    try {
        const response = await fetch(`/api/spectrum?${params.toString()}`);
        const data = await response.json();
        if (requestId !== spectrumRequestId) return;
        if (!response.ok) throw new Error(data.error || `Spectrum request failed (${response.status})`);

        lastSpectrumData = data;
        if (compareMode === 'point') {
            clearActiveSampleMarker();
            const color = COMPARE_COLORS[compareSamples.length % COMPARE_COLORS.length];
            compareSamples.push({
                label: `P${compareSamples.length + 1}`,
                color,
                lat: data.clicked.lat,
                lon: data.clicked.lon,
                radius: sampleRadius,
                data
            });
            if (compareSamples.length > COMPARE_COLORS.length) compareSamples.shift();
            compareSamples.forEach((sample, index) => sample.label = `P${index + 1}`);
            renderCompareMarkers();
        } else {
            compareSamples = [];
            sampleMarkers.clearLayers();
            setActiveSampleMarker(latlng, data, false);
        }
        updateUndoSampleButton();
        renderCompareList();
        renderQa(data);
        redrawSpectra();
        updateWlUI();
        renderIndices(data);
        renderBandReadout(data);
        renderCoveringScenes();
        setSpectrumStatus(`Spectrum loaded at ${data.clicked.lat.toFixed(5)}, ${data.clicked.lon.toFixed(5)}.`, 'ready');
        const source = Object.values(data.products || {}).find(p => p?.available)?.source || 'unknown';
        setLinkState('ok', `SPECTRUM OK · SOURCE ${source.toUpperCase()}`);
    } catch (error) {
        if (requestId !== spectrumRequestId) return;
        renderUnavailable('spectrum-plot', SPECTRUM_PRODUCTS[spectrumProduct.value]?.label || 'Spectrum', error.message);
        setSpectrumStatus(error.message, 'error');
        setLinkState('error', 'SPECTRUM REQUEST FAILED');
    }
}

/* ── Show scene ─────────────────────────────────────────────────────── */
function parseBbox(row) {
    try {
        const bbox = JSON.parse(row.bbox);
        return Array.isArray(bbox) && bbox.length >= 4 ? bbox : null;
    } catch {
        return null;
    }
}

function currentAnalysisExtent() {
    if (roiGeometry?.type === 'Polygon') {
        const ring = roiGeometry.coordinates?.[0] || [];
        if (!ring.length) return null;
        const lons = ring.map(coordinate => Number(coordinate[0]));
        const lats = ring.map(coordinate => Number(coordinate[1]));
        return {
            west: Math.min(...lons),
            south: Math.min(...lats),
            east: Math.max(...lons),
            north: Math.max(...lats),
            kind: 'area'
        };
    }
    if (lastSampleLatLng) {
        return {
            west: lastSampleLatLng.lng,
            south: lastSampleLatLng.lat,
            east: lastSampleLatLng.lng,
            north: lastSampleLatLng.lat,
            kind: 'point'
        };
    }
    return null;
}

function bboxCoversExtent(bbox, extent) {
    return bbox &&
        bbox[0] <= extent.west &&
        bbox[1] <= extent.south &&
        bbox[2] >= extent.east &&
        bbox[3] >= extent.north;
}

function renderCoveringScenes() {
    const extent = currentAnalysisExtent();
    if (!extent) {
        coverageCount.textContent = '';
        coverageList.innerHTML = '<div class="grid-note">Sample a point or area to find other catalogue dates.</div>';
        return;
    }
    const matches = SCENES
        .filter(row => bboxCoversExtent(parseBbox(row), extent))
        .sort((a, b) => new Date(b.datetime || 0) - new Date(a.datetime || 0));
    coverageCount.textContent = `${matches.length} bbox match${matches.length === 1 ? '' : 'es'}`;
    if (!matches.length) {
        coverageList.innerHTML = '<div class="grid-note">No other catalogue bounding boxes fully cover this sample.</div>';
        return;
    }
    coverageList.innerHTML = matches.map(row => {
        const current = row.item_id === selectedScene?.item_id;
        const date = row.datetime ? row.datetime.slice(0, 10) : 'Unknown date';
        const mode = (row.collection_mode || 'unknown mode').replaceAll('_', ' ');
        return `
            <button class="coverage-row ${current ? 'current' : ''}" type="button"
                    data-coverage-scene="${escapeHtml(row.item_id)}" ${current ? 'disabled' : ''}>
                <span>
                    <span class="coverage-date">${escapeHtml(date)}${current ? ' · current' : ''}</span>
                    <span class="coverage-meta">${escapeHtml(row.collection)} · ${escapeHtml(mode)} · sun ${escapeHtml(row.sun_elevation)}° · off ${escapeHtml(row.off_nadir)}°</span>
                </span>
                <span class="coverage-cloud">cloud ${escapeHtml(row.cloud_percent)}%</span>
            </button>
        `;
    }).join('');
    coverageList.querySelectorAll('[data-coverage-scene]:not([disabled])').forEach(button => {
        button.addEventListener('click', () => switchAnalysisScene(button.dataset.coverageScene));
    });
}

function switchAnalysisScene(sceneId) {
    const row = SCENES.find(scene => scene.item_id === sceneId);
    if (!row) return;
    const savedGeometry = roiGeometry ? JSON.parse(JSON.stringify(roiGeometry)) : null;
    const savedShapeKind = roiShapeKind;
    const savedPoint = !savedGeometry && lastSampleLatLng
        ? L.latLng(lastSampleLatLng.lat, lastSampleLatLng.lng)
        : null;
    showScene(row, COLORS[row.collection] || '#888');
    if (savedGeometry) {
        roiGeometry = savedGeometry;
        roiShapeKind = savedShapeKind || 'custom';
        const latlngs = savedGeometry.coordinates[0].map(([lon, lat]) => [lat, lon]);
        roiLayer = L.polygon(latlngs, ROI_STYLE).addTo(map);
        setActiveRoiShape(roiShapeKind);
        sampleRoi(savedGeometry);
    } else if (savedPoint) {
        sampleSpectrum(savedPoint);
    }
}

function dayOfYear(dateStr) {
    const d = new Date(dateStr);
    if (Number.isNaN(d.getTime())) return null;
    const start = Date.UTC(d.getUTCFullYear(), 0, 0);
    return Math.floor((d.getTime() - start) / 86400000);
}

function syncSpectrumProductAvailability(row) {
    const availability = {
        basic_radiance: false,
        ortho_radiance: productPresent(row, 'ortho_radiance_hdf5'),
        basic_sr: false,
        ortho_sr: productPresent(row, 'ortho_sr_hdf5'),
    };
    Array.from(spectrumProduct.options).forEach(option => {
        option.disabled = !availability[option.value];
    });
    if (!availability[spectrumProduct.value]) {
        spectrumProduct.value = availability.ortho_sr ? 'ortho_sr' : 'ortho_radiance';
    }
}

function showScene(row, color) {
    closePanel();

    const bbox = parseBbox(row);
    if (!bbox) return;

    const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]; // [[south,west],[north,east]]
    const thumbPath = thumbnailUrl(row.item_id);
    selectedScene = row;
    document.getElementById('analysis-scene-id').textContent = row.item_id;
    syncSpectrumProductAvailability(row);
    syncCoastalAnalysis(row);
    syncGhgAnalysis(row);
    selectedBounds = L.latLngBounds(bounds);
    selectedThumbPath = thumbPath;
    activeMapTool = 'browse';
    lastSampleLatLng = null;
    clearSpectrum();
    inspectorSections.forEach(section => section.open = false);
    activateInspectorTab('overview');
    renderSceneTable(filteredRows());
    renderMapLayers(filteredRows());

    currentOverlay = L.imageOverlay(thumbPath, bounds, {
        opacity: 0.88,
        interactive: false,
        className: 'scene-overlay-img'
    }).addTo(map);

    // Marching-ants outline marks the armed sampling area
    currentBorder = L.rectangle(bounds, {
        color: color,
        weight: 2,
        fill: false,
        interactive: false,
        className: 'selected-footprint'
    }).addTo(map);

    map.flyToBounds(bounds, { padding: [70, 70], maxZoom: 11, duration: 0.6 });

    document.getElementById('sp-id').innerText = row.item_id;
    const collEl = document.getElementById('sp-coll');
    collEl.innerText = (row.collections || [row.collection]).join(' · ');
    collEl.style.color = color;
    document.getElementById('sp-quality').innerText = row.quality_category || '';

    const doy = dayOfYear(row.datetime);
    document.getElementById('sp-date').innerText = row.datetime
        ? new Date(row.datetime).toISOString().replace('T', ' ').slice(0, 16) + ' UTC' + (doy ? ` · DOY ${doy}` : '')
        : '-';
    const loc = row.location_description || 'Offshore / Ocean';
    const locEl = document.getElementById('sp-loc');
    locEl.innerText = loc;
    locEl.title = loc;
    document.getElementById('sp-cloud').innerText = row.cloud_percent + '%';
    document.getElementById('sp-sun').innerText = row.sun_elevation + '°';
    document.getElementById('sp-gsd').innerText = row.gsd + ' m';
    renderSceneHealth(row);
    renderReviewButtons();
    renderOverlayControls(row, bounds);

    const downloadIcon = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
    const linksEl = document.getElementById('sp-links');
    linksEl.innerHTML = '';
    if (row.asset_ortho_visual) {
        linksEl.innerHTML += `<a class="sp-link" href="${row.asset_ortho_visual}" target="_blank">${downloadIcon} RGB TIF</a>`;
    }
    const srUrl = row.asset_ortho_sr_hdf5 || row.asset_basic_sr_hdf5;
    if (srUrl) {
        linksEl.innerHTML += `<a class="sp-link" href="${srUrl}" target="_blank">${downloadIcon} HDF5 SR</a>`;
    }
    if (row.asset_ortho_radiance_hdf5) {
        linksEl.innerHTML += `<a class="sp-link" href="${row.asset_ortho_radiance_hdf5}" target="_blank">${downloadIcon} HDF5 RAD</a>`;
    }

    inspector.classList.add('open');
    setWorkspaceTab('analysis', { promptForScene: false });
    setModeState();
    queueMapResize();
    updateUrlState(row.item_id);
}

/* ── Filtering ──────────────────────────────────────────────────────── */
function normalizedSearchTokens() {
    return searchInput.value
        .trim()
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean);
}

function rowMatchesSearch(row, tokens) {
    if (!tokens.length) return true;
    const haystack = [
        row.item_id,
        row.location_description || ''
    ].join(' ').toLowerCase();
    return tokens.every(token => haystack.includes(token));
}

function monthKey(row) {
    return (row.datetime || '').slice(0, 7);
}

function baseFilteredRows() {
    const maxCloud = parseInt(cloudSlider.value);
    const minSun = Number(sunSlider.value);
    const maxOffNadir = Number(offNadirSlider.value);
    const maxHaze = Number(hazeSlider.value);
    const dateFrom = dateFromInput.value;
    const dateTo = dateToInput.value;
    const mode = modeSelect.value;
    const tokens = normalizedSearchTokens();
    return SCENES.filter(row =>
        row.collections.some(name => activeColls.has(name)) &&
        Number(row.cloud_percent ?? 0) <= maxCloud &&
        Number(row.sun_elevation ?? -Infinity) >= minSun &&
        Number(row.off_nadir ?? Infinity) <= maxOffNadir &&
        Number(row.light_haze_percent ?? Infinity) <= maxHaze &&
        (!dateFrom || (row.datetime || '').slice(0, 10) >= dateFrom) &&
        (!dateTo || (row.datetime || '').slice(0, 10) <= dateTo) &&
        (!mode || row.collection_mode === mode) &&
        rowMatchesSearch(row, tokens)
    );
}

function sortRows(rows) {
    return rows.sort((a, b) => {
        if (sceneSort === 'cloud') return Number(a.cloud_percent || 0) - Number(b.cloud_percent || 0);
        return new Date(b.datetime || 0) - new Date(a.datetime || 0);
    });
}

function filteredRows() {
    let rows = baseFilteredRows();
    if (timelineMonth) rows = rows.filter(row => monthKey(row) === timelineMonth);
    return sortRows(rows);
}

/* ── URL state ──────────────────────────────────────────────────────── */
function updateUrlState(sceneId=null) {
    const url = new URL(window.location.href);
    if (searchInput.value) url.searchParams.set('q', searchInput.value);
    else url.searchParams.delete('q');
    url.searchParams.set('cloud', cloudSlider.value);
    if (dateFromInput.value) url.searchParams.set('from', dateFromInput.value);
    else url.searchParams.delete('from');
    if (dateToInput.value) url.searchParams.set('to', dateToInput.value);
    else url.searchParams.delete('to');
    if (modeSelect.value) url.searchParams.set('mode', modeSelect.value);
    else url.searchParams.delete('mode');
    url.searchParams.set('sun', sunSlider.value);
    url.searchParams.set('off_nadir', offNadirSlider.value);
    url.searchParams.set('haze', hazeSlider.value);
    url.searchParams.set('sort', sceneSort);
    if (timelineMonth) url.searchParams.set('month', timelineMonth);
    else url.searchParams.delete('month');
    if (sceneId) url.searchParams.set('scene', sceneId);
    window.history.replaceState(null, '', url);
}

function applyUrlState() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q');
    const cloud = params.get('cloud');
    const dateFrom = params.get('from');
    const dateTo = params.get('to');
    const mode = params.get('mode');
    const sun = params.get('sun');
    const offNadir = params.get('off_nadir');
    const haze = params.get('haze');
    const sort = params.get('sort');
    const month = params.get('month');
    if (q !== null) searchInput.value = q;
    if (cloud !== null && !Number.isNaN(Number(cloud))) {
        cloudSlider.value = Math.max(0, Math.min(100, Number(cloud)));
        cloudLabel.innerText = cloudSlider.value + '%';
    }
    if (dateFrom && /^\d{4}-\d{2}-\d{2}$/.test(dateFrom)) dateFromInput.value = dateFrom;
    if (dateTo && /^\d{4}-\d{2}-\d{2}$/.test(dateTo)) dateToInput.value = dateTo;
    if ([...modeSelect.options].some(option => option.value === mode)) modeSelect.value = mode || '';
    if (sun !== null && !Number.isNaN(Number(sun))) {
        sunSlider.value = Math.max(0, Math.min(80, Number(sun)));
        sunLabel.innerText = sunSlider.value + '°';
    }
    if (offNadir !== null && !Number.isNaN(Number(offNadir))) {
        offNadirSlider.value = Math.max(0, Math.min(31, Number(offNadir)));
        offNadirLabel.innerText = offNadirSlider.value + '°';
    }
    if (haze !== null && !Number.isNaN(Number(haze))) {
        hazeSlider.value = Math.max(0, Math.min(100, Number(haze)));
        hazeLabel.innerText = hazeSlider.value + '%';
    }
    if (sort === 'cloud' || sort === 'date') sceneSort = sort;
    if (month && /^\d{4}-\d{2}$/.test(month)) timelineMonth = month;
}

/* ── Timeline strip ─────────────────────────────────────────────────── */
function monthLabel(key) {
    const [y, m] = key.split('-').map(Number);
    return new Date(Date.UTC(y, m - 1, 1)).toLocaleDateString('en-GB', { month: 'short', year: '2-digit', timeZone: 'UTC' });
}

function nextMonthKey(key) {
    let [y, m] = key.split('-').map(Number);
    m += 1;
    if (m > 12) { m = 1; y += 1; }
    return `${y}-${String(m).padStart(2, '0')}`;
}

function renderTimeline() {
    // Histogram over the month-agnostic filter result, so a selected month
    // stays in context and other months remain clickable.
    const rows = baseFilteredRows();
    const counts = {};
    rows.forEach(row => {
        const key = monthKey(row);
        if (/^\d{4}-\d{2}$/.test(key)) counts[key] = (counts[key] || 0) + 1;
    });
    const keys = Object.keys(counts).sort();
    if (!keys.length) {
        tlBars.innerHTML = '';
        tlAxis.innerHTML = '';
        tlNote.innerHTML = 'no scenes in range';
        return;
    }

    // Continuous month axis from first to last acquisition
    const months = [];
    for (let key = keys[0]; key <= keys[keys.length - 1]; key = nextMonthKey(key)) {
        months.push(key);
        if (months.length > 240) break; // safety
    }
    const maxCount = Math.max(...months.map(key => counts[key] || 0), 1);

    tlBars.innerHTML = months.map(key => {
        const count = counts[key] || 0;
        const height = count ? Math.max(3, Math.round((count / maxCount) * 28)) : 2;
        const classes = ['tl-bar'];
        if (!count) classes.push('empty');
        if (key === timelineMonth) classes.push('selected');
        return `<div class="${classes.join(' ')}" data-month="${key}" style="height:${height}px"
                     title="${monthLabel(key)}: ${count} ${count === 1 ? 'scene' : 'scenes'}"></div>`;
    }).join('');

    tlAxis.innerHTML = `<span>${monthLabel(months[0])}</span><span>${monthLabel(months[months.length - 1])}</span>`;
    tlNote.innerHTML = timelineMonth
        ? `filtered to <span class="sel">${monthLabel(timelineMonth)}</span> · click bar to clear`
        : 'click a bar to filter by month';

    tlBars.querySelectorAll('.tl-bar:not(.empty)').forEach(bar => {
        bar.addEventListener('click', () => {
            timelineMonth = timelineMonth === bar.dataset.month ? null : bar.dataset.month;
            refresh();
        });
    });
}

/* ── Results list ───────────────────────────────────────────────────── */
function renderSceneTable(rows) {
    const visible = rows.slice(0, 80);
    // Keep the selected scene visible even when it falls outside the slice
    if (selectedScene && !visible.some(row => row.item_id === selectedScene.item_id)) {
        visible.unshift(selectedScene);
    }
    sceneTable.innerHTML = visible.map(row => {
        const color = COLORS[row.collection] || '#888';
        const state = reviewFor(row.item_id);
        const date = row.datetime ? new Date(row.datetime).toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'2-digit' }) : '-';
        const location = row.location_description || 'Offshore / Ocean';
        const active = selectedScene?.item_id === row.item_id;
        const thumb = thumbnailUrl(row.item_id);
        return `
            <div class="scene-table-row ${active ? 'active' : ''}">
                <button class="scene-row-main" type="button" data-scene="${escapeHtml(row.item_id)}">
                    <img class="scene-thumb" src="${thumb}" alt="" loading="lazy"
                         onerror="this.classList.add('missing')">
                    <span>
                        <span class="scene-row-id" style="color:${color}">${escapeHtml(row.item_id)}</span>
                        <span class="scene-row-meta">
                            <span class="mini-chip">${date}</span>
                            <span class="mini-chip">cloud ${escapeHtml(row.cloud_percent)}%</span>
                            ${(row.collections || [row.collection]).map(name => `<span class="mini-chip">${escapeHtml(name)}</span>`).join('')}
                        </span>
                        <span class="scene-row-place" title="${escapeHtml(location)}">${escapeHtml(location)}</span>
                    </span>
                </button>
                <span class="queue-tags">
                    <button class="queue-tag ${state.starred ? 'on' : ''}" type="button" data-review-scene="${escapeHtml(row.item_id)}" data-review-flag="starred" title="Toggle shortlist">S</button>
                    <button class="queue-tag ${state.reviewed ? 'on' : ''}" type="button" data-review-scene="${escapeHtml(row.item_id)}" data-review-flag="reviewed" title="Toggle reviewed">${state.reviewed ? '✓' : 'R'}</button>
                </span>
            </div>
        `;
    }).join('');
    sceneTable.querySelectorAll('[data-scene]').forEach(btn => {
        btn.addEventListener('click', () => {
            const row = SCENES.find(item => item.item_id === btn.dataset.scene);
            if (row) showScene(row, COLORS[row.collection] || '#888');
        });
    });
    sceneTable.querySelectorAll('[data-review-scene]').forEach(button => {
        button.addEventListener('click', event => {
            event.stopPropagation();
            const state = reviewFor(button.dataset.reviewScene);
            state[button.dataset.reviewFlag] = !state[button.dataset.reviewFlag];
            saveReviewState();
            renderSceneTable(filteredRows());
        });
    });
}

/* ── Map markers ────────────────────────────────────────────────────── */
function sceneTooltip(row, color) {
    return `<div class="tt">
        <div class="tt-title">${escapeHtml(row.item_id)}</div>
        <div class="tt-coll" style="color:${color}">${escapeHtml(row.collection.toUpperCase())}</div>
        <div class="tt-meta">
            <span>cloud ${escapeHtml(row.cloud_percent)}%</span>
            <span>sun ${escapeHtml(row.sun_elevation)}°</span>
        </div>
    </div>`;
}

// A marker click always refers to a specific scene. If it is the currently
// selected scene, sample a spectrum at that point; otherwise switch to the
// clicked scene — so an overlapping dot from another scene is always reachable.
function routeMarkerClick(e, row, color) {
    e.originalEvent._fromMarker = true;
    if (selectedScene && selectedScene.item_id === row.item_id && activeWorkspaceTab === 'analysis' && activeMapTool === 'point') {
        sampleSpectrum(e.latlng);
        return;
    }
    showScene(row, color);
}

function renderMapLayers(rows) {
    markers.clearLayers();

    rows.forEach(row => {
        const color = COLORS[row.collection] || '#888';
        const isSelected = selectedScene?.item_id === row.item_id;
        const radius = isSelected ? 6 : 5;

        const icon = L.divIcon({
            html: `<div class="marker-dot${isSelected ? ' selected' : ''}" style="width:${radius*2}px;height:${radius*2}px;background:${color};"></div>`,
            className: '',
            iconSize:   [radius*2, radius*2],
            iconAnchor: [radius, radius]
        });

        // riseOnHover lifts the hovered dot above any overlapping dots so it
        // is always the one that receives the click.
        const m = L.marker([row.centroid_lat, row.centroid_lon], {
            icon,
            riseOnHover: true,
            riseOffset: 400,
            zIndexOffset: isSelected ? 1000 : 0,
        });
        m.bindTooltip(sceneTooltip(row, color), { direction: 'top', offset: [0,-6] });
        m.on('click', e => routeMarkerClick(e, row, color));
        m.addTo(markers);
    });
}

/* ── Main refresh ───────────────────────────────────────────────────── */
function refresh() {
    const rows = filteredRows();
    renderSortButtons();
    renderSummary(rows);
    renderSceneTable(rows);
    renderMapLayers(rows);
    updateUrlState(selectedScene?.item_id || null);
}

async function initExplorer() {
    applyUrlState();
    setWorkspaceTab('overview', { promptForScene: false });
    if (window.innerWidth <= 760) {
        setCatalogOpen(false);
    }
    activateCatalogTab('scenes');
    activateInspectorTab('overview');
    activateDrawerTab('spectrum', false);
    setModeState();
    await loadScienceManifest();
    clearSpectrum();
    refresh();
    const sceneId = new URLSearchParams(window.location.search).get('scene');
    if (sceneId) {
        const row = SCENES.find(item => item.item_id === sceneId);
        if (row) showScene(row, COLORS[row.collection] || '#888');
    }
}

initExplorer();

/* Guided tutorial */
const tutorialOverlay = document.getElementById('tutorial-overlay');
const tutorialDialog = document.getElementById('tutorial-dialog');
const tutorialSpotlight = tutorialOverlay.querySelector('.tutorial-spotlight');
const tutorialEyebrow = document.getElementById('tutorial-eyebrow');
const tutorialTitle = document.getElementById('tutorial-title');
const tutorialBody = document.getElementById('tutorial-body');
const tutorialTip = document.getElementById('tutorial-tip');
const tutorialProgress = document.getElementById('tutorial-progress');
const tutorialStepCount = document.getElementById('tutorial-step-count');
const tutorialLauncher = document.getElementById('tutorial-launcher');
const tutorialCloseButton = document.getElementById('tutorial-close');
const tutorialBackButton = document.getElementById('tutorial-back');
const tutorialNextButton = document.getElementById('tutorial-next');
const tutorialSkipButton = document.getElementById('tutorial-skip');

const tutorialState = {
    active: false,
    index: 0,
    target: null,
    previousFocus: null,
    inertStates: [],
    resizeObserver: null,
    positionFrame: 0,
    renderToken: 0,
};

function tutorialOpenOverview({ filters=false, scenes=true }={}) {
    setWorkspaceTab('overview', { promptForScene: false });
    if (scenes) setScenesOpen(true);
    setFilterOpen(filters);
}

function tutorialEnsureScene() {
    if (selectedScene) return true;
    const row = filteredRows()[0] || SCENES[0];
    if (!row) return false;
    showScene(row, COLORS[row.collection] || '#888');
    return true;
}

function tutorialOpenAnalysis(panelName='browse') {
    if (!tutorialEnsureScene()) return;
    setWorkspaceTab('analysis', { promptForScene: false });
    inspector.classList.add('open');
    const panel = document.querySelector(`[data-analysis-panel="${panelName}"]`);
    if (panel && !panel.hidden) panel.open = true;
    queueMapResize();
}

function tutorialOpenInspectorTab(name) {
    if (!tutorialEnsureScene()) return;
    setWorkspaceTab('analysis', { promptForScene: false });
    inspector.classList.add('open');
    activateInspectorTab(name);
    queueMapResize();
}

function tutorialOpenConditionalAnalysis(id) {
    tutorialOpenAnalysis('browse');
    const panel = document.getElementById(id);
    if (panel && !panel.hidden) panel.open = true;
}

function tutorialConditionalPanel(id) {
    const panel = document.getElementById(id);
    return panel && !panel.hidden ? panel : document.getElementById('analysis-drawer');
}

const TUTORIAL_STEPS = [
    {
        eyebrow: 'Workbench orientation',
        title: 'Start with the whole workspace',
        body: 'Tanager Workbench brings the scene catalogue, map, selected scene inspector, and analysis tools into one workspace. This tour points out what each area does without running an analysis or downloading anything.',
        tip: 'You can leave at any time with Escape, Close, or Skip tutorial.',
        target: '#workspace',
        placement: 'center',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Workspaces',
        title: 'Move between Overview and Analysis',
        body: 'The Overview tab is for finding scenes and reading the map. The Analysis tab is for sampling a selected scene, inspecting spectra, composing imagery, and exporting work. Analysis needs a selected scene.',
        target: '.workspace-tabs',
        placement: 'bottom',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Catalogue search',
        title: 'Search scenes and locations',
        body: 'Search matches scene identifiers and location words. Press the slash key from anywhere outside a form field to focus this search. Press Enter in the search field to open the first matching scene.',
        tip: 'The slash shortcut is useful when your hands are already on the keyboard.',
        target: '.global-search',
        placement: 'bottom',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Catalogue status',
        title: 'Read visible and total scene counts',
        body: 'Visible is the number of scenes that match the current search and filters. Scenes is the full catalogue count. Comparing these counts shows how much the current constraints have narrowed the catalogue.',
        target: '.session-metrics',
        placement: 'bottom',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Catalogue dock',
        title: 'Open or collapse the catalogue',
        body: 'The Catalogue rail opens the dock when it is collapsed. The collapse control in the Catalogue header hides the scene list and makes more room for the map. Your filters remain in place when the dock is collapsed.',
        target: '.catalogue-header',
        placement: 'right',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Collection filters',
        title: 'Choose source collections',
        body: 'Collection checkboxes include or exclude groups of scenes. The count bars show the relative size of each collection. You can combine collections before using the other filters.',
        target: '#coll-list',
        placement: 'right',
        before: () => tutorialOpenOverview({ filters: true }),
    },
    {
        eyebrow: 'Scene filters',
        title: 'Constrain measurements, dates, and mode',
        body: 'Maximum cloud and maximum haze limit obscured scenes. Minimum sun keeps scenes with enough solar elevation. Maximum off-nadir limits the viewing angle away from straight down. From and To set the acquisition date range, and Collection mode selects an imaging mode.',
        tip: 'The live summary describes every active constraint.',
        target: '#filter-subdock',
        placement: 'right',
        before: () => tutorialOpenOverview({ filters: true }),
    },
    {
        eyebrow: 'Boundaries and reset',
        title: 'Show reference boundaries or start over',
        body: 'Reference boundaries draw the Equatorial belt, Tropical belt, and Southeast Asia on the map. Reset filters restores all collections, numeric limits, dates, collection mode, search, and scene sorting to their defaults.',
        target: '#btn-reset',
        placement: 'right',
        before: () => tutorialOpenOverview({ filters: true }),
    },
    {
        eyebrow: 'Scene results',
        title: 'Sort and scan the scene list',
        body: 'Newest sorts by acquisition date. Least cloud sorts by cloud percentage. The scene list shows each scene identifier, date, cloud amount, collection, location, and review flags so you can compare candidates quickly.',
        target: '#scene-results',
        placement: 'right',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Map',
        title: 'Read locations and change the basemap',
        body: 'Map markers locate the scenes that remain visible. The basemap control switches between Imagery and Reference backgrounds. Reference boundaries and selected scene layers are drawn above the chosen basemap.',
        target: '#map-stage',
        placement: 'left',
        before: () => {
            tutorialOpenOverview({ filters: false, scenes: window.innerWidth > 760 });
            if (window.innerWidth <= 760) setScenesOpen(false);
        },
    },
    {
        eyebrow: 'Scene selection',
        title: 'Select a scene for detailed work',
        body: 'Choose a row in the scene list or a marker on the map to select that scene. Selection opens the scene inspector, frames its footprint on the map, and makes the Analysis workspace available.',
        target: '#scene-table',
        placement: 'right',
        before: () => tutorialOpenOverview({ filters: false }),
    },
    {
        eyebrow: 'Selected scene',
        title: 'Read the scene overview record',
        body: 'Scene Overview identifies the selected scene and records its acquisition time, location, cloud cover, sun elevation, ground sampling distance, collections, quality category, and available source links.',
        target: '.scene-record-section',
        placement: 'left',
        before: () => tutorialOpenInspectorTab('overview'),
    },
    {
        eyebrow: 'Scene context',
        title: 'Check quality, layers, and related scenes',
        body: 'Scene quality reports availability and cautions before interpretation. Layers controls available map overlays. Related scenes lists other dates that cover a sampled point or area, which supports comparison through time.',
        target: '[data-ins-panel="overview"]',
        placement: 'left',
        before: () => tutorialOpenInspectorTab('overview'),
    },
    {
        eyebrow: 'Analysis tools',
        title: 'Browse the map or sample one point',
        body: 'Browse keeps map navigation active. Point sample arms a single map click and requests a spectrum for that location only when you click the map. A spectrum is a set of measured values across wavelengths.',
        target: '[data-analysis-panel="browse"]',
        placement: 'right',
        before: () => tutorialOpenAnalysis('browse'),
    },
    {
        eyebrow: 'Area sampling',
        title: 'Choose one of four area shapes',
        body: 'Square, Rectangle, and Pentagon create regular areas by pressing and dragging on the map. Custom lets you draw a freeform polygon point by point. An area sample summarizes all valid pixels inside the completed shape.',
        target: '.area-tools',
        placement: 'right',
        before: () => tutorialOpenAnalysis('browse'),
    },
    {
        eyebrow: 'Comparisons',
        title: 'Compare points or areas',
        body: 'Compare Points collects spectra from several map clicks. Compare Area collects several shaped regions. Each mode keeps the samples together in the spectrum plot so their curves can be compared.',
        target: '.compare-actions',
        placement: 'right',
        before: () => tutorialOpenAnalysis('browse'),
    },
    {
        eyebrow: 'Spectrum product',
        title: 'Choose the measurement product',
        body: 'Product chooses surface reflectance or top of atmosphere radiance when the selected scene provides it. Surface reflectance estimates the fraction of light reflected by the ground. Top of atmosphere radiance is the light measured at the sensor.',
        target: '#spectrum-product',
        placement: 'right',
        before: () => tutorialOpenAnalysis('spectrum'),
    },
    {
        eyebrow: 'Wavelength range',
        title: 'Focus the spectral window',
        body: 'The two wavelength handles set the minimum and maximum shown in the plot. Wavelength is measured in nm, meaning nanometres. Reset range restores the full visible, near infrared, and shortwave infrared span.',
        target: '#spectral-range',
        placement: 'right',
        before: () => tutorialOpenAnalysis('spectrum'),
    },
    {
        eyebrow: 'Spectrum workspace',
        title: 'Undo, clear, and read the plot',
        body: 'Undo removes the most recent point or compared area. Clear area removes the active shape. Clear spectrum removes all collected samples. The spectrum plot below draws each available product only after a real point or area result exists.',
        target: '[data-analysis-panel="spectrum"]',
        placement: 'right',
        before: () => tutorialOpenAnalysis('spectrum'),
    },
    {
        eyebrow: 'Derived indices',
        title: 'Interpret spectral indices',
        body: 'Indices shows calculated ratios such as vegetation, water, and burn indicators when a sample is available. An index combines selected wavelengths into one value that can make a material or condition easier to compare.',
        target: '[data-analysis-panel="indices"]',
        placement: 'right',
        before: () => tutorialOpenAnalysis('indices'),
    },
    {
        eyebrow: 'Band readout',
        title: 'Inspect individual bands',
        body: 'Bands lists measured values near standard wavelengths for the chosen product. Use it to connect a point on the spectrum plot with its exact band value and wavelength.',
        target: '[data-analysis-panel="bands"]',
        placement: 'right',
        before: () => tutorialOpenAnalysis('bands'),
    },
    {
        eyebrow: 'Conditional coastal tools',
        title: 'Use coastal analysis on matching scenes',
        body: 'Coastal Analysis appears only for scenes in the coastal water collection. Coastal indicators reports turbidity, colored dissolved organic matter, and NDCI chlorophyll response. Quantitative FNU estimates turbidity in Formazin Nephelometric Units. Results can also be shown as map overlays.',
        tip: 'If the selected scene is not coastal, this step points to the Analysis tools dock where the panel would appear.',
        target: () => tutorialConditionalPanel('coastal-analysis-panel'),
        placement: 'right',
        before: () => tutorialOpenConditionalAnalysis('coastal-analysis-panel'),
    },
    {
        eyebrow: 'Conditional methane tools',
        title: 'Use methane analysis on matching scenes',
        body: 'Methane Analysis appears only for supported methane scenes. CWMF is the primary methane retrieval result. The CWMF, Artifact-suppressed, and Comparison buttons switch between available result views. Overlay places a result on the map, and Comparison can show the independent result beside a published reference.',
        tip: 'If the selected scene has no methane product, this step points to the Analysis tools dock where the panel would appear.',
        target: () => tutorialConditionalPanel('ghg-methane-panel'),
        placement: 'right',
        before: () => tutorialOpenConditionalAnalysis('ghg-methane-panel'),
    },
    {
        eyebrow: 'Compose setup',
        title: 'Build a custom image recipe',
        body: 'Compose Product chooses surface reflectance or top of atmosphere radiance. Preset supplies common three-band views and calculated indices. Custom wavelengths assign exact red, green, and blue channels. Low and High percentile stretch set the brightness range while reducing the effect of extreme pixels.',
        target: '.composer-form',
        placement: 'left',
        before: () => tutorialOpenInspectorTab('compose'),
    },
    {
        eyebrow: 'Compose output',
        title: 'Render, download, or clear a composite',
        body: 'Render composite sends the current recipe only when you choose it. Download PNG saves a completed preview. Clear removes the preview and its map layer. The tutorial does not render, download, or clear anything.',
        target: '[data-ins-panel="compose"]',
        placement: 'left',
        before: () => tutorialOpenInspectorTab('compose'),
    },
    {
        eyebrow: 'Export formats',
        title: 'Save JSON, CSV, or PNG output',
        body: 'JSON report saves scene, sample, and derived metrics in structured text. CSV spectra saves wavelength values in a table. PNG snapshot and sample saves the current plot image plus a reusable GeoJSON coordinate file. GeoJSON is a text format for map geometry.',
        target: '.export-section',
        placement: 'left',
        before: () => tutorialOpenInspectorTab('export'),
    },
    {
        eyebrow: 'Restore and share',
        title: 'Import a sample or package the session',
        body: 'Load sample file restores an exported point or area without drawing it again. Copy session link captures current catalogue and scene state. Download evidence package bundles results, metadata, geometry, and a reproduction script for review.',
        target: '[data-ins-panel="export"]',
        placement: 'left',
        before: () => tutorialOpenInspectorTab('export'),
    },
    {
        eyebrow: 'Tutorial complete',
        title: 'Explore with the workflow in view',
        body: 'You now know how to find a scene, inspect its context, sample spectral data, compare results, compose imagery, and export evidence. Select Finish to return to the workbench.',
        tip: 'Choose Tutorial in the header whenever you want to start again from the first step.',
        target: null,
        placement: 'center',
        before: () => {},
    },
];

function tutorialTargetElement(step) {
    const target = typeof step.target === 'function' ? step.target() : step.target;
    if (!target) return null;
    if (target instanceof Element) return target;
    return document.querySelector(target);
}

function tutorialTargetIsVisible(element) {
    if (!element || element.hidden || element.closest('[hidden]')) return false;
    const style = window.getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
}

function tutorialNextFrame() {
    return new Promise(resolve => setTimeout(resolve, 0));
}

function tutorialPlacement(rect, preferred) {
    if (!rect || preferred === 'center') return 'center';
    const gap = 20;
    const dialogRect = tutorialDialog.getBoundingClientRect();
    const spaces = {
        top: rect.top,
        bottom: window.innerHeight - rect.bottom,
        left: rect.left,
        right: window.innerWidth - rect.right,
    };
    const needs = {
        top: dialogRect.height + gap,
        bottom: dialogRect.height + gap,
        left: dialogRect.width + gap,
        right: dialogRect.width + gap,
    };
    const mobileOrder = rect.top > window.innerHeight / 2 ? ['top', 'bottom'] : ['bottom', 'top'];
    const order = window.innerWidth <= 700
        ? mobileOrder
        : [preferred, 'right', 'left', 'bottom', 'top'];
    const uniqueOrder = [...new Set(order.filter(Boolean))];
    const fittingSide = uniqueOrder.find(side => spaces[side] >= needs[side]);
    if (fittingSide) return fittingSide;
    if (window.innerWidth <= 700) return 'center';
    return uniqueOrder.sort((a, b) => spaces[b] - spaces[a])[0] || 'center';
}

function positionTutorial() {
    tutorialState.positionFrame = 0;
    if (!tutorialState.active) return;

    const target = tutorialState.target;
    if (!tutorialTargetIsVisible(target)) {
        tutorialSpotlight.hidden = true;
        tutorialDialog.dataset.placement = 'center';
        return;
    }

    const rect = target.getBoundingClientRect();
    tutorialSpotlight.hidden = false;
    tutorialOverlay.style.setProperty('--tutorial-x', `${rect.left}px`);
    tutorialOverlay.style.setProperty('--tutorial-y', `${rect.top}px`);
    tutorialOverlay.style.setProperty('--tutorial-w', `${rect.width}px`);
    tutorialOverlay.style.setProperty('--tutorial-h', `${rect.height}px`);
    tutorialDialog.dataset.placement = tutorialPlacement(
        rect,
        TUTORIAL_STEPS[tutorialState.index]?.placement
    );
}

function scheduleTutorialPosition() {
    if (!tutorialState.active || tutorialState.positionFrame) return;
    tutorialState.positionFrame = requestAnimationFrame(positionTutorial);
}

function observeTutorialTarget(target) {
    tutorialState.resizeObserver?.disconnect();
    tutorialState.resizeObserver = null;
    if (!target || !window.ResizeObserver) return;
    tutorialState.resizeObserver = new ResizeObserver(scheduleTutorialPosition);
    tutorialState.resizeObserver.observe(target);
}

async function showTutorialStep() {
    if (!tutorialState.active) return;
    const token = ++tutorialState.renderToken;
    const step = TUTORIAL_STEPS[tutorialState.index];
    await step.before?.();
    await tutorialNextFrame();
    if (!tutorialState.active || token !== tutorialState.renderToken) return;

    let requestedTarget = tutorialTargetElement(step);
    if (step.target && !tutorialTargetIsVisible(requestedTarget)) {
        await new Promise(resolve => setTimeout(resolve, 260));
        if (!tutorialState.active || token !== tutorialState.renderToken) return;
        requestedTarget = tutorialTargetElement(step);
    }
    const target = tutorialTargetIsVisible(requestedTarget)
        ? requestedTarget
        : (step.target ? document.getElementById('workspace') : null);

    tutorialState.target?.classList.remove('tutorial-target');
    tutorialState.target = target;
    target?.classList.add('tutorial-target');
    target?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'auto' });

    tutorialEyebrow.textContent = step.eyebrow;
    tutorialTitle.textContent = step.title;
    tutorialBody.textContent = step.body;
    tutorialTip.textContent = step.tip || '';
    tutorialTip.hidden = !step.tip;

    const current = tutorialState.index + 1;
    const total = TUTORIAL_STEPS.length;
    tutorialStepCount.textContent = `Step ${current} of ${total}`;
    tutorialProgress.setAttribute('aria-valuemax', String(total));
    tutorialProgress.setAttribute('aria-valuenow', String(current));
    tutorialProgress.setAttribute('aria-valuetext', `Step ${current} of ${total}`);
    tutorialProgress.style.setProperty('--tutorial-progress', String(current / total));
    tutorialBackButton.disabled = tutorialState.index === 0;
    tutorialNextButton.textContent = tutorialState.index === total - 1 ? 'Finish' : 'Next';

    observeTutorialTarget(target);
    await tutorialNextFrame();
    if (!tutorialState.active || token !== tutorialState.renderToken) return;
    positionTutorial();
}

function tutorialFocusableElements() {
    return [...tutorialDialog.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'
    )].filter(element => !element.hidden && element.getClientRects().length);
}

function handleTutorialKeydown(event) {
    if (!tutorialState.active) return;
    if (event.key === 'Escape') {
        event.preventDefault();
        event.stopImmediatePropagation();
        endTutorial();
        return;
    }

    if (event.key === 'Tab') {
        const focusable = tutorialFocusableElements();
        if (!focusable.length) {
            event.preventDefault();
            tutorialDialog.focus({ preventScroll: true });
            return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && (document.activeElement === first || !tutorialDialog.contains(document.activeElement))) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !tutorialDialog.contains(document.activeElement))) {
            event.preventDefault();
            first.focus();
        }
        return;
    }

    const inFormField = /^(INPUT|SELECT|TEXTAREA)$/.test(event.target?.tagName || '');
    if (inFormField || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (event.key === 'ArrowLeft') {
            if (tutorialState.index > 0) {
                tutorialState.index -= 1;
                showTutorialStep();
            }
        } else if (tutorialState.index < TUTORIAL_STEPS.length - 1) {
            tutorialState.index += 1;
            showTutorialStep();
        } else {
            endTutorial();
        }
    }
}

function setTutorialBackgroundInert(inert) {
    if (inert) {
        tutorialState.inertStates = [...document.body.children]
            .filter(element => element !== tutorialOverlay)
            .map(element => [element, element.inert]);
        tutorialState.inertStates.forEach(([element]) => {
            element.inert = true;
        });
        return;
    }
    tutorialState.inertStates.forEach(([element, wasInert]) => {
        element.inert = wasInert;
    });
    tutorialState.inertStates = [];
}

function startTutorial() {
    if (tutorialState.active) endTutorial({ restoreFocus: false });
    tutorialState.active = true;
    tutorialState.index = 0;
    tutorialState.previousFocus = document.activeElement;
    tutorialOverlay.hidden = false;
    document.body.classList.add('tutorial-active');
    setTutorialBackgroundInert(true);
    document.addEventListener('keydown', handleTutorialKeydown, true);
    window.addEventListener('resize', scheduleTutorialPosition);
    document.addEventListener('scroll', scheduleTutorialPosition, true);
    showTutorialStep();
    setTimeout(() => tutorialDialog.focus({ preventScroll: true }), 0);
}

function endTutorial({ restoreFocus=true }={}) {
    if (!tutorialState.active) return;
    tutorialState.active = false;
    tutorialState.renderToken += 1;
    tutorialState.target?.classList.remove('tutorial-target');
    tutorialState.target = null;
    tutorialState.resizeObserver?.disconnect();
    tutorialState.resizeObserver = null;
    if (tutorialState.positionFrame) cancelAnimationFrame(tutorialState.positionFrame);
    tutorialState.positionFrame = 0;
    document.removeEventListener('keydown', handleTutorialKeydown, true);
    window.removeEventListener('resize', scheduleTutorialPosition);
    document.removeEventListener('scroll', scheduleTutorialPosition, true);
    document.body.classList.remove('tutorial-active');
    tutorialOverlay.hidden = true;
    tutorialOverlay.style.removeProperty('--tutorial-x');
    tutorialOverlay.style.removeProperty('--tutorial-y');
    tutorialOverlay.style.removeProperty('--tutorial-w');
    tutorialOverlay.style.removeProperty('--tutorial-h');
    tutorialDialog.dataset.placement = 'center';
    tutorialState.index = 0;
    setTutorialBackgroundInert(false);

    const focusTarget = tutorialState.previousFocus?.isConnected
        ? tutorialState.previousFocus
        : tutorialLauncher;
    tutorialState.previousFocus = null;
    if (restoreFocus) focusTarget?.focus({ preventScroll: true });
}

tutorialLauncher.addEventListener('click', startTutorial);
tutorialCloseButton.addEventListener('click', () => endTutorial());
tutorialSkipButton.addEventListener('click', () => endTutorial());
tutorialBackButton.addEventListener('click', () => {
    if (tutorialState.index === 0) return;
    tutorialState.index -= 1;
    showTutorialStep();
});
tutorialNextButton.addEventListener('click', () => {
    if (tutorialState.index === TUTORIAL_STEPS.length - 1) {
        endTutorial();
        return;
    }
    tutorialState.index += 1;
    showTutorialStep();
});

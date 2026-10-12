import {
  useEffect,
  useMemo,
  useState,
} from "react";

import L from "leaflet";

import {
  Circle,
  CircleMarker,
  GeoJSON,
  MapContainer,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";

import type {
  GeoJsonObject,
} from "geojson";

import {
  getNearbyComplaints,
} from "../api/analytics";

import {
  getSaricamServiceArea,
} from "../api/serviceAreas";

import type {
  AnalyticsDateRange,
  MapPointItem,
  NearbyComplaintsResponse,
} from "../types/analytics";

import type {
  ServiceAreaFeature,
} from "../types/serviceArea";


const ALL_CATEGORIES = "ALL";

const RADIUS_OPTIONS = [
  {
    value: 500,
    label: "500 m",
  },
  {
    value: 1000,
    label: "1 km",
  },
  {
    value: 2000,
    label: "2 km",
  },
];


const CATEGORY_LABELS:
  Record<string, string> = {
    TEMIZLIK_ATIK:
      "Temizlik / Atık",

    YOL_KALDIRIM:
      "Yol / Kaldırım",

    PARK_YESIL_ALAN:
      "Park / Yeşil Alan",

    KENT_DUZENI_DENETIM:
      "Kent Düzeni / Denetim",

    HAYVAN_VETERINER:
      "Hayvan / Veteriner",
  };


const CATEGORY_COLORS:
  Record<string, string> = {
    TEMIZLIK_ATIK: "#2563eb",
    YOL_KALDIRIM: "#d97706",
    PARK_YESIL_ALAN: "#16a34a",
    KENT_DUZENI_DENETIM: "#7c3aed",
    HAYVAN_VETERINER: "#dc2626",
  };


const STATUS_LABELS:
  Record<string, string> = {
    REVIEW_REQUIRED:
      "İnceleme Bekliyor",

    AUTO_ACCEPTED:
      "Otomatik Kategorize",

    REVIEWED:
      "İncelenmiş",
  };


interface AnalysisCenter {
  latitude: number;
  longitude: number;
}


function categoryLabel(
  category: string,
): string {
  return (
    CATEGORY_LABELS[
      category
    ]
    ?? category
  );
}


function categoryColor(
  category: string,
): string {
  return (
    CATEGORY_COLORS[
      category
    ]
    ?? "#64748b"
  );
}


function statusLabel(
  status: string,
): string {
  return (
    STATUS_LABELS[
      status
    ]
    ?? status
  );
}


function formatComplaintDate(
  value: string,
): string {
  const date =
    new Date(value);

  if (
    Number.isNaN(
      date.getTime(),
    )
  ) {
    return value;
  }

  return new Intl.DateTimeFormat(
    "tr-TR",
    {
      dateStyle: "medium",
      timeStyle: "short",
    },
  ).format(
    date,
  );
}


function formatDistance(
  meters: number,
): string {
  if (meters < 1000) {
    return `${Math.round(meters)} m`;
  }

  return `${(meters / 1000).toFixed(2)} km`;
}


function FitServiceAreaBounds({
  serviceArea,
}: {
  serviceArea: ServiceAreaFeature;
}) {
  const map = useMap();

  useEffect(() => {
    const layer = L.geoJSON(
      serviceArea as GeoJsonObject,
    );

    const bounds =
      layer.getBounds();

    if (
      bounds.isValid()
    ) {
      map.fitBounds(
        bounds,
        {
          padding: [24, 24],
          maxZoom: 12,
        },
      );
    }

    const timer = window.setTimeout(
      () => {
        map.invalidateSize();
      },
      80,
    );

    return () => {
      window.clearTimeout(
        timer,
      );
    };
  }, [
    map,
    serviceArea,
  ]);

  return null;
}


function AnalysisClickHandler({
  onSelect,
}: {
  onSelect: (
    center: AnalysisCenter,
  ) => void;
}) {
  useMapEvents({
    click(event) {
      onSelect({
        latitude:
          event.latlng.lat,
        longitude:
          event.latlng.lng,
      });
    },
  });

  return null;
}


interface NearbyAnalysisContext {
  latitude: number;
  longitude: number;
  radiusMeters: number;
  category?: string;
}


interface ManagerComplaintMapProps {
  points: MapPointItem[];
  dateRange: AnalyticsDateRange;
  onAnalysisContextChange?: (
    context: NearbyAnalysisContext | null,
  ) => void;
}


function ManagerComplaintMap({
  points,
  dateRange,
  onAnalysisContextChange,
}: ManagerComplaintMapProps) {
  const [
    serviceArea,
    setServiceArea,
  ] = useState<
    ServiceAreaFeature | null
  >(null);

  const [
    serviceAreaError,
    setServiceAreaError,
  ] = useState<
    string | null
  >(null);

  const [
    selectedCategory,
    setSelectedCategory,
  ] = useState(
    ALL_CATEGORIES,
  );

  const [
    analysisCenter,
    setAnalysisCenter,
  ] = useState<
    AnalysisCenter | null
  >(null);

  const [
    analysisRadius,
    setAnalysisRadius,
  ] = useState(1000);

  const [
    nearbyData,
    setNearbyData,
  ] = useState<
    NearbyComplaintsResponse | null
  >(null);

  const [
    nearbyLoading,
    setNearbyLoading,
  ] = useState(false);

  const [
    nearbyError,
    setNearbyError,
  ] = useState<
    string | null
  >(null);


  useEffect(() => {
    let active = true;

    async function loadServiceArea() {
      try {
        setServiceAreaError(
          null,
        );

        const data =
          await getSaricamServiceArea();

        if (active) {
          setServiceArea(
            data,
          );
        }
      } catch (error) {
        if (!active) {
          return;
        }

        setServiceAreaError(
          error instanceof Error
            ? error.message
            : (
              "Sarıçam hizmet alanı "
              + "yüklenemedi."
            ),
        );
      }
    }

    loadServiceArea();

    return () => {
      active = false;
    };
  }, []);


  useEffect(() => {
    if (!analysisCenter) {
      return;
    }

    const center = analysisCenter;

    let active = true;

    async function loadNearby() {
      try {
        setNearbyLoading(
          true,
        );

        setNearbyData(
          null,
        );

        setNearbyError(
          null,
        );

        const data =
          await getNearbyComplaints({
            latitude:
              center.latitude,
            longitude:
              center.longitude,
            radiusMeters:
              analysisRadius,
            category:
              selectedCategory
              === ALL_CATEGORIES
                ? undefined
                : selectedCategory,
            range: dateRange,
          });

        if (active) {
          setNearbyData(
            data,
          );
        }
      } catch (error) {
        if (!active) {
          return;
        }

        setNearbyData(
          null,
        );

        setNearbyError(
          error instanceof Error
            ? error.message
            : (
              "Yakın çevre analizi "
              + "yüklenemedi."
            ),
        );
      } finally {
        if (active) {
          setNearbyLoading(
            false,
          );
        }
      }
    }

    loadNearby();

    return () => {
      active = false;
    };
  }, [
    analysisCenter,
    analysisRadius,
    selectedCategory,
    dateRange,
  ]);


  useEffect(() => {
    if (!onAnalysisContextChange) {
      return;
    }

    if (!analysisCenter) {
      onAnalysisContextChange(
        null,
      );
      return;
    }

    onAnalysisContextChange({
      latitude:
        analysisCenter.latitude,
      longitude:
        analysisCenter.longitude,
      radiusMeters:
        analysisRadius,
      category:
        selectedCategory
        === ALL_CATEGORIES
          ? undefined
          : selectedCategory,
    });
  }, [
    analysisCenter,
    analysisRadius,
    selectedCategory,
    onAnalysisContextChange,
  ]);


  const counts = useMemo(
    () => {
      const result =
        new Map<string, number>();

      for (
        const point of points
      ) {
        result.set(
          point.category,
          (
            result.get(
              point.category,
            ) ?? 0
          ) + 1,
        );
      }

      return result;
    },
    [points],
  );


  const filteredPoints =
    useMemo(
      () => (
        selectedCategory
        === ALL_CATEGORIES
          ? points
          : points.filter(
              (point) =>
                point.category
                === selectedCategory,
            )
      ),
      [
        points,
        selectedCategory,
      ],
    );


  const pointById = useMemo(
    () => new Map(
      points.map(
        (point) => [
          point.id,
          point,
        ],
      ),
    ),
    [points],
  );


  const nearbyIds = useMemo(
    () => new Set(
      nearbyData?.results.map(
        (item) => item.id,
      ) ?? [],
    ),
    [nearbyData],
  );


  const serviceAreaGeoJson:
    GeoJsonObject | null =
      serviceArea
        ? (serviceArea as GeoJsonObject)
        : null;


  function selectAnalysisCenter(
    center: AnalysisCenter,
  ) {
    setAnalysisCenter(
      center,
    );
  }


  function clearAnalysis() {
    setAnalysisCenter(
      null,
    );

    setNearbyData(
      null,
    );

    setNearbyError(
      null,
    );
  }


  return (
    <div className="manager-map-content">
      <div className="manager-map-filters">
        <button
          type="button"
          className={
            selectedCategory
            === ALL_CATEGORIES
              ? "active"
              : ""
          }
          onClick={() => {
            setSelectedCategory(
              ALL_CATEGORIES,
            );
          }}
        >
          <span className="manager-map-filter-dot manager-map-filter-dot-all" />
          Tümü
          <strong>
            {points.length}
          </strong>
        </button>

        {Object.entries(
          CATEGORY_LABELS,
        ).map(
          ([category, label]) => (
            <button
              key={category}
              type="button"
              className={
                selectedCategory
                === category
                  ? "active"
                  : ""
              }
              onClick={() => {
                setSelectedCategory(
                  category,
                );
              }}
            >
              <span
                className="manager-map-filter-dot"
                style={{
                  background:
                    categoryColor(
                      category,
                    ),
                }}
              />

              {label}

              <strong>
                {counts.get(
                  category,
                ) ?? 0}
              </strong>
            </button>
          ),
        )}
      </div>

      <div className="manager-map-visible-summary">
        <span>
          Haritada gösterilen
        </span>

        <strong>
          {filteredPoints.length}
          {" başvuru noktası"}
        </strong>
      </div>

      <div className="manager-nearby-toolbar">
        <div>
          <span className="manager-nearby-toolbar-label">
            Yakın çevre analizi
          </span>

          <strong>
            Haritada bir noktaya veya başvuru işaretine tıklayın
          </strong>
        </div>

        <div className="manager-nearby-radius-options">
          {RADIUS_OPTIONS.map(
            (option) => (
              <button
                key={option.value}
                type="button"
                className={
                  analysisRadius
                  === option.value
                    ? "active"
                    : ""
                }
                onClick={() => {
                  setAnalysisRadius(
                    option.value,
                  );
                }}
              >
                {option.label}
              </button>
            ),
          )}
        </div>
      </div>

      {serviceAreaError && (
        <div className="manager-map-state error">
          {serviceAreaError}
        </div>
      )}

      {(
        !serviceArea
        && !serviceAreaError
      ) && (
        <div className="manager-map-state">
          Sarıçam haritası yükleniyor...
        </div>
      )}

      {serviceArea && (
        <div className="manager-map-frame">
          <MapContainer
            center={[
              37.05,
              35.47,
            ]}
            zoom={11}
            scrollWheelZoom
            preferCanvas
            className="manager-map-leaflet"
          >
            <TileLayer
              attribution={
                "&copy; OpenStreetMap contributors"
              }
              url={
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              }
            />

            <GeoJSON
              data={
                serviceAreaGeoJson!
              }
              style={{
                color: "#315fd6",
                weight: 2,
                fillColor: "#315fd6",
                fillOpacity: 0.035,
              }}
            />

            <FitServiceAreaBounds
              serviceArea={
                serviceArea
              }
            />

            <AnalysisClickHandler
              onSelect={
                selectAnalysisCenter
              }
            />

            {analysisCenter && (
              <>
                <Circle
                  center={[
                    analysisCenter.latitude,
                    analysisCenter.longitude,
                  ]}
                  radius={
                    analysisRadius
                  }
                  pathOptions={{
                    color: "#0f172a",
                    fillColor: "#315fd6",
                    fillOpacity: 0.06,
                    weight: 2,
                    dashArray: "6 6",
                  }}
                />

                <CircleMarker
                  center={[
                    analysisCenter.latitude,
                    analysisCenter.longitude,
                  ]}
                  radius={8}
                  pathOptions={{
                    color: "#0f172a",
                    fillColor: "#ffffff",
                    fillOpacity: 1,
                    weight: 3,
                  }}
                >
                  <Popup>
                    Yakın çevre analizi merkezi
                  </Popup>
                </CircleMarker>
              </>
            )}

            {filteredPoints.map(
              (point) => {
                const color =
                  categoryColor(
                    point.category,
                  );

                const isNearby =
                  nearbyIds.has(
                    point.id,
                  );

                return (
                  <CircleMarker
                    key={point.id}
                    center={[
                      point.latitude,
                      point.longitude,
                    ]}
                    radius={
                      isNearby
                        ? 8
                        : 6
                    }
                    pathOptions={{
                      color,
                      fillColor: color,
                      fillOpacity:
                        isNearby
                          ? 0.95
                          : 0.78,
                      weight:
                        isNearby
                          ? 4
                          : 2,
                    }}
                    eventHandlers={{
                      click: () => {
                        selectAnalysisCenter({
                          latitude:
                            point.latitude,
                          longitude:
                            point.longitude,
                        });
                      },
                    }}
                  >
                    <Popup
                      minWidth={260}
                      maxWidth={340}
                    >
                      <div className="manager-map-popup">
                        <strong className="manager-map-popup-application">
                          {point.application_no}
                        </strong>

                        <div className="manager-map-popup-meta">
                          <span>
                            {point.neighborhood
                              ?? "Mahalle belirtilmemiş"}
                          </span>

                          <span>
                            {formatComplaintDate(
                              point.created_at,
                            )}
                          </span>
                        </div>

                        <div className="manager-map-popup-badges">
                          <span
                            className="manager-map-popup-category"
                            style={{
                              borderColor: color,
                              color,
                            }}
                          >
                            {categoryLabel(
                              point.category,
                            )}
                          </span>

                          <span className="manager-map-popup-status">
                            {statusLabel(
                              point.status,
                            )}
                          </span>
                        </div>

                        <p className="manager-map-popup-text">
                          {point.complaint_text}
                        </p>

                        <small className="manager-map-popup-hint">
                          Bu işarete tıklamak noktayı yakın çevre analizi merkezi yapar.
                        </small>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              },
            )}
          </MapContainer>
        </div>
      )}

      {analysisCenter && (
        <section className="manager-nearby-panel">
          <div className="manager-nearby-panel-header">
            <div>
              <span>
                PostGIS Yakın Çevre Analizi
              </span>

              <strong>
                {analysisRadius < 1000
                  ? `${analysisRadius} m`
                  : `${analysisRadius / 1000} km`}
                {" çevresindeki başvurular"}
              </strong>
            </div>

            <button
              type="button"
              onClick={
                clearAnalysis
              }
            >
              Analizi Temizle
            </button>
          </div>

          <div className="manager-nearby-context">
            <span>
              Tarih filtresi
              <strong>
                {dateRange.fromDate
                  || dateRange.toDate
                    ? (
                      `${dateRange.fromDate ?? "İlk kayıt"}`
                      + " → "
                      + `${dateRange.toDate ?? "Bugün"}`
                    )
                    : "Tüm kayıtlar"}
              </strong>
            </span>

            <span>
              Kategori filtresi
              <strong>
                {selectedCategory
                === ALL_CATEGORIES
                  ? "Tüm kategoriler"
                  : categoryLabel(
                      selectedCategory,
                    )}
              </strong>
            </span>

            <span>
              Bulunan kayıt
              <strong>
                {nearbyLoading
                  ? "…"
                  : nearbyData?.count ?? 0}
              </strong>
            </span>
          </div>

          {nearbyLoading && (
            <div className="manager-nearby-state">
              Gerçek PostGIS mesafe sorgusu çalıştırılıyor...
            </div>
          )}

          {nearbyError && (
            <div className="manager-nearby-state error">
              {nearbyError}
            </div>
          )}

          {(
            !nearbyLoading
            && !nearbyError
            && nearbyData
            && nearbyData.count === 0
          ) && (
            <div className="manager-nearby-state">
              Seçilen yarıçap, dönem ve kategori için yakın başvuru bulunamadı.
            </div>
          )}

          {(
            !nearbyLoading
            && !nearbyError
            && nearbyData
            && nearbyData.count > 0
          ) && (
            <div className="manager-nearby-results">
              {nearbyData.results
                .slice(0, 8)
                .map(
                  (item) => {
                    const detail =
                      pointById.get(
                        item.id,
                      );

                    return (
                      <article
                        key={item.id}
                        className="manager-nearby-result"
                      >
                        <div className="manager-nearby-result-heading">
                          <div>
                            <strong>
                              {item.application_no}
                            </strong>

                            <span>
                              {detail?.neighborhood
                                ?? "Mahalle belirtilmemiş"}
                            </span>
                          </div>

                          <strong className="manager-nearby-distance">
                            {formatDistance(
                              item.distance_meters,
                            )}
                          </strong>
                        </div>

                        <div className="manager-nearby-result-meta">
                          <span>
                            {categoryLabel(
                              item.category,
                            )}
                          </span>

                          <span>
                            {statusLabel(
                              item.status,
                            )}
                          </span>

                          {detail && (
                            <span>
                              {formatComplaintDate(
                                detail.created_at,
                              )}
                            </span>
                          )}
                        </div>

                        <p>
                          {item.complaint_text}
                        </p>
                      </article>
                    );
                  },
                )}

              {nearbyData.count > 8 && (
                <div className="manager-nearby-more">
                  En yakın 8 kayıt gösteriliyor. Toplam {nearbyData.count} kayıt yarıçap içinde.
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}


export default ManagerComplaintMap;

import {
  useEffect,
  useState,
} from "react";

import {
  useNavigate,
} from "react-router-dom";

import {
  getAnalyticsSummary,
  getCategoryDistribution,
  getComplaintTrend,
  getComplaintMapPoints,
  getNeighborhoodAnalytics,
  getModelHumanComparison,
} from "../api/analytics";

import type {
  AnalyticsDateRange,
  AnalyticsSummary,
  CategoryDistributionItem,
  TrendItem,
  MapPointItem,
  NeighborhoodAnalyticsItem,
  ModelHumanComparisonResponse,
} from "../types/analytics";

import ManagerComplaintMap
  from "../components/ManagerComplaintMap";

import ManagerAssistantPanel
  from "../components/ManagerAssistantPanel";

import "./ManagerPage.css";


type DatePreset =
  | "ALL"
  | "7D"
  | "30D"
  | "90D"
  | "CUSTOM";


interface NearbyAnalysisContext {
  latitude: number;
  longitude: number;
  radiusMeters: number;
  category?: string;
}


const DATE_PRESETS: {
  value: DatePreset;
  label: string;
}[] = [
  {
    value: "ALL",
    label: "Tümü",
  },
  {
    value: "7D",
    label: "Son 7 Gün",
  },
  {
    value: "30D",
    label: "Son 30 Gün",
  },
  {
    value: "90D",
    label: "Son 90 Gün",
  },
  {
    value: "CUSTOM",
    label: "Özel Tarih",
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


const CATEGORY_ORDER = [
  "TEMIZLIK_ATIK",
  "YOL_KALDIRIM",
  "PARK_YESIL_ALAN",
  "KENT_DUZENI_DENETIM",
  "HAYVAN_VETERINER",
];

const NEIGHBORHOOD_PREVIEW_LIMIT = 8;


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


function formatLocalDate(
  value: Date,
): string {
  const year =
    value.getFullYear();

  const month =
    String(
      value.getMonth() + 1,
    ).padStart(
      2,
      "0",
    );

  const day =
    String(
      value.getDate(),
    ).padStart(
      2,
      "0",
    );

  return `${year}-${month}-${day}`;
}


function presetDateRange(
  preset: DatePreset,
): AnalyticsDateRange {
  if (
    preset === "ALL"
    || preset === "CUSTOM"
  ) {
    return {};
  }

  const days =
    preset === "7D"
      ? 7
      : preset === "30D"
        ? 30
        : 90;

  const toDate =
    new Date();

  const fromDate =
    new Date();

  fromDate.setDate(
    fromDate.getDate()
    - (days - 1),
  );

  return {
    fromDate:
      formatLocalDate(
        fromDate,
      ),

    toDate:
      formatLocalDate(
        toDate,
      ),
  };
}


function percentage(
  value: number,
): string {
  return (
    `${(value * 100).toFixed(1)}%`
  );
}


function parseDateOnly(
  value: string,
): Date {
  const [
    year,
    month,
    day,
  ] = value
    .split("-")
    .map(Number);

  return new Date(
    year,
    month - 1,
    day,
  );
}


function formatTrendDate(
  value: string,
): string {
  const date =
    parseDateOnly(
      value,
    );

  return new Intl.DateTimeFormat(
    "tr-TR",
    {
      day: "2-digit",
      month: "short",
    },
  ).format(
    date,
  );
}


function fillDailyTrend(
  items: TrendItem[],
  range: AnalyticsDateRange,
): TrendItem[] {
  if (items.length === 0) {
    return [];
  }

  const countByDate =
    new Map(
      items.map(
        (item) => [
          item.date,
          item.count,
        ],
      ),
    );

  const startValue =
    range.fromDate
    ?? items[0].date;

  const endValue =
    range.toDate
    ?? items[
      items.length - 1
    ].date;

  const current =
    parseDateOnly(
      startValue,
    );

  const end =
    parseDateOnly(
      endValue,
    );

  const result: TrendItem[] = [];

  while (
    current <= end
  ) {
    const date =
      formatLocalDate(
        current,
      );

    result.push({
      date,
      count:
        countByDate.get(
          date,
        ) ?? 0,
    });

    current.setDate(
      current.getDate() + 1,
    );
  }

  return result;
}


function TrendChart({
  items,
}: {
  items: TrendItem[];
}) {
  const width = 1000;
  const height = 270;

  const plotLeft = 42;
  const plotRight = 978;
  const plotTop = 24;
  const plotBottom = 214;

  const plotWidth =
    plotRight - plotLeft;

  const plotHeight =
    plotBottom - plotTop;

  const maxCount =
    Math.max(
      1,
      ...items.map(
        (item) =>
          item.count,
      ),
    );

  const points =
    items.map(
      (item, index) => {
        const x =
          items.length === 1
            ? (
              plotLeft
              + plotWidth / 2
            )
            : (
              plotLeft
              + (
                index
                / (items.length - 1)
              )
              * plotWidth
            );

        const y =
          plotBottom
          - (
            item.count
            / maxCount
          )
          * plotHeight;

        return {
          ...item,
          x,
          y,
        };
      },
    );

  const linePoints =
    points
      .map(
        (point) =>
          `${point.x},${point.y}`,
      )
      .join(" ");

  const areaPoints = [
    `${plotLeft},${plotBottom}`,
    linePoints,
    `${plotRight},${plotBottom}`,
  ].join(" ");

  const middleIndex =
    Math.floor(
      (items.length - 1) / 2,
    );

  const markerStep =
    Math.max(
      1,
      Math.ceil(
        items.length / 36,
      ),
    );

  const gridLevels = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];

  return (
    <div className="manager-trend-chart-wrap">
      <svg
        className="manager-trend-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Günlük başvuru sayısı zaman trendi"
      >
        {gridLevels.map(
          (level) => {
            const y =
              plotBottom
              - level
              * plotHeight;

            const label =
              Math.round(
                maxCount
                * level,
              );

            return (
              <g
                key={level}
              >
                <line
                  className="manager-trend-grid-line"
                  x1={plotLeft}
                  y1={y}
                  x2={plotRight}
                  y2={y}
                />

                <text
                  className="manager-trend-y-label"
                  x={plotLeft - 12}
                  y={y + 4}
                  textAnchor="end"
                >
                  {label}
                </text>
              </g>
            );
          },
        )}

        <polygon
          className="manager-trend-area"
          points={areaPoints}
        />

        <polyline
          className="manager-trend-line"
          points={linePoints}
          fill="none"
        />

        {points.map(
          (point, index) => {
            const showMarker =
              index === 0
              || index
                === points.length - 1
              || index % markerStep
                === 0;

            if (!showMarker) {
              return null;
            }

            return (
              <circle
                key={point.date}
                className="manager-trend-point"
                cx={point.x}
                cy={point.y}
                r="4.5"
              >
                <title>
                  {formatTrendDate(
                    point.date,
                  )}
                  {" · "}
                  {point.count}
                  {" başvuru"}
                </title>
              </circle>
            );
          },
        )}

        <text
          className="manager-trend-x-label"
          x={plotLeft}
          y={252}
          textAnchor="start"
        >
          {formatTrendDate(
            items[0].date,
          )}
        </text>

        <text
          className="manager-trend-x-label"
          x={
            plotLeft
            + plotWidth / 2
          }
          y={252}
          textAnchor="middle"
        >
          {formatTrendDate(
            items[
              middleIndex
            ].date,
          )}
        </text>

        <text
          className="manager-trend-x-label"
          x={plotRight}
          y={252}
          textAnchor="end"
        >
          {formatTrendDate(
            items[
              items.length - 1
            ].date,
          )}
        </text>
      </svg>
    </div>
  );
}


function ManagerPage() {
  const navigate =
    useNavigate();

  const [
    activePreset,
    setActivePreset,
  ] = useState<DatePreset>(
    "ALL",
  );

  const [
    dateRange,
    setDateRange,
  ] = useState<
    AnalyticsDateRange
  >({});

  const [
    customFromDate,
    setCustomFromDate,
  ] = useState("");

  const [
    customToDate,
    setCustomToDate,
  ] = useState("");

  const [
    summary,
    setSummary,
  ] = useState<
    AnalyticsSummary | null
  >(null);


  const [
    categoryDistribution,
    setCategoryDistribution,
  ] = useState<
    CategoryDistributionItem[]
  >([]);


  const [
    trend,
    setTrend,
  ] = useState<
    TrendItem[]
  >([]);


  const [
    mapPoints,
    setMapPoints,
  ] = useState<
    MapPointItem[]
  >([]);


  const [
    nearbyAnalysisContext,
    setNearbyAnalysisContext,
  ] = useState<
    NearbyAnalysisContext | null
  >(null);


  const [
    neighborhoodAnalytics,
    setNeighborhoodAnalytics,
  ] = useState<
    NeighborhoodAnalyticsItem[]
  >([]);


  const [
    modelHumanComparison,
    setModelHumanComparison,
  ] = useState<
    ModelHumanComparisonResponse | null
  >(null);


  const [
    showAllNeighborhoods,
    setShowAllNeighborhoods,
  ] = useState(false);

  const [
    compareLeft,
    setCompareLeft,
  ] = useState("");

  const [
    compareRight,
    setCompareRight,
  ] = useState("");

  const [
    loading,
    setLoading,
  ] = useState(true);

  const [
    error,
    setError,
  ] = useState<
    string | null
  >(null);

  const [
    customDateError,
    setCustomDateError,
  ] = useState<
    string | null
  >(null);


  useEffect(() => {
    let active = true;

    async function loadAnalytics() {
      try {
        setLoading(
          true,
        );

        setError(
          null,
        );

        const [
          summaryData,
          categoryData,
          trendData,
          mapPointData,
          neighborhoodData,
          modelHumanData,
        ] = await Promise.all([
          getAnalyticsSummary(
            dateRange,
          ),
          getCategoryDistribution(
            dateRange,
          ),
          getComplaintTrend(
            dateRange,
          ),
          getComplaintMapPoints(
            dateRange,
          ),
          getNeighborhoodAnalytics(
            dateRange,
          ),
          getModelHumanComparison(
            dateRange,
          ),
        ]);

        if (!active) {
          return;
        }

        setSummary(
          summaryData,
        );

        setCategoryDistribution(
          categoryData,
        );

        setTrend(
          trendData,
        );

        setMapPoints(
          mapPointData,
        );

        setNeighborhoodAnalytics(
          neighborhoodData,
        );

        const neighborhoodNames =
          neighborhoodData.map(
            (item) =>
              item.neighborhood,
          );

        if (
          neighborhoodNames.length
          === 0
        ) {
          setCompareLeft("");
          setCompareRight("");
        } else {
          setCompareLeft(
            (current) =>
              neighborhoodNames.includes(
                current,
              )
                ? current
                : neighborhoodNames[0],
          );

          setCompareRight(
            (current) => {
              if (
                neighborhoodNames.includes(
                  current,
                )
                && current
                  !== neighborhoodNames[0]
              ) {
                return current;
              }

              return (
                neighborhoodNames[1]
                ?? neighborhoodNames[0]
              );
            },
          );
        }

        setModelHumanComparison(
          modelHumanData,
        );
      } catch (loadError) {
        if (!active) {
          return;
        }

        const message =
          loadError instanceof Error
            ? loadError.message
            : (
              "Analitik verileri "
              + "yüklenemedi."
            );

        if (
          message
          === "Oturumunuz sona erdi."
        ) {
          sessionStorage.clear();

          navigate(
            "/login",
            {
              replace: true,
            },
          );

          return;
        }

        setError(
          message,
        );
      } finally {
        if (active) {
          setLoading(
            false,
          );
        }
      }
    }

    loadAnalytics();

    return () => {
      active = false;
    };
  }, [
    dateRange,
    navigate,
  ]);



  function handlePresetChange(
    preset: DatePreset,
  ) {
    setActivePreset(
      preset,
    );

    setCustomDateError(
      null,
    );

    if (
      preset !== "CUSTOM"
    ) {
      setShowAllNeighborhoods(
        false,
      );

      setDateRange(
        presetDateRange(
          preset,
        ),
      );
    }
  }


  function applyCustomDates() {
    if (
      !customFromDate
      && !customToDate
    ) {
      setCustomDateError(
        (
          "En az bir başlangıç veya "
          + "bitiş tarihi seçin."
        ),
      );

      return;
    }

    if (
      customFromDate
      && customToDate
      && customFromDate
        > customToDate
    ) {
      setCustomDateError(
        (
          "Başlangıç tarihi bitiş "
          + "tarihinden sonra olamaz."
        ),
      );

      return;
    }

    setCustomDateError(
      null,
    );
    setShowAllNeighborhoods(
      false,
    );
    setDateRange({
      fromDate:
        customFromDate
        || undefined,

      toDate:
        customToDate
        || undefined,
    });
  }


  function handleLogout() {
    sessionStorage.clear();

    navigate(
      "/login",
      {
        replace: true,
      },
    );
  }


  const dailyTrend =
    fillDailyTrend(
      trend,
      dateRange,
    );

  const trendTotal =
    dailyTrend.reduce(
      (total, item) =>
        total + item.count,
      0,
    );

  const busiestDay =
    dailyTrend.length > 0
      ? dailyTrend.reduce(
          (highest, item) =>
            item.count
            > highest.count
              ? item
              : highest,
        )
      : null;

  const topNeighborhood =
    neighborhoodAnalytics[0]
    ?? null;

  const visibleNeighborhoods =
    showAllNeighborhoods
      ? neighborhoodAnalytics
      : neighborhoodAnalytics.slice(
          0,
          NEIGHBORHOOD_PREVIEW_LIMIT,
        );

  const hiddenNeighborhoodCount =
    Math.max(
      neighborhoodAnalytics.length
      - visibleNeighborhoods.length,
      0,
    );

  const comparedLeft =
    neighborhoodAnalytics.find(
      (item) =>
        item.neighborhood
        === compareLeft,
    ) ?? null;

  const comparedRight =
    neighborhoodAnalytics.find(
      (item) =>
        item.neighborhood
        === compareRight,
    ) ?? null;


  return (
    <main className="page manager-page">
      <div className="manager-container">
        <section className="manager-header">
          <div>
            <div className="eyebrow">
              Yönetici Karar Destek
            </div>

            <h1>
              Analitik &amp; CBS Paneli
            </h1>

            <p>
              Belediye başvurularını dönem,
              inceleme durumu ve model-insan
              etkileşimi açısından izleyin.
            </p>
          </div>

          <div className="manager-user">
            <span>
              {
                sessionStorage.getItem(
                  "username",
                )
              }
            </span>

            <button
              type="button"
              onClick={
                handleLogout
              }
            >
              Çıkış Yap
            </button>
          </div>
        </section>


        <section className="manager-filter-card">
          <div className="manager-section-heading">
            <div>
              <span>
                Analiz Dönemi
              </span>

              <strong>
                Tarih aralığını seçin
              </strong>
            </div>
          </div>

          <div className="manager-date-presets">
            {DATE_PRESETS.map(
              (preset) => (
                <button
                  key={
                    preset.value
                  }
                  type="button"
                  className={
                    activePreset
                    === preset.value
                      ? "active"
                      : ""
                  }
                  onClick={() => {
                    handlePresetChange(
                      preset.value,
                    );
                  }}
                >
                  {preset.label}
                </button>
              ),
            )}
          </div>

          {activePreset
            === "CUSTOM" && (
            <div className="manager-custom-dates">
              <label>
                <span>
                  Başlangıç
                </span>

                <input
                  type="date"
                  value={
                    customFromDate
                  }
                  onChange={(event) => {
                    setCustomFromDate(
                      event.target.value,
                    );

                    setCustomDateError(
                      null,
                    );
                  }}
                />
              </label>

              <label>
                <span>
                  Bitiş
                </span>

                <input
                  type="date"
                  value={
                    customToDate
                  }
                  onChange={(event) => {
                    setCustomToDate(
                      event.target.value,
                    );

                    setCustomDateError(
                      null,
                    );
                  }}
                />
              </label>

              <button
                type="button"
                onClick={
                  applyCustomDates
                }
              >
                Tarihi Uygula
              </button>
            </div>
          )}

          {customDateError && (
            <div className="manager-inline-error">
              {customDateError}
            </div>
          )}

          <div className="manager-active-range">
            {dateRange.fromDate
              || dateRange.toDate
              ? (
                <>
                  <span>
                    Aktif aralık
                  </span>

                  <strong>
                    {
                      dateRange.fromDate
                      ?? "İlk kayıt"
                    }
                    {" → "}
                    {
                      dateRange.toDate
                      ?? "Bugün"
                    }
                  </strong>
                </>
              )
              : (
                <>
                  <span>
                    Aktif aralık
                  </span>

                  <strong>
                    Tüm kayıtlar
                  </strong>
                </>
              )}
          </div>
        </section>


        {loading && (
          <section className="manager-state-card">
            Analitik veriler yükleniyor...
          </section>
        )}


        {error && (
          <section className="manager-error-card">
            {error}
          </section>
        )}


        {(
          !loading
          && !error
          && summary
        ) && (
          <>
            <section className="manager-kpi-grid">
              <article className="manager-kpi-card">
                <span>
                  Toplam Başvuru
                </span>

                <strong>
                  {
                    summary
                      .total_complaints
                  }
                </strong>

                <small>
                  Seçilen dönemdeki kayıtlar
                </small>
              </article>

              <article className="manager-kpi-card">
                <span>
                  Otomatik Kategorize
                </span>

                <strong>
                  {
                    summary
                      .auto_accepted
                  }
                </strong>

                <small>
                  {percentage(
                    summary
                      .auto_acceptance_rate,
                  )}
                  {" "}
                  mevcut oran
                </small>
              </article>

              <article className="manager-kpi-card">
                <span>
                  İnceleme Bekleyen
                </span>

                <strong>
                  {
                    summary
                      .review_pending
                  }
                </strong>

                <small>
                  Operatör kararı bekliyor
                </small>
              </article>

              <article className="manager-kpi-card">
                <span>
                  İncelenmiş
                </span>

                <strong>
                  {
                    summary
                      .reviewed
                  }
                </strong>

                <small>
                  İnsan kararı tamamlanmış
                </small>
              </article>

              <article className="manager-kpi-card">
                <span>
                  İnsan İnceleme Oranı
                </span>

                <strong>
                  {percentage(
                    summary
                      .human_review_rate,
                  )}
                </strong>

                <small>
                  Bekleyen + incelenmiş
                </small>
              </article>

              <article className="manager-kpi-card">
                <span>
                  İncelenenlerde AI–İnsan
                  Düzeltme Oranı
                </span>

                <strong>
                  {summary.reviewed > 0
                    ? percentage(
                        summary
                          .correction_rate,
                      )
                    : "—"}
                </strong>

                <small>
                  {summary.reviewed > 0
                    ? (
                      <>
                        {
                          summary
                            .corrected_by_human
                        }
                        {" / "}
                        {
                          summary
                            .reviewed
                        }
                        {" "}
                        incelenen kayıt
                        düzeltilmiş
                      </>
                    )
                    : (
                      "Henüz insan incelemesi yok"
                    )}
                </small>
              </article>
            </section>

            <section className="manager-category-card">
              <div className="manager-section-heading">
                <div>
                  <span>
                    Hizmet Kategorileri
                  </span>

                  <strong>
                    Kategori Dağılımı
                  </strong>
                </div>
              </div>

              {categoryDistribution.length > 0
                ? (
                  <div className="manager-category-list">
                    {categoryDistribution.map(
                      (item) => (
                        <div
                          key={
                            item.category
                          }
                          className="manager-category-row"
                        >
                          <div className="manager-category-row-header">
                            <strong>
                              {categoryLabel(
                                item.category,
                              )}
                            </strong>

                            <span>
                              {item.count}
                              {" kayıt · "}
                              {percentage(
                                item.percentage,
                              )}
                            </span>
                          </div>

                          <div
                            className="manager-category-track"
                            aria-hidden="true"
                          >
                            <div
                              className="manager-category-fill"
                              style={{
                                width:
                                  `${Math.max(
                                    0,
                                    Math.min(
                                      100,
                                      item.percentage
                                      * 100,
                                    ),
                                  )}%`,
                              }}
                            />
                          </div>
                        </div>
                      ),
                    )}
                  </div>
                )
                : (
                  <div className="manager-category-empty">
                    Seçilen dönemde kategori
                    dağılımı oluşturacak başvuru
                    bulunmuyor.
                  </div>
                )}
            </section>

            <section className="manager-trend-card">
              <div className="manager-section-heading manager-trend-heading">
                <div>
                  <span>
                    Başvuru Yoğunluğu
                  </span>

                  <strong>
                    Günlük Zaman Trendi
                  </strong>
                </div>

                {busiestDay && (
                  <div className="manager-trend-peak">
                    <span>
                      En yoğun gün
                    </span>

                    <strong>
                      {formatTrendDate(
                        busiestDay.date,
                      )}
                      {" · "}
                      {busiestDay.count}
                      {" başvuru"}
                    </strong>
                  </div>
                )}
              </div>

              {dailyTrend.length > 0
                ? (
                  <>
                    <div className="manager-trend-summary">
                      <span>
                        Gösterilen gün
                        <strong>
                          {dailyTrend.length}
                        </strong>
                      </span>

                      <span>
                        Dönem toplamı
                        <strong>
                          {trendTotal}
                        </strong>
                      </span>

                      <span>
                        Günlük ortalama
                        <strong>
                          {(
                            trendTotal
                            / dailyTrend.length
                          ).toFixed(1)}
                        </strong>
                      </span>
                    </div>

                    <TrendChart
                      items={dailyTrend}
                    />
                  </>
                )
                : (
                  <div className="manager-trend-empty">
                    Seçilen dönemde zaman trendi
                    oluşturacak başvuru bulunmuyor.
                  </div>
                )}
            </section>

            <section className="manager-map-card">
              <div className="manager-section-heading manager-map-heading">
                <div>
                  <span>
                    Coğrafi Dağılım
                  </span>

                  <strong>
                    Sarıçam Şikâyet Haritası
                  </strong>
                </div>

                <div className="manager-map-total">
                  <span>
                    Seçili dönemde
                  </span>

                  <strong>
                    {mapPoints.length} nokta
                  </strong>
                </div>
              </div>

              <ManagerComplaintMap
                points={mapPoints}
                dateRange={dateRange}
                onAnalysisContextChange={
                  setNearbyAnalysisContext
                }
              />
            </section>

            <section className="manager-neighborhood-card">
              <div className="manager-section-heading manager-neighborhood-heading">
                <div>
                  <span>
                    Mekânsal Özet
                  </span>

                  <strong>
                    Mahalle Bazlı Başvuru Dağılımı
                  </strong>
                </div>

                {topNeighborhood && (
                  <div className="manager-neighborhood-peak">
                    <span>
                      En fazla başvuru
                    </span>

                    <strong>
                      {topNeighborhood.neighborhood}
                      {" · "}
                      {topNeighborhood.count}
                      {" kayıt"}
                    </strong>
                  </div>
                )}
              </div>

              {neighborhoodAnalytics.length > 0
                ? (
                  <>
                    <div className="manager-neighborhood-summary">
                      <span>
                        Başvuru bulunan mahalle
                        <strong>
                          {neighborhoodAnalytics.length}
                        </strong>
                      </span>

                      <span>
                        Seçili dönem toplamı
                        <strong>
                          {neighborhoodAnalytics.reduce(
                            (total, item) =>
                              total + item.count,
                            0,
                          )}
                        </strong>
                      </span>

                      <span>
                        En yoğun mahallenin payı
                        <strong>
                          {topNeighborhood
                            ? percentage(
                                topNeighborhood.percentage,
                              )
                            : "—"}
                        </strong>
                      </span>
                    </div>

                    <div className="manager-neighborhood-list">
                      {visibleNeighborhoods.map(
                        (item) => (
                          <div
                            key={item.neighborhood}
                            className="manager-neighborhood-row"
                          >
                            <div className="manager-neighborhood-row-header">
                              <div>
                                <strong>
                                  {item.neighborhood}
                                </strong>

                                <span>
                                  Baskın kategori: {categoryLabel(
                                    item.dominant_category,
                                  )}
                                  {" · "}
                                  {item.dominant_category_count}
                                  {" kayıt"}
                                </span>
                              </div>

                              <span>
                                {item.count}
                                {" kayıt · "}
                                {percentage(
                                  item.percentage,
                                )}
                              </span>
                            </div>

                            <div
                              className="manager-neighborhood-track"
                              aria-hidden="true"
                            >
                              <div
                                className="manager-neighborhood-fill"
                                style={{
                                  width:
                                    `${Math.max(
                                      0,
                                      Math.min(
                                        100,
                                        item.percentage
                                        * 100,
                                      ),
                                    )}%`,
                                }}
                              />
                            </div>
                          </div>
                        ),
                      )}
                    </div>

                    {neighborhoodAnalytics.length
                      > NEIGHBORHOOD_PREVIEW_LIMIT && (
                      <div className="manager-neighborhood-list-controls">
                        <span>
                          {showAllNeighborhoods
                            ? (
                              `${neighborhoodAnalytics.length} mahallenin tamamı gösteriliyor`
                            )
                            : (
                              `En fazla başvuru alan ilk ${NEIGHBORHOOD_PREVIEW_LIMIT} mahalle gösteriliyor`
                            )}
                        </span>

                        <button
                          type="button"
                          onClick={() => {
                            setShowAllNeighborhoods(
                              (current) => !current,
                            );
                          }}
                        >
                          {showAllNeighborhoods
                            ? "Daha az göster"
                            : (
                              `Tüm mahalleleri göster (+${hiddenNeighborhoodCount})`
                            )}
                        </button>
                      </div>
                    )}

                    <div className="manager-neighborhood-compare">
                      <div className="manager-neighborhood-compare-heading">
                        <div>
                          <span>
                            Mahalle Karşılaştırması
                          </span>

                          <strong>
                            İki mahalleyi yan yana inceleyin
                          </strong>
                        </div>

                        <p>
                          Karşılaştırma yalnızca seçili tarih
                          dönemindeki başvuru sayılarına dayanır;
                          mahalle alanına göre normalize edilmiş
                          bir yoğunluk ölçüsü değildir.
                        </p>
                      </div>

                      <div className="manager-neighborhood-selectors">
                        <label>
                          Birinci mahalle
                          <select
                            value={compareLeft}
                            onChange={(event) => {
                              setCompareLeft(
                                event.target.value,
                              );
                            }}
                          >
                            {neighborhoodAnalytics.map(
                              (item) => (
                                <option
                                  key={item.neighborhood}
                                  value={item.neighborhood}
                                >
                                  {item.neighborhood}
                                </option>
                              ),
                            )}
                          </select>
                        </label>

                        <label>
                          İkinci mahalle
                          <select
                            value={compareRight}
                            onChange={(event) => {
                              setCompareRight(
                                event.target.value,
                              );
                            }}
                          >
                            {neighborhoodAnalytics.map(
                              (item) => (
                                <option
                                  key={item.neighborhood}
                                  value={item.neighborhood}
                                >
                                  {item.neighborhood}
                                </option>
                              ),
                            )}
                          </select>
                        </label>
                      </div>

                      {comparedLeft && comparedRight && (
                        <div className="manager-neighborhood-comparison-grid">
                          {[
                            comparedLeft,
                            comparedRight,
                          ].map(
                            (item) => (
                              <article
                                key={item.neighborhood}
                                className="manager-neighborhood-comparison-card"
                              >
                                <div className="manager-neighborhood-comparison-title">
                                  <strong>
                                    {item.neighborhood}
                                  </strong>

                                  <span>
                                    {item.count}
                                    {" kayıt · "}
                                    {percentage(
                                      item.percentage,
                                    )}
                                  </span>
                                </div>

                                <div className="manager-neighborhood-dominant">
                                  <span>
                                    Baskın kategori
                                  </span>

                                  <strong>
                                    {categoryLabel(
                                      item.dominant_category,
                                    )}
                                    {" · "}
                                    {item.dominant_category_count}
                                  </strong>
                                </div>

                                <div className="manager-neighborhood-category-breakdown">
                                  {CATEGORY_ORDER.map(
                                    (category) => (
                                      <div
                                        key={category}
                                      >
                                        <span>
                                          {categoryLabel(
                                            category,
                                          )}
                                        </span>

                                        <strong>
                                          {item.category_counts[
                                            category
                                          ] ?? 0}
                                        </strong>
                                      </div>
                                    ),
                                  )}
                                </div>
                              </article>
                            ),
                          )}
                        </div>
                      )}
                    </div>
                  </>
                )
                : (
                  <div className="manager-neighborhood-empty">
                    Seçilen dönemde mahalle bazlı dağılım
                    oluşturacak başvuru bulunmuyor.
                  </div>
                )}
            </section>

            <section className="manager-model-human-card">
              <div className="manager-section-heading manager-model-human-heading">
                <div>
                  <span>
                    Model–İnsan Etkileşimi
                  </span>

                  <strong>
                    AI–İnsan Karar Karşılaştırması
                  </strong>
                </div>

                {modelHumanComparison
                  && modelHumanComparison.reviewed > 0 && (
                  <div className="manager-model-human-highlight">
                    <span>
                      İncelenenlerde düzeltme
                    </span>

                    <strong>
                      {percentage(
                        modelHumanComparison
                          .correction_rate,
                      )}
                    </strong>
                  </div>
                )}
              </div>

              {modelHumanComparison
                && modelHumanComparison.reviewed > 0
                ? (
                  <>
                    <div className="manager-model-human-summary">
                      <div>
                        <span>
                          İncelenen
                        </span>

                        <strong>
                          {modelHumanComparison.reviewed}
                        </strong>
                      </div>

                      <div>
                        <span>
                          AI tahmini onaylanan
                        </span>

                        <strong>
                          {modelHumanComparison.confirmed}
                        </strong>
                      </div>

                      <div>
                        <span>
                          AI tahmini değiştirilen
                        </span>

                        <strong>
                          {modelHumanComparison.corrected}
                        </strong>
                      </div>

                      <div>
                        <span>
                          AI–insan uyum oranı
                        </span>

                        <strong>
                          {percentage(
                            modelHumanComparison
                              .agreement_rate,
                          )}
                        </strong>
                      </div>
                    </div>

                    <div className="manager-model-human-note">
                      Bu göstergeler yalnızca insan tarafından
                      incelenmiş kayıtları karşılaştırır; genel
                      model doğruluğu olarak yorumlanmamalıdır.
                    </div>

                    <div className="manager-model-human-content">
                      <div className="manager-model-human-category-panel">
                        <div className="manager-model-human-subheading">
                          <span>
                            Kategori Bazında
                          </span>

                          <strong>
                            İlk AI tahminine göre inceleme sonucu
                          </strong>
                        </div>

                        <div className="manager-model-human-category-list">
                          {modelHumanComparison
                            .by_predicted_category
                            .map(
                              (item) => (
                                <div
                                  key={item.category}
                                  className="manager-model-human-category-row"
                                >
                                  <div className="manager-model-human-category-title">
                                    <strong>
                                      {categoryLabel(
                                        item.category,
                                      )}
                                    </strong>

                                    <span>
                                      {item.reviewed}
                                      {" incelendi"}
                                    </span>
                                  </div>

                                  <div className="manager-model-human-category-values">
                                    <span>
                                      Onaylanan
                                      <strong>
                                        {item.confirmed}
                                      </strong>
                                    </span>

                                    <span>
                                      Değiştirilen
                                      <strong>
                                        {item.corrected}
                                      </strong>
                                    </span>

                                    <span>
                                      Düzeltme oranı
                                      <strong>
                                        {percentage(
                                          item.correction_rate,
                                        )}
                                      </strong>
                                    </span>
                                  </div>

                                  <div
                                    className="manager-model-human-track"
                                    aria-hidden="true"
                                  >
                                    <div
                                      className="manager-model-human-fill"
                                      style={{
                                        width:
                                          `${Math.max(
                                            0,
                                            Math.min(
                                              100,
                                              item.correction_rate
                                              * 100,
                                            ),
                                          )}%`,
                                      }}
                                    />
                                  </div>
                                </div>
                              ),
                            )}
                        </div>
                      </div>

                      <div className="manager-model-human-flow-panel">
                        <div className="manager-model-human-subheading">
                          <span>
                            Düzeltme Akışları
                          </span>

                          <strong>
                            AI tahmini hangi nihai kategoriye çevrildi?
                          </strong>
                        </div>

                        {modelHumanComparison
                          .correction_flows.length > 0
                          ? (
                            <div className="manager-model-human-flow-list">
                              {modelHumanComparison
                                .correction_flows
                                .map(
                                  (flow) => (
                                    <div
                                      key={
                                        flow.predicted_category
                                        + "->"
                                        + flow.final_category
                                      }
                                      className="manager-model-human-flow-row"
                                    >
                                      <div>
                                        <span>
                                          {categoryLabel(
                                            flow.predicted_category,
                                          )}
                                        </span>

                                        <b aria-hidden="true">
                                          →
                                        </b>

                                        <strong>
                                          {categoryLabel(
                                            flow.final_category,
                                          )}
                                        </strong>
                                      </div>

                                      <span>
                                        {flow.count}
                                        {" kayıt"}
                                      </span>
                                    </div>
                                  ),
                                )}
                            </div>
                          )
                          : (
                            <div className="manager-model-human-flow-empty">
                              Seçilen dönemde insan tarafından
                              değiştirilen AI tahmini bulunmuyor.
                            </div>
                          )}
                      </div>
                    </div>
                  </>
                )
                : (
                  <div className="manager-model-human-empty">
                    Seçilen dönemde henüz insan tarafından
                    incelenmiş kayıt bulunmuyor.
                  </div>
                )}
            </section>

            <ManagerAssistantPanel
              dateRange={dateRange}
              nearbyContext={
                nearbyAnalysisContext
              }
            />
          </>
        )}
      </div>
    </main>
  );
}


export default ManagerPage;

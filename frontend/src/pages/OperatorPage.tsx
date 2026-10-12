import {
  useEffect,
  useState,
} from "react";

import {
  useNavigate,
} from "react-router-dom";

import {
  getComplaintsByStatus,
  reviewComplaint,
} from "../api/complaints";

import type {
  ComplaintCategory,
  ComplaintListItem,
  ComplaintStatus,
} from "../types/complaint";


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


const REVIEW_REASON_LABELS:
  Record<string, string> = {
    LOW_CONFIDENCE:
      "Model güveni düşük",

    CLOSE_TOP2:
      (
        "İlk iki kategori arasındaki "
        + "güven farkı düşük"
      ),

    LOW_CONFIDENCE_AND_CLOSE_TOP2:
      (
        "Model güveni düşük ve ilk iki "
        + "kategori birbirine yakın"
      ),
  };


const COMPLAINT_CATEGORIES:
  ComplaintCategory[] = [
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
  ];


const STATUS_TABS: {
  status: ComplaintStatus;
  label: string;
}[] = [
  {
    status: "REVIEW_REQUIRED",
    label: "İnceleme Bekleyen",
  },
  {
    status: "AUTO_ACCEPTED",
    label: "Otomatik Kategorize",
  },
  {
    status: "REVIEWED",
    label: "İncelenmiş",
  },
];


const STATUS_SUMMARY_LABELS:
  Record<ComplaintStatus, string> = {
    REVIEW_REQUIRED:
      "İnceleme bekleyen başvuru",

    AUTO_ACCEPTED:
      "Otomatik kategorize edilen başvuru",

    REVIEWED:
      "İnsan tarafından incelenmiş başvuru",
  };


const STATUS_BADGE_LABELS:
  Record<ComplaintStatus, string> = {
    REVIEW_REQUIRED:
      "İnsan Kontrolü",

    AUTO_ACCEPTED:
      "Otomatik Kategorize",

    REVIEWED:
      "İncelendi",
  };


const EMPTY_STATE_TITLES:
  Record<ComplaintStatus, string> = {
    REVIEW_REQUIRED:
      "İnceleme bekleyen başvuru yok.",

    AUTO_ACCEPTED:
      "Otomatik kategorize edilmiş başvuru yok.",

    REVIEWED:
      "İncelenmiş başvuru yok.",
  };


const EMPTY_STATE_DESCRIPTIONS:
  Record<ComplaintStatus, string> = {
    REVIEW_REQUIRED:
      (
        "Düşük güvenli bir başvuru "
        + "oluştuğunda burada "
        + "görüntülenecek."
      ),

    AUTO_ACCEPTED:
      (
        "Modelin yüksek güvenle "
        + "otomatik kategorize ettiği "
        + "başvurular burada "
        + "görüntülenecek."
      ),

    REVIEWED:
      (
        "Operatör tarafından "
        + "incelenmiş başvurular "
        + "burada görüntülenecek."
      ),
  };


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


function reviewReasonLabel(
  reason: string,
): string {
  return (
    REVIEW_REASON_LABELS[
      reason
    ]
    ?? reason
  );
}


function isComplaintCategory(
  value: string | undefined,
): value is ComplaintCategory {
  return (
    value === "TEMIZLIK_ATIK"
    || value === "YOL_KALDIRIM"
    || value === "PARK_YESIL_ALAN"
    || value === "KENT_DUZENI_DENETIM"
    || value === "HAYVAN_VETERINER"
  );
}
function isComplaintStatus(
  value: string,
): value is ComplaintStatus {
  return (
    value === "REVIEW_REQUIRED"
    || value === "AUTO_ACCEPTED"
    || value === "REVIEWED"
  );
}


function statusBadgeLabel(
  status: string,
): string {
  if (
    isComplaintStatus(
      status,
    )
  ) {
    return (
      STATUS_BADGE_LABELS[
        status
      ]
    );
  }

  return status;
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


async function getComplaintByApplicationNo(
  applicationNo: string,
): Promise<ComplaintListItem> {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  if (!accessToken) {
    throw new Error(
      "Oturum bulunamadı.",
    );
  }

  const apiBaseUrl =
    import.meta.env.VITE_API_BASE_URL;

  if (!apiBaseUrl) {
    throw new Error(
      "API adresi yapılandırılmamış.",
    );
  }

  const normalizedApplicationNo =
    applicationNo
      .trim()
      .toUpperCase();

  const response = await fetch(
    (
      `${apiBaseUrl}`
      + "/complaints/by-application-no/"
      + encodeURIComponent(
        normalizedApplicationNo,
      )
    ),
    {
      headers: {
        Authorization:
          `Bearer ${accessToken}`,
      },
    },
  );

  if (response.status === 401) {
    sessionStorage.clear();

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (response.status === 404) {
    throw new Error(
      "Bu başvuru numarasına ait kayıt bulunamadı.",
    );
  }

  if (!response.ok) {
    let message =
      "Başvuru sorgulanamadı.";

    try {
      const body =
        await response.json();

      if (
        body
        && typeof body.detail
        === "string"
      ) {
        message =
          body.detail;
      }
    } catch {
      // JSON olmayan hata cevabında
      // varsayılan mesaj kullanılır.
    }

    throw new Error(
      message,
    );
  }

  return response.json();
}


function OperatorPage() {
  const navigate =
    useNavigate();

  const [
    applicationSearch,
    setApplicationSearch,
  ] = useState("");

  const [
    searchResult,
    setSearchResult,
  ] = useState<
    ComplaintListItem | null
  >(null);

  const [
    searchLoading,
    setSearchLoading,
  ] = useState(false);

  const [
    searchError,
    setSearchError,
  ] = useState<
    string | null
  >(null);

  const [
    activeStatus,
    setActiveStatus,
  ] = useState<ComplaintStatus>(
    "REVIEW_REQUIRED",
  );

  const [
    complaints,
    setComplaints,
  ] = useState<
    ComplaintListItem[]
  >([]);

  const PAGE_SIZE = 100;

  const [
    currentPage,
    setCurrentPage,
  ] = useState(1);

  const [
    totalComplaints,
    setTotalComplaints,
  ] = useState(0);

  const [
    totalPages,
    setTotalPages,
  ] = useState(0);

  const [
    refreshKey,
    setRefreshKey,
  ] = useState(0);

  const [
    selectedCategories,
    setSelectedCategories,
  ] = useState<
    Record<number, string>
  >({});

  const [
    reviewingId,
    setReviewingId,
  ] = useState<
    number | null
  >(null);

  const [
    reviewErrors,
    setReviewErrors,
  ] = useState<
    Record<number, string>
  >({});

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


  useEffect(() => {
    let active = true;

    async function loadQueue() {
      try {
        setLoading(
          true,
        );

        setError(
          null,
        );

        const data =
          await getComplaintsByStatus(
            activeStatus,
            currentPage,
            PAGE_SIZE,
          );

        if (!active) {
          return;
        }

        if (
          data.total_pages > 0
          && currentPage > data.total_pages
        ) {
          setCurrentPage(
            data.total_pages,
          );

          return;
        }

        setComplaints(
          data.items,
        );

        setTotalComplaints(
          data.total,
        );

        setTotalPages(
          data.total_pages,
        );

        const initialCategories:
          Record<number, string> = {};

        for (
          const complaint
          of data.items
        ) {
          initialCategories[
            complaint.id
          ] =
            complaint
              .predicted_category;
        }

        setSelectedCategories(
          initialCategories,
        );

        setReviewErrors(
          {},
        );
      } catch (error) {
        if (!active) {
          return;
        }

        const message =
          error instanceof Error
            ? error.message
            : (
              "Başvurular "
              + "yüklenemedi."
            );

        setError(
          message,
        );

        if (
          message
          === "Oturumunuz sona erdi."
        ) {
          navigate(
            "/login",
            {
              replace: true,
            },
          );
        }
      } finally {
        if (active) {
          setLoading(
            false,
          );
        }
      }
    }

    loadQueue();

    return () => {
      active = false;
    };
  }, [
    navigate,
    activeStatus,
    currentPage,
    refreshKey,
  ]);


  async function handleApplicationSearch() {
    const normalized =
      applicationSearch
        .trim()
        .toUpperCase();

    if (!normalized) {
      setSearchError(
        "Lütfen başvuru numarası girin.",
      );

      setSearchResult(
        null,
      );

      return;
    }

    try {
      setSearchLoading(
        true,
      );

      setSearchError(
        null,
      );

      setSearchResult(
        null,
      );

      const result =
        await getComplaintByApplicationNo(
          normalized,
        );

      setSearchResult(
        result,
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : (
            "Başvuru aranırken "
            + "bir hata oluştu."
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

      setSearchError(
        message,
      );
    } finally {
      setSearchLoading(
        false,
      );
    }
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


  function handleCategoryChange(
    complaintId: number,
    category: string,
  ) {
    setSelectedCategories(
      (current) => ({
        ...current,

        [complaintId]:
          category,
      }),
    );

    setReviewErrors(
      (current) => {
        const next = {
          ...current,
        };

        delete next[
          complaintId
        ];

        return next;
      },
    );
  }


  async function handleReview(
    complaint:
      ComplaintListItem,
  ) {
    if (
      reviewingId !== null
    ) {
      return;
    }

    const finalCategory =
      selectedCategories[
        complaint.id
      ];

    if (
      !isComplaintCategory(
        finalCategory,
      )
    ) {
      setReviewErrors(
        (current) => ({
          ...current,

          [complaint.id]:
            (
              "Lütfen geçerli bir "
              + "nihai kategori seçin."
            ),
        }),
      );

      return;
    }

    try {
      setReviewingId(
        complaint.id,
      );

      setReviewErrors(
        (current) => {
          const next = {
            ...current,
          };

          delete next[
            complaint.id
          ];

          return next;
        },
      );

      await reviewComplaint(
        complaint.id,
        {
          final_category:
            finalCategory,
        },
      );

      setSearchResult(
        (current) => {
          if (
            !current
            || current.id
              !== complaint.id
          ) {
            return current;
          }

          return {
            ...current,
            status: "REVIEWED",
            final_category:
              finalCategory,
            was_corrected:
              finalCategory
              !== complaint
                .predicted_category,
          };
        },
      );

      setSelectedCategories(
        (current) => {
          const next = {
            ...current,
          };

          delete next[
            complaint.id
          ];

          return next;
        },
      );

      setReviewErrors(
        (current) => {
          const next = {
            ...current,
          };

          delete next[
            complaint.id
          ];

          return next;
        },
      );

      setRefreshKey(
        (current) =>
          current + 1,
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : (
            "İnceleme "
            + "tamamlanamadı."
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

      setReviewErrors(
        (current) => ({
          ...current,

          [complaint.id]:
            message,
        }),
      );
    } finally {
      setReviewingId(
        null,
      );
    }
  }


  return (
    <main className="page operator-page">
      <div className="operator-container">
        <section className="operator-header">
          <div>
            <div className="eyebrow">
              Operatör Paneli
            </div>

            <h1>
              İnsan Kontrollü
              İnceleme
            </h1>

            <p>
              BERTurk tarafından
              sınıflandırılan başvuruları
              inceleyin ve gerektiğinde
              nihai kategoriyi düzeltin.
            </p>
          </div>

          <div className="operator-user">
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


        <section className="review-decision">
          <div className="review-decision-heading">
            <div>
              <span>
                Başvuru Sorgulama
              </span>

              <strong>
                Başvuru numarasıyla ara
              </strong>
            </div>
          </div>

          <form
            className="review-decision-controls"
            onSubmit={(event) => {
              event.preventDefault();

              handleApplicationSearch();
            }}
          >
            <input
              type="text"
              value={
                applicationSearch
              }
              placeholder={
                "Örn. SRM-97F47C2E0FA1"
              }
              autoComplete="off"
              disabled={
                searchLoading
              }
              onChange={(event) => {
                setApplicationSearch(
                  event.target.value,
                );

                if (searchError) {
                  setSearchError(
                    null,
                  );
                }
              }}
              style={{
                width: "100%",
                minHeight: "46px",
                padding: "0 14px",
                border:
                  "1px solid #d9e0ec",
                borderRadius: "10px",
                font: "inherit",
                boxSizing: "border-box",
              }}
            />

            <button
              type="submit"
              className="review-submit-button"
              disabled={
                searchLoading
              }
            >
              {searchLoading
                ? "Aranıyor..."
                : "Ara"}
            </button>
          </form>

          {searchError && (
            <div
              className={
                "location-message "
                + "error"
              }
            >
              {searchError}
            </div>
          )}

          {searchResult && (
            <article
              className="review-card"
              style={{
                marginTop: "18px",
              }}
            >
              <div className="review-card-header">
                <div>
                  <span className="review-application-no">
                    {
                      searchResult
                        .application_no
                    }
                  </span>

                  <span className="review-neighborhood">
                    {
                      searchResult
                        .neighborhood
                      ?? (
                        "Mahalle "
                        + "belirtilmemiş"
                      )
                    }
                  </span>

                  <span
                    className="review-neighborhood"
                    style={{
                      marginTop: "5px",
                      opacity: 0.78,
                    }}
                  >
                    Başvuru tarihi:{" "}
                    {formatComplaintDate(
                      searchResult
                        .created_at,
                    )}
                  </span>
                </div>

                <span className="review-required-badge">
                  {statusBadgeLabel(
                    searchResult.status,
                  )}
                </span>
              </div>

              <p className="review-complaint-text">
                {
                  searchResult
                    .complaint_text
                }
              </p>

              <div className="review-ai-grid">
                <div>
                  <span>
                    AI Tahmini
                  </span>

                  <strong>
                    {categoryLabel(
                      searchResult
                        .predicted_category,
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Nihai Kategori
                  </span>

                  <strong>
                    {searchResult.final_category
                      ? categoryLabel(
                          searchResult
                            .final_category,
                        )
                      : "Henüz belirlenmedi"}
                  </strong>
                </div>

                <div>
                  <span>
                    Model Güveni
                  </span>

                  <strong>
                    {(
                      searchResult
                        .model_score
                      * 100
                    ).toFixed(1)}
                    %
                  </strong>
                </div>

                <div>
                  <span>
                    Durum
                  </span>

                  <strong>
                    {statusBadgeLabel(
                      searchResult.status,
                    )}
                  </strong>
                </div>
              </div>

              {searchResult.review_reason && (
                <div className="review-reason">
                  <span>
                    İnceleme nedeni
                  </span>

                  <strong>
                    {reviewReasonLabel(
                      searchResult
                        .review_reason,
                    )}
                  </strong>
                </div>
              )}

              {isComplaintStatus(
                searchResult.status,
              ) && (
                <div
                  className="review-decision-controls"
                  style={{
                    marginTop: "16px",
                  }}
                >
                  <button
                    type="button"
                    className="review-submit-button"
                    onClick={() => {
                      if (
                        isComplaintStatus(
                          searchResult.status,
                        )
                      ) {
                        setActiveStatus(
                          searchResult.status,
                        );

                        setCurrentPage(
                          1,
                        );
                      }
                    }}
                  >
                    İlgili Durum Sekmesini Aç
                  </button>
                </div>
              )}
            </article>
          )}
        </section>


        <nav
          className="operator-tabs"
          aria-label="Başvuru durumları"
        >
          {STATUS_TABS.map(
            (tab) => (
              <button
                key={
                  tab.status
                }
                type="button"
                className={
                  activeStatus
                  === tab.status
                    ? (
                      "operator-tab "
                      + "active"
                    )
                    : "operator-tab"
                }
                onClick={() => {
                  setActiveStatus(
                    tab.status,
                  );

                  setCurrentPage(
                    1,
                  );
                }}
              >
                {tab.label}
              </button>
            ),
          )}
        </nav>


        <section className="operator-summary">
          <span>
            {
              STATUS_SUMMARY_LABELS[
                activeStatus
              ]
            }
          </span>

          <strong>
            {totalComplaints}
          </strong>
        </section>


        {loading && (
          <section className="operator-state">
            Başvurular
            yükleniyor...
          </section>
        )}


        {error && (
          <section
            className={
              "location-message error"
            }
          >
            {error}
          </section>
        )}


        {(
          !loading
          && !error
          && complaints.length === 0
        ) && (
          <section className="operator-state">
            <strong>
              {
                EMPTY_STATE_TITLES[
                  activeStatus
                ]
              }
            </strong>

            <span>
              {
                EMPTY_STATE_DESCRIPTIONS[
                  activeStatus
                ]
              }
            </span>
          </section>
        )}


        {(
          !loading
          && !error
          && complaints.length > 0
        ) && (
          <section className="review-list">
            {complaints.map(
              (complaint) => {
                const selectedCategory =
                  selectedCategories[
                    complaint.id
                  ];

                const isCorrected =
                  Boolean(
                    selectedCategory
                    && selectedCategory
                    !== (
                      complaint
                        .predicted_category
                    ),
                  );

                const isSubmitting =
                  reviewingId
                  === complaint.id;

                

                const isReviewed =
                  complaint.status
                  === "REVIEWED";

                return (
                  <article
                    key={
                      complaint.id
                    }
                    className="review-card"
                  >
                    <div className="review-card-header">
                      <div>
                        <span className="review-application-no">
                          {
                            complaint
                              .application_no
                          }
                        </span>

                        <span className="review-neighborhood">
                          {
                            complaint
                              .neighborhood
                            ?? (
                              "Mahalle "
                              + "belirtilmemiş"
                            )
                          }
                        </span>

                        <span
                          className="review-neighborhood"
                          style={{
                            marginTop: "5px",
                            opacity: 0.78,
                          }}
                        >
                          Başvuru tarihi:{" "}
                          {formatComplaintDate(
                            complaint
                              .created_at,
                          )}
                        </span>
                      </div>

                      <span className="review-required-badge">
                        {statusBadgeLabel(
                          complaint.status,
                        )}
                      </span>
                    </div>


                    <p className="review-complaint-text">
                      {
                        complaint
                          .complaint_text
                      }
                    </p>


                    <div className="review-ai-grid">
                      <div>
                        <span>
                          AI Tahmini
                        </span>

                        <strong>
                          {categoryLabel(
                            complaint
                              .predicted_category,
                          )}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Model Güveni
                        </span>

                        <strong>
                          {(
                            complaint
                              .model_score
                            * 100
                          ).toFixed(1)}
                          %
                        </strong>
                      </div>

                      <div>
                        <span>
                          İkinci Kategori
                        </span>

                        <strong>
                          {categoryLabel(
                            complaint
                              .secondary_category,
                          )}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Güven Farkı
                        </span>

                        <strong>
                          {(
                            complaint
                              .score_margin
                            * 100
                          ).toFixed(1)}
                          %
                        </strong>
                      </div>
                    </div>


                    {complaint.review_reason && (
                      <div className="review-reason">
                        <span>
                          İnceleme nedeni
                        </span>

                        <strong>
                          {reviewReasonLabel(
                            complaint.review_reason,
                          )}
                        </strong>
                      </div>
                    )}


                    {!isReviewed && (
                      <div className="review-decision">
                        <div className="review-decision-heading">
                          <div>
                            <span>
                              Operatör Kararı
                            </span>

                            <strong>
                              Nihai Kategori
                            </strong>
                          </div>

                          <span
                            className={
                              isCorrected
                                ? (
                                  "decision-status "
                                  + "corrected"
                                )
                                : (
                                  "decision-status "
                                  + "accepted"
                                )
                            }
                          >
                            {isCorrected
                              ? (
                                "AI tahmini "
                                + "düzeltilecek"
                              )
                              : (
                                "AI tahmini "
                                + "onaylanacak"
                              )}
                          </span>
                        </div>


                        <div className="review-decision-controls">
                          <select
                            value={
                              selectedCategory
                              ?? ""
                            }
                            disabled={
                              isSubmitting
                            }
                            onChange={(event) => {
                              handleCategoryChange(
                                complaint.id,
                                event.target.value,
                              );
                            }}
                          >
                            {COMPLAINT_CATEGORIES.map(
                              (category) => (
                                <option
                                  key={
                                    category
                                  }
                                  value={
                                    category
                                  }
                                >
                                  {
                                    CATEGORY_LABELS[
                                      category
                                    ]
                                  }
                                </option>
                              ),
                            )}
                          </select>


                          <button
                            type="button"
                            className="review-submit-button"
                            disabled={
                              isSubmitting
                              || !isComplaintCategory(
                                selectedCategory,
                              )
                            }
                            onClick={() => {
                              handleReview(
                                complaint,
                              );
                            }}
                          >
                            {isSubmitting
                              ? "Kaydediliyor..."
                              : (
                                complaint.status
                                === "AUTO_ACCEPTED"
                                  ? "Kaydı Revize Et"
                                  : "İncelemeyi Tamamla"
                              )}
                          </button>
                        </div>


                        {reviewErrors[
                          complaint.id
                        ] && (
                          <div
                            className={
                              "location-message "
                              + "error"
                            }
                          >
                            {
                              reviewErrors[
                                complaint.id
                              ]
                            }
                          </div>
                        )}
                      </div>
                    )}


                    {isReviewed && (
                      <div className="review-decision">
                        <div className="review-decision-heading">
                          <div>
                            <span>
                              İnsan Kararı
                            </span>

                            <strong>
                              Nihai Kategori
                            </strong>
                          </div>

                          <span
                            className={
                              complaint
                                .was_corrected
                                ? (
                                  "decision-status "
                                  + "corrected"
                                )
                                : (
                                  "decision-status "
                                  + "accepted"
                                )
                            }
                          >
                            {
                              complaint
                                .was_corrected
                                ? (
                                  "AI tahmini "
                                  + "düzeltildi"
                                )
                                : (
                                  "AI tahmini "
                                  + "onaylandı"
                                )
                            }
                          </span>
                        </div>

                        <strong>
                          {
                            complaint
                              .final_category
                              ? categoryLabel(
                                  complaint
                                    .final_category,
                                )
                              : (
                                "Nihai kategori "
                                + "bulunamadı"
                              )
                          }
                        </strong>
                      </div>
                    )}
                  </article>
                );
              },
            )}
          </section>
        )}

        {(
          !loading
          && !error
          && totalComplaints > 0
        ) && (
          <section className="review-decision">
            <div className="review-decision-heading">
              <div>
                <span>
                  Sayfalama
                </span>

                <strong>
                  {
                    (
                      currentPage - 1
                    ) * PAGE_SIZE + 1
                  }
                  {"–"}
                  {Math.min(
                    currentPage * PAGE_SIZE,
                    totalComplaints,
                  )}
                  {" / "}
                  {totalComplaints}
                  {" kayıt"}
                </strong>
              </div>

              <span className="decision-status accepted">
                Sayfa {currentPage}
                {" / "}
                {Math.max(
                  totalPages,
                  1,
                )}
              </span>
            </div>

            <div className="review-decision-controls">
              <button
                type="button"
                className="review-submit-button"
                disabled={
                  currentPage <= 1
                  || loading
                }
                onClick={() => {
                  setCurrentPage(
                    (current) =>
                      Math.max(
                        1,
                        current - 1,
                      ),
                  );
                }}
              >
                ← Önceki
              </button>

              <button
                type="button"
                className="review-submit-button"
                disabled={
                  totalPages === 0
                  || currentPage >= totalPages
                  || loading
                }
                onClick={() => {
                  setCurrentPage(
                    (current) =>
                      current + 1,
                  );
                }}
              >
                Sonraki →
              </button>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}


export default OperatorPage;
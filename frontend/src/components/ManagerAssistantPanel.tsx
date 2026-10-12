import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  FormEvent,
} from "react";

import {
  useNavigate,
} from "react-router-dom";

import {
  askManagerAssistant,
} from "../api/managerAssistant";

import type {
  AnalyticsDateRange,
} from "../types/analytics";

import type {
  ManagerAssistantAnalysisType,
  ManagerAssistantComplaintCategory,
} from "../types/managerAssistant";

import "./ManagerAssistantPanel.css";


interface NearbyAnalysisContext {
  latitude: number;
  longitude: number;
  radiusMeters: number;
  category?: string;
}


interface ManagerAssistantPanelProps {
  dateRange: AnalyticsDateRange;
  nearbyContext:
    NearbyAnalysisContext | null;
}


interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  source?:
    | "gemini"
    | "deterministic_fallback";
  notice?: string;
}


interface QuickAnalysis {
  title: string;
  categoryLabel: string;
  description: string;
  prompt: string;
  analysisType:
    ManagerAssistantAnalysisType;
  featured?: boolean;
  requiresMapSelection?: boolean;
}


const QUICK_ANALYSES: QuickAnalysis[] = [
  {
    title: "Yönetici İçgörüsü",
    categoryLabel: "LLM sentezi",
    description:
      "KPI, kategori, zaman, mahalle ve AI–insan sonuçlarını birlikte yorumla",
    prompt:
      "Seçili dönem için genel özet, kategori dağılımı, zaman trendi, mahalle dağılımı ve AI–insan karşılaştırmasını birlikte değerlendir. Yalnızca backend tarafından hesaplanan verilere dayanarak 3-5 maddelik yönetici içgörüsü oluştur. Birden fazla analitik bölüm arasında dikkat çekici ilişkileri belirt; ancak neden-sonuç ilişkisi, risk seviyesi veya veride bulunmayan açıklamalar uydurma.",
    analysisType: "executive_insight",
    featured: true,
  },
  {
    title: "Genel Özet",
    categoryLabel: "Başvuru",
    description:
      "Toplam, otomatik kabul ve insan inceleme durumu",
    prompt:
      "Seçili dönem için toplam başvuru sayısını, otomatik kategorize edilen kayıtları, insan incelemesine giren kayıtları ve insan düzeltme durumunu kısa bir yönetici özeti olarak ver.",
    analysisType: "summary",
  },
  {
    title: "Kategori Dağılımı",
    categoryLabel: "Hizmet",
    description:
      "Kategori sayılarını ve paylarını karşılaştır",
    prompt:
      "Seçili dönemde hizmet kategorilerinin başvuru sayılarını ve paylarını analitik araçtan alarak özetle. En yüksek ve en düşük başvuru sayısına sahip kategorileri belirt.",
    analysisType:
      "category_distribution",
  },
  {
    title: "Zaman Trendi",
    categoryLabel: "Zaman",
    description:
      "Toplam, günlük ortalama ve en yoğun gün",
    prompt:
      "Seçili dönem için başvuru zaman trendini analiz et. Analitik araçtan gelen toplam başvuru, takvim günü, aktif gün, günlük ortalama ve en yoğun gün bilgilerini kısa şekilde özetle.",
    analysisType: "trend",
  },
  {
    title: "Mahalle Dağılımı",
    categoryLabel: "Mahalle",
    description:
      "Mahalle sayılarını ve baskın kategorileri özetle",
    prompt:
      "Seçili dönemde mahalle bazlı başvuru dağılımını analitik araçtan alarak özetle. En fazla başvuru bulunan ilk mahalleleri, başvuru sayılarını, paylarını ve baskın kategorilerini belirt. Bunu nüfus veya alan yoğunluğu olarak yorumlama.",
    analysisType: "neighborhoods",
  },
  {
    title: "AI–İnsan Karşılaştırması",
    categoryLabel: "AI kontrolü",
    description:
      "İncelenen kayıtlardaki uyum ve düzeltmeler",
    prompt:
      "Seçili dönemde insan tarafından incelenmiş kayıtlar için AI–insan karar karşılaştırmasını analitik araçtan alarak özetle. İncelenen, onaylanan ve düzeltilen kayıtları, anlaşma oranını, düzeltme oranını ve varsa düzeltme akışlarını belirt. Bu oranları genel model doğruluğu olarak yorumlama.",
    analysisType: "model_human",
  },
  {
    title: "Yakın Çevre",
    categoryLabel: "Mekânsal",
    description:
      "Haritadaki nokta, yarıçap ve kategori filtresini kullan",
    prompt:
      "Haritada seçilen analiz noktasının çevresindeki başvuruları yakın çevre analitik aracıyla özetle. Toplam başvuru sayısını, en yakın başvuru mesafesini, baskın kategoriyi ve kategori dağılımını belirt. Bunu nüfus yoğunluğu veya risk seviyesi olarak yorumlama.",
    analysisType: "nearby",
    requiresMapSelection: true,
  },
];


function isAssistantCategory(
  value: string | undefined,
): value is ManagerAssistantComplaintCategory {
  return (
    value === "TEMIZLIK_ATIK"
    || value === "YOL_KALDIRIM"
    || value === "PARK_YESIL_ALAN"
    || value === "KENT_DUZENI_DENETIM"
    || value === "HAYVAN_VETERINER"
  );
}


const SUPPORTED_ITEMS = [
  "KPI ve oran hesapları",
  "Kategori dağılımı",
  "Zaman trendi",
  "Mahalle istatistikleri",
  "AI–insan karşılaştırması",
  "PostGIS yakın çevre analizi",
];


const UNSUPPORTED_ITEMS = [
  "Tekil şikâyet metinlerini serbestçe taramaz",
  "Personel / vatandaş kişisel verilerine erişmez",
  "Serbest SQL çalıştırmaz",
  "Ham veritabanına doğrudan erişmez",
];


function cleanAssistantText(
  value: string,
): string {
  return value
    .replace(
      /\$\s*\\rightarrow\s*\$/g,
      "→",
    )
    .replace(
      /\\rightarrow/g,
      "→",
    )
    .replace(
      /^#{1,6}\s+/gm,
      "",
    )
    .replace(
      /^\s*---+\s*$/gm,
      "",
    )
    .replace(
      /^\s*[-*+]\s+/gm,
      "• ",
    )
    .replace(
      /\*\*([^*\n]+)\*\*/g,
      "$1",
    )
    .replace(
      /__([^_\n]+)__/g,
      "$1",
    )
    .replace(
      /\*([^*\n]+)\*/g,
      "$1",
    )
    .replace(
      /`([^`\n]+)`/g,
      "$1",
    )
    .replace(
      /\n{3,}/g,
      "\n\n",
    )
    .trim();
}


function rangeLabel(
  range: AnalyticsDateRange,
): string {
  if (
    !range.fromDate
    && !range.toDate
  ) {
    return "Tüm kayıtlar";
  }

  return [
    range.fromDate
      ?? "İlk kayıt",
    range.toDate
      ?? "Bugün",
  ].join(" → ");
}


function buildAssistantMessage(
  question: string,
  range: AnalyticsDateRange,
): string {
  if (
    !range.fromDate
    && !range.toDate
  ) {
    return question;
  }

  const fromDate =
    range.fromDate
    ?? "ilk kayıt";

  const toDate =
    range.toDate
    ?? "bugün";

  return (
    "Aktif yönetici dashboard tarih filtresi: "
    + `${fromDate} - ${toDate}. `
    + "Kullanıcı 'bu dönem', 'seçili dönem' veya benzeri "
    + "bir ifade kullanırsa bu tarih aralığını varsayılan "
    + "analitik bağlam olarak kullan. Kullanıcı sorusunda "
    + "açıkça başka bir tarih aralığı verilirse kullanıcı "
    + "tarafından belirtilen tarihi tercih et. "
    + `Kullanıcı sorusu: ${question}`
  );
}


function formatRadiusLabel(
  radiusMeters: number,
): string {
  if (radiusMeters < 1000) {
    return `${radiusMeters} m`;
  }

  return `${radiusMeters / 1000} km`;
}


function ManagerAssistantPanel({
  dateRange,
  nearbyContext,
}: ManagerAssistantPanelProps) {
  const navigate =
    useNavigate();

  const messagesEndRef =
    useRef<HTMLDivElement | null>(
      null,
    );

  const [
    input,
    setInput,
  ] = useState("");

  const [
    messages,
    setMessages,
  ] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      text: (
        "Bir hızlı analiz seçin veya aşağıdaki kapsam "
        + "dahilinde analitik bir soru sorun. Sayısal "
        + "sonuçlar backend analitik servislerinden alınır."
      ),
    },
  ]);
  const nextMessageIdRef = useRef(2);
  const [
    sending,
    setSending,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );

  const activeRangeLabel =
    useMemo(
      () => rangeLabel(
        dateRange,
      ),
      [
        dateRange,
      ],
    );


  useEffect(() => {
    messagesEndRef.current
      ?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
  }, [
    messages,
    sending,
  ]);


  async function sendQuestion(
    rawQuestion: string,
    quickAnalysis?: QuickAnalysis,
  ) {
    const question =
      rawQuestion.trim();

    if (
      !question
      || sending
    ) {
      return;
    }

    if (question.length > 2000) {
      setError(
        "Soru en fazla 2000 karakter olabilir.",
      );

      return;
    }

    if (
      quickAnalysis?.requiresMapSelection
      && !nearbyContext
    ) {
      setError(
        "Yakın çevre analizi için önce haritadan bir analiz noktası seçin.",
      );
      return;
    }

    const userMessage: ChatMessage = {
      id: nextMessageIdRef.current++,
      role: "user",
      text:
        quickAnalysis?.analysisType
        === "nearby"
        && nearbyContext
          ? (
            "Yakın Çevre · "
            + formatRadiusLabel(
              nearbyContext.radiusMeters,
            )
          )
          : (
            quickAnalysis?.title
            ?? question
          ),
    };

    setMessages([
      userMessage,
    ]);

    setInput("");
    setError(null);
    setSending(true);

    try {
      const response =
        await askManagerAssistant(
          quickAnalysis
            ? {
              message: question,
              analysis_type:
                quickAnalysis.analysisType,
              from_date:
                dateRange.fromDate
                || undefined,
              to_date:
                dateRange.toDate
                || undefined,
              latitude:
                quickAnalysis.analysisType
                === "nearby"
                  ? nearbyContext?.latitude
                  : undefined,
              longitude:
                quickAnalysis.analysisType
                === "nearby"
                  ? nearbyContext?.longitude
                  : undefined,
              radius_meters:
                quickAnalysis.analysisType
                === "nearby"
                  ? nearbyContext?.radiusMeters
                  : undefined,
              category:
                quickAnalysis.analysisType
                === "nearby"
                && isAssistantCategory(
                  nearbyContext?.category,
                )
                  ? nearbyContext?.category
                  : undefined,
            }
            : {
              message:
                buildAssistantMessage(
                  question,
                  dateRange,
                ),
            },
        );

      const assistantMessage:
        ChatMessage = {
          id: nextMessageIdRef.current++,
          role: "assistant",
          text: cleanAssistantText(
            response.answer,
          ),
          source:
            response.source,
          notice:
            response.notice,
        };

      setMessages(
        (current) => [
          ...current,
          assistantMessage,
        ],
      );
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : (
            "Yönetici analitik asistanı "
            + "yanıt veremedi."
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

        return;
      }

      setError(
        message,
      );
    } finally {
      setSending(false);
    }
  }


  function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    void sendQuestion(
      input,
    );
  }


  function clearConversation() {
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        text: (
          "Sonuç alanı temizlendi. "
          + "Yeni bir hazır analiz seçebilir "
          + "veya kapsam dahilinde soru sorabilirsiniz."
        ),
      },
    ]);

    setError(null);
  }


  return (
    <section className="manager-assistant-card">
      <div className="manager-assistant-hero">
        <div className="manager-assistant-title-block">
          <div className="manager-assistant-kicker">
            <span className="manager-assistant-status-dot" />
            Kontrollü karar destek mimarisi
          </div>

          <h2>
            LLM Destekli Yönetici Analiz Asistanı
          </h2>

          <p>
            Sayısal ve mekânsal sonuçlar backend analitik
            servisleri ve PostGIS tarafından hesaplanır.
            Gemini, bu doğrulanmış sonuçları yönetici için
            yorumlar ve özetler.
          </p>
        </div>

        <div className="manager-assistant-range">
          <span>
            Aktif analiz dönemi
          </span>

          <strong>
            {activeRangeLabel}
          </strong>
        </div>
      </div>


      <div className="manager-assistant-scope-grid">
        <div className="manager-assistant-scope-card supported">
          <div className="manager-assistant-scope-heading">
            <span className="manager-assistant-scope-icon">
              ✓
            </span>

            <div>
              <strong>
                Hesaplama katmanı
              </strong>

              <small>
                Sayılar LLM tarafından üretilmez
              </small>
            </div>
          </div>

          <div className="manager-assistant-scope-tags">
            {SUPPORTED_ITEMS.map(
              (item) => (
                <span key={item}>
                  {item}
                </span>
              ),
            )}
          </div>
        </div>

        <div className="manager-assistant-scope-card restricted">
          <div className="manager-assistant-scope-heading">
            <span className="manager-assistant-scope-icon">
              ×
            </span>

            <div>
              <strong>
                LLM sınırları
              </strong>

              <small>
                Gemini yalnızca izin verilen analitik çıktıları yorumlar
              </small>
            </div>
          </div>

          <ul>
            {UNSUPPORTED_ITEMS.map(
              (item) => (
                <li key={item}>
                  {item}
                </li>
              ),
            )}
          </ul>
        </div>
      </div>

      <div className="manager-assistant-architecture">
        <div>
          <span>1</span>
          <strong>Backend hesaplar</strong>
          <small>SQL, analitik servisler ve PostGIS</small>
        </div>

        <b>→</b>

        <div>
          <span>2</span>
          <strong>Gemini yorumlar</strong>
          <small>Doğal dilde yönetici özeti üretir</small>
        </div>

        <b>→</b>

        <div>
          <span>3</span>
          <strong>Sonuç gösterilir</strong>
          <small>Kaynak türü kullanıcıya açıkça belirtilir</small>
        </div>
      </div>


      <div className="manager-assistant-section-title">
        <div>
          <span>
            Hızlı analizler
          </span>

          <strong>
            Hazır yönetici analizleri
          </strong>
        </div>

        <small>
          Backend sonucu hesaplar; Gemini erişilebilirse yönetici yorumunu ekler
        </small>
      </div>


      <div className="manager-assistant-quick-grid">
        {QUICK_ANALYSES.map(
          (analysis) => (
            <button
              key={analysis.title}
              type="button"
              disabled={
                sending
                || (
                  analysis.requiresMapSelection
                  && !nearbyContext
                )
              }
              title={
                analysis.requiresMapSelection
                && !nearbyContext
                  ? (
                    "Önce yukarıdaki haritadan "
                    + "bir analiz noktası seçin."
                  )
                  : undefined
              }
              className={
                "manager-assistant-quick-card"
                + (analysis.featured
                  ? " featured"
                  : "")
              }
              onClick={() => {
                void sendQuestion(
                  analysis.prompt,
                  analysis,
                );
              }}
            >
              <div className="manager-assistant-quick-top">
                <span className="manager-assistant-quick-category">
                  {analysis.categoryLabel}
                </span>

                <span className="manager-assistant-quick-arrow">
                  →
                </span>
              </div>

              <strong>
                {analysis.analysisType
                  === "nearby"
                  && nearbyContext
                    ? (
                      "Yakın Çevre · "
                      + formatRadiusLabel(
                        nearbyContext.radiusMeters,
                      )
                    )
                    : analysis.title}
              </strong>

              <small>
                {analysis.requiresMapSelection
                  && !nearbyContext
                    ? (
                      "Önce haritadan bir analiz "
                      + "noktası seçin"
                    )
                    : analysis.description}
              </small>

              <span className="manager-assistant-quick-action">
                {analysis.requiresMapSelection
                  && !nearbyContext
                    ? "Harita seçimi gerekli"
                    : analysis.analysisType
                      === "executive_insight"
                        ? "İçgörüyü oluştur"
                        : "Özeti oluştur"}
              </span>
            </button>
          ),
        )}
      </div>


      <div className="manager-assistant-results-heading">
        <div>
          <span>
            Yönetici özeti
          </span>

          <strong>
            Son analiz sonucu
          </strong>
        </div>

        <button
          type="button"
          disabled={sending}
          onClick={
            clearConversation
          }
        >
          Sonucu temizle
        </button>
      </div>


      <div
        className="manager-assistant-messages"
        aria-live="polite"
      >
        {messages.map(
          (message) => (
            <article
              key={message.id}
              className={
                "manager-assistant-message "
                + (
                  message.role
                  === "user"
                    ? "user"
                    : "assistant"
                )
              }
            >
              <div className="manager-assistant-message-meta">
                <span>
                  {message.role
                    === "user"
                    ? "Seçilen analiz"
                    : "Özet"}
                </span>

                {message.role === "assistant"
                  && message.source && (
                    <span
                      className={
                        "manager-assistant-source "
                        + message.source
                      }
                    >
                      {message.source
                        === "gemini"
                        ? "Gemini yorumu"
                        : "Backend analitik sonucu"}
                    </span>
                  )}
              </div>

              <div>
                {message.text}
              </div>

              {message.notice && (
                <div className="manager-assistant-notice">
                  <strong>
                    LLM yorumu kullanılamadı
                  </strong>
                  <span>
                    {message.notice}
                  </span>
                </div>
              )}
            </article>
          ),
        )}

        {sending && (
          <article className="manager-assistant-message assistant">
            <span>
              Özet
            </span>

            <div className="manager-assistant-thinking">
              Veriler hazırlanıyor ve özetleniyor...
            </div>
          </article>
        )}

        <div
          ref={messagesEndRef}
          aria-hidden="true"
        />
      </div>


      {error && (
        <div className="manager-assistant-error">
          <strong>
            Analiz tamamlanamadı
          </strong>

          <span>
            {error}
          </span>
        </div>
      )}


      <form
        className="manager-assistant-form"
        onSubmit={
          handleSubmit
        }
      >
        <div className="manager-assistant-form-heading">
          <div>
            <span>
              Gemini destekli özel analitik soru
            </span>

            <strong>
              Kapsam dahilinde kendi sorunuzu yazın
            </strong>
          </div>

          <small>
            Serbest soru yalnızca izin verilen analitik araçlar kapsamında çalışır
          </small>
        </div>

        <textarea
          value={input}
          maxLength={2000}
          rows={3}
          disabled={sending}
          placeholder={
            "Örn. Seçili dönemde en fazla başvuru "
            + "bulunan mahalleleri ve baskın "
            + "kategorilerini özetle."
          }
          onChange={(event) => {
            setInput(
              event.target.value,
            );

            setError(null);
          }}
        />

        <div className="manager-assistant-form-footer">
          <span>
            {input.length} / 2000
          </span>

          <button
            className="manager-assistant-send"
            type="submit"
            disabled={
              sending
              || !input.trim()
            }
          >
            {sending
              ? "Analiz ediliyor..."
              : "Analizi Çalıştır"}
          </button>
        </div>
      </form>


      <div className="manager-assistant-footnote">
        <span>
          Kontrollü karar destek
        </span>

        <p>
          Sayısal ve mekânsal hesaplamalar deterministik backend
          servislerinde yapılır. Gemini bu hesaplanmış sonuçları
          doğal dilde yorumlayan karar destek katmanıdır; doğrudan
          SQL çalıştırmaz ve ham veritabanını serbestçe taramaz.
          Gemini geçici olarak kullanılamazsa yalnızca backend
          analitik sonucu gösterilir ve bu durum kullanıcıya açıkça
          belirtilir. Demo/sentetik veriler gerçek saha eğilimi
          olarak yorumlanmamalıdır.
        </p>
      </div>
    </section>
  );
}


export default ManagerAssistantPanel;

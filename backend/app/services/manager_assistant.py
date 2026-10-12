import json
from dataclasses import dataclass
from datetime import date

from google import genai
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.services.manager_analytics import (
    get_category_distribution,
    get_model_human_comparison,
    get_nearby_summary,
    get_neighborhood_stats,
    get_summary,
    get_trend_analysis,
)
from backend.app.models.enums import ComplaintCategory

SUMMARY_TOOL = {
    "type": "function",
    "name": "get_summary",
    "description": (
        "Returns deterministic summary statistics "
        "for municipal complaints. "
        "Use this tool whenever the manager asks "
        "about total complaints, automatic "
        "classification, human review, corrections "
        "or related summary rates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_date": {
                "type": "string",
                "description": (
                    "Optional start date in "
                    "YYYY-MM-DD format."
                ),
            },
            "to_date": {
                "type": "string",
                "description": (
                    "Optional end date in "
                    "YYYY-MM-DD format."
                ),
            },
        },
    },
}


CATEGORY_DISTRIBUTION_TOOL = {
    "type": "function",
    "name": "get_category_distribution",
    "description": (
        "Returns the deterministic distribution of "
        "municipal complaints by effective service category. "
        "Use this tool when the manager asks which categories "
        "have the most or fewest complaints, category counts, "
        "category shares, or category distribution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_date": {
                "type": "string",
                "description": (
                    "Optional start date in YYYY-MM-DD format."
                ),
            },
            "to_date": {
                "type": "string",
                "description": (
                    "Optional end date in YYYY-MM-DD format."
                ),
            },
        },
    },
}
TREND_ANALYSIS_TOOL = {
    "type": "function",
    "name": "get_trend_analysis",
    "description": (
        "Returns deterministic daily complaint trend data "
        "and precomputed trend statistics for a date range. "
        "Use this tool when the manager asks about daily "
        "complaint counts, total complaints in a period, "
        "daily average, active days, the busiest day, "
        "or general complaint activity over time. "
        "Do not calculate new numeric trend metrics yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_date": {
                "type": "string",
                "description": (
                    "Optional start date in YYYY-MM-DD format."
                ),
            },
            "to_date": {
                "type": "string",
                "description": (
                    "Optional end date in YYYY-MM-DD format."
                ),
            },
        },
    },
}

NEIGHBORHOOD_STATS_TOOL = {
    "type": "function",
    "name": "get_neighborhood_stats",
    "description": (
        "Returns deterministic municipal complaint statistics "
        "grouped by neighborhood. "
        "Use this tool when the manager asks which neighborhoods "
        "have the most or fewest complaints, neighborhood shares, "
        "dominant service categories by neighborhood, or complaint "
        "distribution across neighborhoods."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_date": {
                "type": "string",
                "description": (
                    "Optional start date in YYYY-MM-DD format."
                ),
            },
            "to_date": {
                "type": "string",
                "description": (
                    "Optional end date in YYYY-MM-DD format."
                ),
            },
        },
    },
}


MODEL_HUMAN_COMPARISON_TOOL = {
    "type": "function",
    "name": "get_model_human_comparison",
    "description": (
        "Returns deterministic comparison statistics between "
        "AI-predicted complaint categories and final human-reviewed "
        "categories. Use this tool when the manager asks about "
        "AI-human agreement, reviewed records, confirmed decisions, "
        "human corrections, correction rates, category-specific "
        "review outcomes, or correction flows. "
        "Agreement rate must not be described as model accuracy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_date": {
                "type": "string",
                "description": (
                    "Optional start date in YYYY-MM-DD format."
                ),
            },
            "to_date": {
                "type": "string",
                "description": (
                    "Optional end date in YYYY-MM-DD format."
                ),
            },
        },
    },
}
NEARBY_SUMMARY_TOOL = {
    "type": "function",
    "name": "get_nearby_summary",
    "description": (
        "Returns deterministic PostGIS-based summary statistics "
        "for municipal complaints within a radius around a "
        "latitude and longitude. Use this tool when the manager "
        "asks how many complaints are near a location, which "
        "service category is dominant nearby, category distribution "
        "within a radius, or the nearest complaint distance. "
        "Distances and counts are calculated by the backend."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Center latitude.",
            },
            "longitude": {
                "type": "number",
                "description": "Center longitude.",
            },
            "radius_meters": {
                "type": "number",
                "description": (
                    "Search radius in meters. "
                    "Must be between 1 and 10000."
                ),
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional complaint category filter."
                ),
                "enum": [
                    "TEMIZLIK_ATIK",
                    "YOL_KALDIRIM",
                    "PARK_YESIL_ALAN",
                    "KENT_DUZENI_DENETIM",
                    "HAYVAN_VETERINER",
                ],
            },
            "from_date": {
                "type": "string",
                "description": (
                    "Optional start date in YYYY-MM-DD format."
                ),
            },
            "to_date": {
                "type": "string",
                "description": (
                    "Optional end date in YYYY-MM-DD format."
                ),
            },
        },
        "required": [
            "latitude",
            "longitude",
        ],
    },
}

SYSTEM_INSTRUCTION = """
Sen Sarıçam Belediyesi yönetici karar destek
panelinde çalışan analitik asistansın.

Temel kurallar:

- Sayısal belediye istatistiklerini kendin üretme.
- Kullanılabilir bir analitik araç varsa veriyi
  mutlaka bu araçtan al.
- Veritabanına veya SQL'e doğrudan erişimin
  olduğunu iddia etme.
- Araçlardan dönen sonuçları yorumla.
- Araç çıktısında bulunmayan sayıları uydurma.
- Oranlar 0 ile 1 arasında döner. Kullanıcıya
  anlatırken uygun olduğunda yüzdeye çevir.
- correction_rate değerini model doğruluğu
  olarak tanımlama. Bu değer yalnızca insan
  tarafından incelenmiş kayıtlardaki düzeltme
  oranıdır.
- auto_acceptance_rate mevcut kayıt durumlarına
  göre hesaplanan operasyonel bir göstergedir.
- Cevaplarını Türkçe, kısa, açık ve yöneticiye
  faydalı olacak şekilde ver.
- Kullanıcı tarih aralığı belirtmezse aracın
  from_date ve to_date alanlarını boş bırak.
- Bir araç sonucu aldıktan sonra aynı veriyi
  gereksiz yere tekrar isteme.
- Araç çıktısında doğrudan bulunmayan yeni toplam,
  yüzde, oran, fark veya başka sayısal metrikleri
  kendin hesaplama. Böyle bir hesaplama gerekiyorsa
  uygun analitik aracı kullan.
- Kullanıcı açıkça öneri istemedikçe personel,
  bütçe veya kaynak dağılımı gibi operasyonel
  karar önerileri verme. Veriyi betimle ve dikkat
  çekici noktaları belirtmekle yetin.
- Günlük toplam, günlük ortalama, aktif gün sayısı
  ve en yoğun gün gibi zaman serisi metriklerinde
  get_trend_analysis aracının verdiği değerleri kullan.
- "Arttı", "azaldı" veya yüzde değişim gibi dönemler
  arası karşılaştırmaları kendin hesaplama. Böyle bir
  karşılaştırma için özel analitik araç yoksa bunu
  açıkça belirt.
- Mahalle istatistiklerini nüfus yoğunluğu, alan
  yoğunluğu veya kişi başına başvuru oranı olarak
  yorumlama. Araç yalnızca kayıtlı başvuru
  sayılarını ve bu başvurular içindeki payları verir.
- Bir mahalledeki başvuru sayısının yüksek olmasını
  tek başına o mahallenin daha sorunlu olduğu
  şeklinde yorumlama.
- Demo veya sentetik veriler kullanılıyorsa bunları
  gerçek belediye saha eğilimi gibi sunma.

- agreement_rate değerini model doğruluğu veya
  genel sınıflandırma başarısı olarak tanımlama.
  Bu oran yalnızca insan tarafından incelenmiş
  kayıtlar içindeki AI-insan anlaşma oranıdır.
- correction_rate yalnızca incelenmiş kayıtlar
  içindeki insan düzeltme oranıdır.
- İncelenmemiş kayıtlar hakkında AI-insan
  anlaşması veya doğruluk sonucu çıkarma.
- Kategori bazında çok az incelenmiş kayıt varsa
  oranları genelleştirme veya model performansı
  hakkında güçlü sonuç çıkarma.

- Yakın çevre analizlerinde mesafe, sayı, kategori
  dağılımı ve baskın kategori için get_nearby_summary
  aracının sonuçlarını kullan.
- Yakın çevre aracından dönmeyen yeni mesafe veya
  mekânsal metrikleri kendin hesaplama.
- Yakın çevre sonuçlarını nüfus yoğunluğu, sorun
  yoğunluğu veya gerçek saha risk seviyesi olarak
  yorumlama.
- get_nearby_summary ham şikâyet metni içermez;
  kullanıcı açıkça istemedikçe tekil başvuru içeriği
  hakkında çıkarım yapma.
- Bu asistan genel amaçlı sohbet botu değildir.
  Yalnızca mevcut analitik araçların kapsadığı yönetici
  analizleri hakkında yardımcı olur.
- Tekil şikâyet metni, en son başvurunun içeriği,
  vatandaş/personel kişisel verileri, serbest SQL veya
  ham veritabanı erişimi istenirse araç çağırmaya çalışma.
  Kısa biçimde bunun kapsam dışında olduğunu belirt ve
  desteklenen alanları söyle.
- "Açık ara", "kritik", "çok ciddi", "en sorunlu" gibi
  değerlendirme yüklü ifadeleri yalnızca araç çıktısı bunu
  doğrudan desteklemiyorsa kullanma; sayısal bulguyu tarafsız
  biçimde aktar.
""".strip()


PRESET_SYSTEM_INSTRUCTION = """
Sen Sarıçam Belediyesi yönetici karar destek panelinde
çalışan kontrollü analitik asistansın.

Sana backend tarafından önceden hesaplanmış deterministik
bir analitik JSON sonucu verilecek.

Kurallar:
- Yalnızca verilen JSON sonucunu açıkla.
- Yeni sayı, oran, fark veya metrik hesaplama.
- JSON'da bulunmayan bilgi uydurma.
- Cevabı Türkçe, kısa, açık ve yönetici odaklı ver.
- Demo/sentetik verileri gerçek saha eğilimi gibi sunma.
- Mahalle sayılarını nüfus veya alan yoğunluğu olarak
  yorumlama.
- AI-insan agreement_rate değerini genel model doğruluğu
  olarak tanımlama.
- correction_rate yalnızca incelenmiş kayıtlardaki insan
  düzeltme oranıdır.
- Yakın çevre sonucunu risk veya nüfus yoğunluğu olarak
  yorumlama.
- Teknik kategori ve durum kodlarını kullanıcıya ham biçimde
  gösterme. Örneğin TEMIZLIK_ATIK yerine "Temizlik / Atık",
  YOL_KALDIRIM yerine "Yol / Kaldırım", AUTO_ACCEPTED yerine
  "Otomatik Kategorize" de.
- Analiz türü executive_insight ise summary, category_distribution,
  trend, neighborhoods ve model_human bölümlerini birlikte değerlendir.
  Birden fazla bölümün desteklediği ilişkileri yönetici açısından
  öne çıkar; ancak neden-sonuç ilişkisi, risk seviyesi veya veri içinde
  bulunmayan açıklamalar uydurma.
- executive_insight cevabını kısa bir "Yönetici İçgörüsü" özeti ve
  3-5 maddelik bulgular şeklinde ver. Her bulgu hangi sayısal çıktıya
  dayandığını açıkça belli etsin.
- Gereksiz güçlü nitelemeler kullanma; bulguları tarafsız
  biçimde aktar.
""".strip()


class ManagerAssistantError(RuntimeError):
    pass


class ManagerAssistantRateLimitError(
    ManagerAssistantError
):
    pass


@dataclass(frozen=True)
class ManagerAssistantResult:
    answer: str
    source: str
    notice: str | None = None


def _is_rate_limit_error(
    exc: BaseException,
) -> bool:
    current: BaseException | None = exc

    for _ in range(5):
        if current is None:
            break

        class_name = (
            current.__class__.__name__
            .lower()
        )
        message = str(
            current
        ).lower()

        status_code = getattr(
            current,
            "status_code",
            None,
        )

        code = getattr(
            current,
            "code",
            None,
        )

        if (
            "ratelimiterror" in class_name
            or status_code == 429
            or code == 429
            or (
                "429" in message
                and (
                    "quota" in message
                    or "too_many_requests"
                    in message
                    or "rate limit"
                    in message
                )
            )
        ):
            return True

        current = (
            current.__cause__
            or current.__context__
        )

    return False


def _category_label(
    value: object,
) -> str:
    labels = {
        "TEMIZLIK_ATIK":
            "Temizlik / Atık",
        "YOL_KALDIRIM":
            "Yol / Kaldırım",
        "PARK_YESIL_ALAN":
            "Park / Yeşil Alan",
        "KENT_DUZENI_DENETIM":
            "Kent Düzeni / Denetim",
        "HAYVAN_VETERINER":
            "Hayvan / Veteriner",
    }

    normalized = str(
        value
    )

    return labels.get(
        normalized,
        normalized,
    )


def _percentage_text(
    value: object,
) -> str:
    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return "-"

    return f"%{numeric * 100:.1f}"


def _format_preset_fallback(
    analysis_type: str,
    result: object,
) -> str:
    if (
        analysis_type == "summary"
        and isinstance(result, dict)
    ):
        return (
            "Deterministik yönetici özeti:\n"
            f"• Toplam başvuru: "
            f"{result.get('total_complaints', 0)}\n"
            f"• Otomatik kategorize: "
            f"{result.get('auto_accepted', 0)} "
            f"({_percentage_text(result.get('auto_acceptance_rate'))})\n"
            f"• İnsan incelemesi: "
            f"{result.get('human_review_total', result.get('reviewed', 0))} "
            f"({_percentage_text(result.get('human_review_rate'))})\n"
            f"• İnceleme bekleyen: "
            f"{result.get('review_pending', 0)}\n"
            f"• İnsan tarafından düzeltilen: "
            f"{result.get('corrected_by_human', result.get('corrected', 0))} "
            f"(incelenenlerde "
            f"{_percentage_text(result.get('correction_rate'))})"
        )

    if (
        analysis_type
        == "category_distribution"
        and isinstance(result, list)
    ):
        lines = [
            "Deterministik kategori dağılımı:"
        ]

        for item in result:
            if not isinstance(item, dict):
                continue

            lines.append(
                "• "
                + _category_label(
                    item.get("category")
                )
                + ": "
                + str(
                    item.get(
                        "count",
                        0,
                    )
                )
                + " başvuru ("
                + _percentage_text(
                    item.get(
                        "percentage"
                    )
                )
                + ")"
            )

        return "\n".join(
            lines
        )

    if (
        analysis_type == "trend"
        and isinstance(result, dict)
    ):
        return (
            "Deterministik zaman özeti:\n"
            f"• Dönem: "
            f"{result.get('from_date', '-')} → "
            f"{result.get('to_date', '-')}\n"
            f"• Toplam başvuru: "
            f"{result.get('total_complaints', 0)}\n"
            f"• Takvim günü: "
            f"{result.get('calendar_days', 0)}\n"
            f"• Aktif gün: "
            f"{result.get('active_days', 0)}\n"
            f"• Günlük ortalama: "
            f"{result.get('daily_average', 0):.2f}\n"
            f"• En yoğun gün: "
            f"{result.get('busiest_day', '-')} "
            f"({result.get('busiest_day_count', 0)} başvuru)"
        )

    if (
        analysis_type == "neighborhoods"
        and isinstance(result, list)
    ):
        lines = [
            "Deterministik mahalle özeti:"
        ]

        for item in result[:5]:
            if not isinstance(item, dict):
                continue

            lines.append(
                "• "
                + str(
                    item.get(
                        "neighborhood",
                        "-"
                    )
                )
                + ": "
                + str(
                    item.get(
                        "count",
                        0,
                    )
                )
                + " başvuru ("
                + _percentage_text(
                    item.get(
                        "percentage"
                    )
                )
                + "), baskın kategori: "
                + _category_label(
                    item.get(
                        "dominant_category",
                        "-"
                    )
                )
            )

        lines.append(
            "Not: Bunlar ham başvuru sayılarıdır; "
            "nüfus veya alan yoğunluğu değildir."
        )

        return "\n".join(
            lines
        )

    if (
        analysis_type == "model_human"
        and isinstance(result, dict)
    ):
        lines = [
            "Deterministik AI–insan karşılaştırması:",
            (
                "• İncelenen: "
                + str(
                    result.get(
                        "reviewed",
                        0,
                    )
                )
            ),
            (
                "• AI tahmini onaylanan: "
                + str(
                    result.get(
                        "confirmed",
                        0,
                    )
                )
            ),
            (
                "• AI tahmini değiştirilen: "
                + str(
                    result.get(
                        "corrected",
                        0,
                    )
                )
            ),
            (
                "• AI–insan uyum oranı: "
                + _percentage_text(
                    result.get(
                        "agreement_rate"
                    )
                )
            ),
            (
                "• İnsan düzeltme oranı: "
                + _percentage_text(
                    result.get(
                        "correction_rate"
                    )
                )
            ),
            (
                "Not: Uyum oranı yalnızca insan "
                "tarafından incelenmiş kayıtları kapsar; "
                "genel model doğruluğu değildir."
            ),
        ]

        flows = result.get(
            "correction_flows",
            [],
        )

        if isinstance(
            flows,
            list,
        ) and flows:
            lines.append(
                "Düzeltme akışları:"
            )

            for flow in flows:
                if not isinstance(
                    flow,
                    dict,
                ):
                    continue

                lines.append(
                    "• "
                    + _category_label(
                        flow.get(
                            "predicted_category"
                        )
                    )
                    + " → "
                    + _category_label(
                        flow.get(
                            "final_category"
                        )
                    )
                    + ": "
                    + str(
                        flow.get(
                            "count",
                            0,
                        )
                    )
                    + " kayıt"
                )

        return "\n".join(
            lines
        )

    if (
        analysis_type == "executive_insight"
        and isinstance(result, dict)
    ):
        sections = [
            "Yönetici İçgörüsü için deterministik analitik paket:",
            "",
            _format_preset_fallback(
                "summary",
                result.get("summary", {}),
            ),
            "",
            _format_preset_fallback(
                "category_distribution",
                result.get("category_distribution", []),
            ),
            "",
            _format_preset_fallback(
                "trend",
                result.get("trend", {}),
            ),
            "",
            _format_preset_fallback(
                "neighborhoods",
                result.get("neighborhoods", []),
            ),
            "",
            _format_preset_fallback(
                "model_human",
                result.get("model_human", {}),
            ),
            "",
            (
                "Not: Bu paket backend tarafından hesaplanan ayrı "
                "analitik sonuçları birlikte gösterir. Gemini "
                "kullanılamadığında çapraz yorum üretilmez."
            ),
        ]

        return "\n".join(sections)

    if (
        analysis_type == "nearby"
        and isinstance(result, dict)
    ):
        lines = [
            "Deterministik yakın çevre özeti:",
            (
                "• Yarıçap: "
                + str(
                    result.get(
                        "radius_meters",
                        "-"
                    )
                )
                + " m"
            ),
            (
                "• Toplam başvuru: "
                + str(
                    result.get(
                        "count",
                        0,
                    )
                )
            ),
            (
                "• En yakın başvuru mesafesi: "
                + str(
                    result.get(
                        "nearest_distance_meters",
                        "-"
                    )
                )
                + " m"
            ),
            (
                "• Baskın kategori: "
                + _category_label(
                    result.get(
                        "dominant_category",
                        "-"
                    )
                )
                + " ("
                + str(
                    result.get(
                        "dominant_category_count",
                        0,
                    )
                )
                + " kayıt, "
                + _percentage_text(
                    result.get(
                        "dominant_category_percentage"
                    )
                )
                + ")"
            ),
        ]

        distribution = result.get(
            "category_distribution",
            [],
        )

        if isinstance(
            distribution,
            list,
        ) and distribution:
            lines.append(
                "Kategori dağılımı:"
            )

            for item in distribution:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                lines.append(
                    "• "
                    + _category_label(
                        item.get(
                            "category"
                        )
                    )
                    + ": "
                    + str(
                        item.get(
                            "count",
                            0,
                        )
                    )
                    + " ("
                    + _percentage_text(
                        item.get(
                            "percentage"
                        )
                    )
                    + ")"
                )

        lines.append(
            "Not: Bu sonuç nüfus yoğunluğu veya "
            "risk seviyesi olarak yorumlanmamalıdır."
        )

        return "\n".join(
            lines
        )

    return (
        "Deterministik analitik sonuç:\n"
        + json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _parse_optional_date(
    value: object,
    field_name: str,
) -> date | None:
    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ManagerAssistantError(
            f"{field_name} must be a date string."
        )

    normalized = value.strip()

    if not normalized:
        return None

    try:
        return date.fromisoformat(
            normalized
        )
    except ValueError as exc:
        raise ManagerAssistantError(
            (
                f"{field_name} must use "
                "YYYY-MM-DD format."
            )
        ) from exc
def _parse_optional_category(
    value: object,
) -> ComplaintCategory | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ManagerAssistantError(
            "category must be a string."
        )

    normalized = value.strip()

    if not normalized:
        return None

    try:
        return ComplaintCategory(
            normalized
        )
    except ValueError as exc:
        raise ManagerAssistantError(
            (
                "Unsupported complaint "
                f"category: {normalized}"
            )
        ) from exc

class ManagerAssistant:
    def __init__(self) -> None:
        api_key = settings.GEMINI_API_KEY

        if not api_key:
            raise ManagerAssistantError(
                "GEMINI_API_KEY is not configured."
            )

        self.model = settings.GEMINI_MODEL

        self.client = genai.Client(
            api_key=api_key
        )

    def _run_preset_analysis(
        self,
        analysis_type: str,
        db: Session,
        from_date: date | None = None,
        to_date: date | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_meters: float = 1000,
        category: ComplaintCategory | None = None,
    ) -> object:
        try:
            if analysis_type == "summary":
                return get_summary(
                    db=db,
                    from_date=from_date,
                    to_date=to_date,
                )

            if (
                analysis_type
                == "category_distribution"
            ):
                return get_category_distribution(
                    db=db,
                    from_date=from_date,
                    to_date=to_date,
                )

            if analysis_type == "trend":
                return get_trend_analysis(
                    db=db,
                    from_date=from_date,
                    to_date=to_date,
                )

            if analysis_type == "neighborhoods":
                return get_neighborhood_stats(
                    db=db,
                    from_date=from_date,
                    to_date=to_date,
                )

            if analysis_type == "model_human":
                return get_model_human_comparison(
                    db=db,
                    from_date=from_date,
                    to_date=to_date,
                )

            if analysis_type == "executive_insight":
                return {
                    "summary": get_summary(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    ),
                    "category_distribution": (
                        get_category_distribution(
                            db=db,
                            from_date=from_date,
                            to_date=to_date,
                        )
                    ),
                    "trend": get_trend_analysis(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    ),
                    "neighborhoods": get_neighborhood_stats(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    ),
                    "model_human": get_model_human_comparison(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    ),
                }

            if analysis_type == "nearby":
                if (
                    latitude is None
                    or longitude is None
                ):
                    raise ValueError(
                        "Nearby preset requires "
                        "latitude and longitude."
                    )

                return get_nearby_summary(
                    db=db,
                    latitude=latitude,
                    longitude=longitude,
                    radius_meters=radius_meters,
                    category=category,
                    from_date=from_date,
                    to_date=to_date,
                )

            raise ValueError(
                (
                    "Unsupported preset analysis type: "
                    f"{analysis_type}"
                )
            )

        except ValueError as exc:
            raise ManagerAssistantError(
                str(exc)
            ) from exc


    def ask_preset(
        self,
        analysis_type: str,
        message: str,
        db: Session,
        from_date: date | None = None,
        to_date: date | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_meters: float = 1000,
        category: ComplaintCategory | None = None,
    ) -> ManagerAssistantResult:
        normalized_message = (
            message.strip()
        )

        if not normalized_message:
            raise ValueError(
                "Message cannot be empty."
            )

        result = self._run_preset_analysis(
            analysis_type=analysis_type,
            db=db,
            from_date=from_date,
            to_date=to_date,
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_meters,
            category=category,
        )

        fallback_answer = (
            _format_preset_fallback(
                analysis_type,
                result,
            )
        )

        input_text = (
            "Yönetici isteği:\n"
            + normalized_message
            + "\n\n"
            + "Analiz türü:\n"
            + analysis_type
            + "\n\n"
            + "Deterministik backend sonucu:\n"
            + json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )
        )

        try:
            interaction = (
                self.client
                .interactions
                .create(
                    model=self.model,
                    input=input_text,
                    system_instruction=(
                        PRESET_SYSTEM_INSTRUCTION
                    ),
                )
            )

        except Exception as exc:
            if _is_rate_limit_error(
                exc
            ):
                return ManagerAssistantResult(
                    answer=fallback_answer,
                    source=(
                        "deterministic_fallback"
                    ),
                    notice=(
                        "Gemini geçici istek sınırına "
                        "ulaştığı için deterministik "
                        "analiz sonucu gösteriliyor."
                    ),
                )

            return ManagerAssistantResult(
                answer=fallback_answer,
                source=(
                    "deterministic_fallback"
                ),
                notice=(
                    "Gemini yorum katmanına "
                    "ulaşılamadığı için deterministik "
                    "analiz sonucu gösteriliyor."
                ),
            )

        output_text = (
            interaction.output_text
            or ""
        ).strip()

        if not output_text:
            return ManagerAssistantResult(
                answer=fallback_answer,
                source=(
                    "deterministic_fallback"
                ),
                notice=(
                    "Gemini boş yanıt verdiği için "
                    "deterministik analiz sonucu "
                    "gösteriliyor."
                ),
            )

        return ManagerAssistantResult(
            answer=output_text,
            source="gemini",
        )


    def ask(
        self,
        message: str,
        db: Session | None = None,
    ) -> str:
        normalized_message = (
            message.strip()
        )

        if not normalized_message:
            raise ValueError(
                "Message cannot be empty."
            )

        try:
            interaction = (
                self.client
                .interactions
                .create(
                    model=self.model,
                    input=normalized_message,
                    system_instruction=(
                        SYSTEM_INSTRUCTION
                    ),
                    tools=[
                        SUMMARY_TOOL,
                        CATEGORY_DISTRIBUTION_TOOL,
                        TREND_ANALYSIS_TOOL,
                        NEIGHBORHOOD_STATS_TOOL,
                        MODEL_HUMAN_COMPARISON_TOOL,
                        NEARBY_SUMMARY_TOOL,
                    ],
                )
            )
        except Exception as exc:
            if _is_rate_limit_error(
                exc
            ):
                raise ManagerAssistantRateLimitError(
                    "Gemini rate limit exceeded."
                ) from exc

            raise ManagerAssistantError(
                "Gemini request failed."
            ) from exc

        function_calls = [
            step
            for step in interaction.steps
            if step.type
            == "function_call"
        ]

        if not function_calls:
            output_text = (
                interaction.output_text
                or ""
            ).strip()

            if not output_text:
                raise ManagerAssistantError(
                    (
                        "Gemini returned an "
                        "empty response."
                    )
                )

            return output_text

        if db is None:
            raise ManagerAssistantError(
                (
                    "A database session is "
                    "required for analytics tools."
                )
            )

        function_results = []

        for call in function_calls:
            arguments = (
                call.arguments
                or {}
            )

            from_date = _parse_optional_date(
                arguments.get(
                    "from_date"
                ),
                "from_date",
            )

            to_date = _parse_optional_date(
                arguments.get(
                    "to_date"
                ),
                "to_date",
            )

            try:
                if call.name == "get_summary":
                    result = get_summary(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    )

                elif (
                    call.name
                    == "get_category_distribution"
                ):
                    result = (
                        get_category_distribution(
                            db=db,
                            from_date=from_date,
                            to_date=to_date,
                        )
                    )
                elif (
                    call.name
                    == "get_trend_analysis"
                ):
                    result = get_trend_analysis(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    )

                elif (
                    call.name
                    == "get_neighborhood_stats"
                ):
                    result = get_neighborhood_stats(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    )

                elif (
                    call.name
                    == "get_model_human_comparison"
                ):
                    result = get_model_human_comparison(
                        db=db,
                        from_date=from_date,
                        to_date=to_date,
                    )

                elif (
                    call.name
                    == "get_nearby_summary"
                ):
                    try:
                        latitude = float(
                            arguments["latitude"]
                        )

                        longitude = float(
                            arguments["longitude"]
                        )

                        radius_meters = float(
                            arguments.get(
                                "radius_meters",
                                1000,
                            )
                        )

                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise ManagerAssistantError(
                            (
                                "Nearby analysis requires "
                                "valid latitude and longitude."
                            )
                        ) from exc

                    category = _parse_optional_category(
                        arguments.get(
                            "category"
                        )
                    )

                    result = get_nearby_summary(
                        db=db,
                        latitude=latitude,
                        longitude=longitude,
                        radius_meters=radius_meters,
                        category=category,
                        from_date=from_date,
                        to_date=to_date,
                    )

                else:
                    raise ManagerAssistantError(
                        (
                            "Unsupported tool requested: "
                            f"{call.name}"
                        )
                    )

            except ValueError as exc:
                raise ManagerAssistantError(
                    str(exc)
                ) from exc

            function_results.append(
                {
                    "type":
                        "function_result",

                    "name":
                        call.name,

                    "call_id":
                        call.id,

                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                }
            )

        try:
            final_interaction = (
                self.client
                .interactions
                .create(
                    model=self.model,
                    previous_interaction_id=(
                        interaction.id
                    ),
                    input=function_results,
                    tools=[
                        SUMMARY_TOOL,
                        CATEGORY_DISTRIBUTION_TOOL,
                        TREND_ANALYSIS_TOOL,
                        NEIGHBORHOOD_STATS_TOOL,
                        MODEL_HUMAN_COMPARISON_TOOL,
                        NEARBY_SUMMARY_TOOL,
                    ],
                    system_instruction=(
                        SYSTEM_INSTRUCTION
                    ),
                )
            )
        except Exception as exc:
            if _is_rate_limit_error(
                exc
            ):
                raise ManagerAssistantRateLimitError(
                    "Gemini rate limit exceeded."
                ) from exc

            raise ManagerAssistantError(
                (
                    "Gemini failed while "
                    "processing analytics results."
                )
            ) from exc

        output_text = (
            final_interaction.output_text
            or ""
        ).strip()

        if not output_text:
            raise ManagerAssistantError(
                (
                    "Gemini returned an empty "
                    "analytics response."
                )
            )

        return output_text

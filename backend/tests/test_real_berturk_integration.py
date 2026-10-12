from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.complaint import (
    Complaint,
)


def test_real_berturk_complaint_flow(
    real_client,
):
    # -------------------------------------------------
    # 1. Gerçek model gerçekten yüklenmiş mi?
    # -------------------------------------------------

    health_response = real_client.get(
        "/health"
    )

    assert (
        health_response.status_code
        == 200
    )

    health = health_response.json()

    assert (
        health["model"]["loaded"]
        is True
    )

    assert (
        health["model"]["version"]
        == "berturk_final_v2_99ea9e29"
    )

    assert (
        health["database"]["name"]
        == "belediye_ai_test"
    )

    # CPU veya CUDA ile çalışabilir.
    assert health["model"]["device"] in (
        "cpu",
        "cuda",
    )

    # -------------------------------------------------
    # 2. Yüksek güvenli gerçek BERTurk tahmini
    # -------------------------------------------------

    automatic_response = real_client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Sokağımızdaki çöp konteyneri "
                "günlerdir boşaltılmadı, "
                "çöpler etrafa yayılıyor ve "
                "kötü koku oluşuyor."
            ),
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert (
        automatic_response.status_code
        == 201
    )

    automatic = (
        automatic_response.json()
    )

    assert (
        automatic["predicted_category"]
        == "TEMIZLIK_ATIK"
    )

    assert (
        automatic["review_required"]
        is False
    )

    assert (
        automatic["status"]
        == "AUTO_ACCEPTED"
    )

    assert (
        automatic["model_version"]
        == "berturk_final_v2_99ea9e29"
    )

    assert (
        automatic["model_version"]
        != "test-model"
    )

    assert (
        0.0
        <= automatic["model_score"]
        <= 1.0
    )

    assert (
        0.0
        <= automatic["score_margin"]
        <= 1.0
    )

    # -------------------------------------------------
    # 3. Bilinen zor örnek:
    #    gerçek BERTurk + gerçek threshold
    # -------------------------------------------------

    review_response = real_client.post(
        "/complaints",
        json={
            "complaint_text": (
                "İşletmenin dışarıya koyduğu "
                "hoparlörlerden gelen yüksek ses "
                "akşam boyunca kesilmiyor; "
                "ses yükseldiğinde sahipsiz "
                "köpekler de ürküp koşuşturuyor. "
                "Gürültü konusunda zabıta "
                "kontrolü yapılmasını talep ederim."
            ),
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1111882,
            "longitude": 35.47325806957672,
        },
    )

    assert (
        review_response.status_code
        == 201
    )

    review = review_response.json()

    assert (
        review["predicted_category"]
        == "HAYVAN_VETERINER"
    )

    assert (
        review["secondary_category"]
        == "KENT_DUZENI_DENETIM"
    )

    assert (
        review["review_required"]
        is True
    )

    assert (
        review["review_reason"]
        == "LOW_CONFIDENCE_AND_CLOSE_TOP2"
    )

    assert (
        review["status"]
        == "REVIEW_REQUIRED"
    )

    assert (
        review["model_version"]
        == "berturk_final_v2_99ea9e29"
    )

    # Bu örnekte modelin gerçekten
    # kararsız davranmasını bekliyoruz.
    assert (
        review["model_score"]
        < 0.988985559
    )

    assert (
        review["score_margin"]
        < 0.981616926
    )

    # -------------------------------------------------
    # 4. Sonuçlar gerçekten test DB'ye yazıldı mı?
    # -------------------------------------------------

    db = SessionLocal()

    try:
        complaints = (
            db.query(Complaint)
            .order_by(Complaint.id)
            .all()
        )

        assert len(complaints) == 2

        assert (
            complaints[0].model_version
            == "berturk_final_v2_99ea9e29"
        )

        assert (
            complaints[1].model_version
            == "berturk_final_v2_99ea9e29"
        )

        assert (
            complaints[1].review_required
            is True
        )

    finally:
        db.close()
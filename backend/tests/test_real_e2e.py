from backend.app.core.security import hash_password
from backend.app.db.database import SessionLocal
from backend.app.models.enums import UserRole
from backend.app.models.user import User


def create_real_test_user(
    username: str,
    password: str,
    role: UserRole,
) -> None:
    db = SessionLocal()

    try:
        user = User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            is_active=True,
        )

        db.add(user)
        db.commit()

    finally:
        db.close()


def test_real_backend_end_to_end(
    real_client,
):
    # -----------------------------------------
    # 1. Gerçek operator ve manager oluştur
    # -----------------------------------------

    create_real_test_user(
        username="e2e_operator",
        password="Operator123!",
        role=UserRole.OPERATOR,
    )

    create_real_test_user(
        username="e2e_manager",
        password="Manager123!",
        role=UserRole.MANAGER,
    )

    # -----------------------------------------
    # 2. Login
    # -----------------------------------------

    operator_login = real_client.post(
        "/auth/login",
        json={
            "username": "e2e_operator",
            "password": "Operator123!",
        },
    )

    assert operator_login.status_code == 200

    operator_headers = {
        "Authorization":
            "Bearer "
            + operator_login.json()["access_token"]
    }

    manager_login = real_client.post(
        "/auth/login",
        json={
            "username": "e2e_manager",
            "password": "Manager123!",
        },
    )

    assert manager_login.status_code == 200

    manager_headers = {
        "Authorization":
            "Bearer "
            + manager_login.json()["access_token"]
    }

    # -----------------------------------------
    # 3. Gerçek BERTurk:
    #    yüksek güvenli complaint
    # -----------------------------------------

    auto_response = real_client.post(
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

    assert auto_response.status_code == 201

    auto_data = auto_response.json()

    assert (
        auto_data["predicted_category"]
        == "TEMIZLIK_ATIK"
    )

    assert (
        auto_data["review_required"]
        is False
    )

    assert (
        auto_data["status"]
        == "AUTO_ACCEPTED"
    )

    assert (
        auto_data["model_version"]
        == "berturk_final_v2_99ea9e29"
    )

    # -----------------------------------------
    # 4. Gerçek BERTurk:
    #    düşük güven / review complaint
    # -----------------------------------------

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

    assert review_response.status_code == 201

    review_data = review_response.json()

    assert (
        review_data["predicted_category"]
        == "HAYVAN_VETERINER"
    )

    assert (
        review_data["secondary_category"]
        == "KENT_DUZENI_DENETIM"
    )

    assert (
        review_data["review_required"]
        is True
    )

    assert (
        review_data["status"]
        == "REVIEW_REQUIRED"
    )

    review_id = review_data["id"]

    # -----------------------------------------
    # 5. Operator insan düzeltmesi
    # -----------------------------------------

    human_review = real_client.post(
        f"/complaints/{review_id}/review",
        headers=operator_headers,
        json={
            "final_category":
                "KENT_DUZENI_DENETIM"
        },
    )

    assert human_review.status_code == 200

    human_data = human_review.json()

    assert (
        human_data["predicted_category"]
        == "HAYVAN_VETERINER"
    )

    assert (
        human_data["final_category"]
        == "KENT_DUZENI_DENETIM"
    )

    assert (
        human_data["was_corrected"]
        is True
    )

    assert (
        human_data["status"]
        == "REVIEWED"
    )

    # -----------------------------------------
    # 6. Detay:
    #    AI tahmini gerçekten korunmuş mu?
    # -----------------------------------------

    detail_response = real_client.get(
        f"/complaints/{review_id}",
        headers=operator_headers,
    )

    assert detail_response.status_code == 200

    detail = detail_response.json()

    assert (
        detail["predicted_category"]
        == "HAYVAN_VETERINER"
    )

    assert (
        detail["final_category"]
        == "KENT_DUZENI_DENETIM"
    )

    assert (
        detail["reviewed_by"]
        is not None
    )

    # -----------------------------------------
    # 7. Manager analytics
    # -----------------------------------------

    summary_response = real_client.get(
        "/analytics/summary",
        headers=manager_headers,
    )

    assert summary_response.status_code == 200

    summary = summary_response.json()

    assert summary["total_complaints"] == 2
    assert summary["auto_accepted"] == 1
    assert summary["review_pending"] == 0
    assert summary["reviewed"] == 1
    assert summary["corrected_by_human"] == 1

    # -----------------------------------------
    # 8. Category analytics:
    #    insan final kararı kullanılmalı
    # -----------------------------------------

    categories_response = real_client.get(
        "/analytics/categories",
        headers=manager_headers,
    )

    assert categories_response.status_code == 200

    category_counts = {
        item["category"]: item["count"]
        for item in categories_response.json()
    }

    assert (
        category_counts["TEMIZLIK_ATIK"]
        == 1
    )

    assert (
        category_counts["KENT_DUZENI_DENETIM"]
        == 1
    )

    assert (
        "HAYVAN_VETERINER"
        not in category_counts
    )

    # -----------------------------------------
    # 9. Gerçek PostGIS ST_DWithin
    # -----------------------------------------

    nearby_response = real_client.get(
        (
            "/analytics/nearby"
            "?latitude=37.1110882"
            "&longitude=35.473158069576726"
            "&radius_meters=500"
            "&category=KENT_DUZENI_DENETIM"
        ),
        headers=manager_headers,
    )

    assert nearby_response.status_code == 200

    nearby = nearby_response.json()

    assert nearby["count"] == 1

    assert (
        nearby["results"][0]["id"]
        == review_id
    )

    assert (
        nearby["results"][0]["category"]
        == "KENT_DUZENI_DENETIM"
    )
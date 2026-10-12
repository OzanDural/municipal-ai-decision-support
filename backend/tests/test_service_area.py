from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.complaint import (
    Complaint,
)


def test_complaint_inside_saricam_is_accepted(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",
            "neighborhood":
                "Kemalpaşa",
            "latitude":
                37.1110882,

            "longitude":
                35.473158069576726,
        },
    )

    assert response.status_code == 201


def test_complaint_outside_saricam_is_rejected(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.0,
            "longitude": 35.0,
        },
    )

    assert response.status_code == 422

    assert (
        response.json()["detail"]
        == (
            "Selected location is outside "
            "the Sarıçam service area."
        )
    )


def test_outside_complaint_is_not_saved(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Bu kayıt veritabanına "
                "girmemelidir.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.0,
            "longitude": 35.0,
        },
    )

    assert response.status_code == 422

    db = SessionLocal()

    try:
        count = (
            db.query(Complaint)
            .count()
        )

        assert count == 0

    finally:
        db.close()
from backend.app.db.database import SessionLocal
from backend.app.models.complaint import Complaint
from backend.app.models.enums import (
    ComplaintCategory,
    ComplaintStatus,
)


def test_auto_accepted_complaint_is_saved(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Sokağımızdaki çöp konteyneri "
                "günlerdir boşaltılmadı.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert (
        data["predicted_category"]
        == "TEMIZLIK_ATIK"
    )

    assert (
        data["review_required"]
        is False
    )

    assert (
        data["status"]
        == "AUTO_ACCEPTED"
    )

    assert (
        data["model_version"]
        == "test-model"
    )

    db = SessionLocal()

    try:
        complaint = db.get(
            Complaint,
            data["id"],
        )

        assert complaint is not None

        assert (
            complaint.predicted_category
            == ComplaintCategory.TEMIZLIK_ATIK
        )

        assert (
            complaint.status
            == ComplaintStatus.AUTO_ACCEPTED
        )

        assert (
            complaint.final_category
            is None
        )

    finally:
        db.close()


def test_low_confidence_complaint_enters_review_queue(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Bu şikâyet review testi için "
                "insan kontrolü gerektirmelidir.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1111882,
            "longitude": 35.47325806957672,
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert (
        data["predicted_category"]
        == "HAYVAN_VETERINER"
    )

    assert (
        data["secondary_category"]
        == "KENT_DUZENI_DENETIM"
    )

    assert (
        data["review_required"]
        is True
    )

    assert (
        data["review_reason"]
        == "LOW_CONFIDENCE_AND_CLOSE_TOP2"
    )

    assert (
        data["status"]
        == "REVIEW_REQUIRED"
    )


def test_operator_can_correct_review_required_complaint(
    client,
    operator_headers,
):
    create_response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Bu kayıt review süreci "
                "testi için oluşturulmuştur.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1111882,
            "longitude": 35.47325806957672,
        },
    )

    assert (
        create_response.status_code
        == 201
    )

    complaint_id = (
        create_response.json()["id"]
    )

    review_response = client.post(
        f"/complaints/{complaint_id}/review",
        headers=operator_headers,
        json={
            "final_category":
                "KENT_DUZENI_DENETIM"
        },
    )

    assert (
        review_response.status_code
        == 200
    )

    data = review_response.json()

    assert (
        data["predicted_category"]
        == "HAYVAN_VETERINER"
    )

    assert (
        data["final_category"]
        == "KENT_DUZENI_DENETIM"
    )

    assert (
        data["was_corrected"]
        is True
    )

    assert (
        data["review_required"]
        is False
    )

    assert (
        data["status"]
        == "REVIEWED"
    )

    db = SessionLocal()

    try:
        complaint = db.get(
            Complaint,
            complaint_id,
        )

        assert (
            complaint.predicted_category
            == ComplaintCategory.HAYVAN_VETERINER
        )

        assert (
            complaint.final_category
            == ComplaintCategory.KENT_DUZENI_DENETIM
        )

        assert (
            complaint.was_corrected
            is True
        )

        assert (
            complaint.review_required
            is False
        )

        assert (
            complaint.status
            == ComplaintStatus.REVIEWED
        )

        assert (
            complaint.reviewed_by
            is not None
        )

        assert (
            complaint.reviewed_at
            is not None
        )

    finally:
        db.close()


def test_reviewed_complaint_cannot_be_reviewed_twice(
    client,
    operator_headers,
):
    create_response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Bu kayıt review tekrar "
                "kontrolü için oluşturulmuştur.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1111882,
            "longitude": 35.47325806957672,
        },
    )

    assert (
        create_response.status_code
        == 201
    )

    complaint_id = (
        create_response.json()["id"]
    )

    first_review = client.post(
        f"/complaints/{complaint_id}/review",
        headers=operator_headers,
        json={
            "final_category":
                "KENT_DUZENI_DENETIM"
        },
    )

    assert (
        first_review.status_code
        == 200
    )

    second_review = client.post(
        f"/complaints/{complaint_id}/review",
        headers=operator_headers,
        json={
            "final_category":
                "TEMIZLIK_ATIK"
        },
    )

    assert (
        second_review.status_code
        == 409
    )

    assert (
        second_review.json()["detail"]
        == (
            "Complaint is not waiting "
            "for human review."
        )
    )
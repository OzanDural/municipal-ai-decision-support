from sqlalchemy import text

from backend.app.db.database import SessionLocal


def test_complaint_location_is_saved_as_postgis_point(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert response.status_code == 201

    complaint_id = response.json()["id"]

    db = SessionLocal()

    try:
        row = db.execute(
            text(
                """
                SELECT
                    ST_Y(location::geometry) AS latitude,
                    ST_X(location::geometry) AS longitude,
                    ST_SRID(location::geometry) AS srid,
                    GeometryType(location::geometry) AS geometry_type
                FROM complaints
                WHERE id = :complaint_id
                """
            ),
            {
                "complaint_id":
                    complaint_id
            },
        ).one()

        assert abs(
            float(row.latitude) - 37.1110882
        ) < 0.000001

        assert abs(
            float(row.longitude) - 35.473158069576726
        ) < 0.000001

        assert row.srid == 4326

        assert (
            row.geometry_type
            == "POINT"
        )

    finally:
        db.close()


def test_nearby_query_uses_real_postgis_distance(
    client,
    manager_headers,
):
    first = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    second = client.post(
        "/complaints",
        json={
            "complaint_text":
                "İkinci yakın kayıt.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1111882,
            "longitude": 35.47325806957672,
        },
    )

    assert first.status_code == 201
    assert second.status_code == 201

    response = client.get(
        (
            "/analytics/nearby"
            "?latitude=37.1110882"
            "&longitude=35.473158069576726"
            "&radius_meters=500"
        ),
        headers=manager_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2

    results = data["results"]

    assert results[0]["id"] == (
        first.json()["id"]
    )

    assert abs(
        results[0]["distance_meters"]
    ) < 0.01

    assert (
        results[1]["distance_meters"]
        > 0
    )

    assert (
        results[1]["distance_meters"]
        < 500
    )


def test_nearby_category_filter_uses_human_final_category(
    client,
    operator_headers,
    manager_headers,
):
    auto_response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    review_response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Bu kayıt review işlemi "
                "için oluşturulmuştur.",
            "neighborhood":
                "Kemalpaşa",
            "latitude": 37.1111882,
            "longitude": 35.47325806957672,
        },
    )

    assert auto_response.status_code == 201
    assert review_response.status_code == 201

    reviewed_id = (
        review_response.json()["id"]
    )

    human_review = client.post(
        f"/complaints/{reviewed_id}/review",
        headers=operator_headers,
        json={
            "final_category":
                "KENT_DUZENI_DENETIM"
        },
    )

    assert human_review.status_code == 200

    kent_response = client.get(
        (
            "/analytics/nearby"
            "?latitude=37.1110882"
            "&longitude=35.473158069576726"
            "&radius_meters=500"
            "&category=KENT_DUZENI_DENETIM"
        ),
        headers=manager_headers,
    )

    assert kent_response.status_code == 200

    kent_data = kent_response.json()

    assert kent_data["count"] == 1
    assert (
        kent_data["results"][0]["id"]
        == reviewed_id
    )

    hayvan_response = client.get(
        (
            "/analytics/nearby"
            "?latitude=37.1110882"
            "&longitude=35.473158069576726"
            "&radius_meters=500"
            "&category=HAYVAN_VETERINER"
        ),
        headers=manager_headers,
    )

    assert hayvan_response.status_code == 200

    hayvan_data = hayvan_response.json()

    assert hayvan_data["count"] == 0
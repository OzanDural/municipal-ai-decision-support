def test_public_can_get_saricam_neighborhoods(
    client,
):
    response = client.get(
        "/locations/neighborhoods"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["district_code"]
        == "SARICAM"
    )

    assert (
        body["district_name"]
        == "Sarıçam"
    )

    assert body["count"] == 69

    assert (
        len(body["neighborhoods"])
        == 69
    )


def test_neighborhood_list_is_canonical(
    client,
):
    response = client.get(
        "/locations/neighborhoods"
    )

    assert response.status_code == 200

    neighborhoods = (
        response.json()[
            "neighborhoods"
        ]
    )

    assert len(
        set(neighborhoods)
    ) == 69

    assert (
        "Kemalpaşa"
        in neighborhoods
    )

    assert (
        "Yeşiltepe"
        in neighborhoods
    )

    assert (
        "Elif Su Uludağ"
        in neighborhoods
    )

    assert (
        "Yeni Mahalle"
        in neighborhoods
    )

    assert (
        "Cumhuriyet (Buruk)"
        in neighborhoods
    )


def test_valid_canonical_neighborhood_is_accepted(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Çöp konteyneri "
                "günlerdir boşaltılmadı."
            ),
            "neighborhood": (
                "Kemalpaşa"
            ),
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert response.status_code == 201

    assert (
        response.json()[
            "neighborhood"
        ]
        == "Kemalpaşa"
    )


def test_invalid_neighborhood_is_rejected(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Çöp konteyneri "
                "günlerdir boşaltılmadı."
            ),
            "neighborhood": (
                "Uydurma Mahalle"
            ),
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert response.status_code == 422
def test_public_can_get_neighborhood_centers(
    client,
):
    response = client.get(
        "/locations/neighborhood-centers"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["district_code"]
        == "SARICAM"
    )

    assert (
        body["district_name"]
        == "Sarıçam"
    )

    centers = body[
        "centers"
    ]

    assert (
        body["count"]
        == len(centers)
    )

    assert (
        body["count"]
        == 69
    )

    names = [
        item["name"]
        for item in centers
    ]

    assert (
        len(names)
        == len(set(names))
    )

    balcalı = next(
        item
        for item in centers
        if item["name"]
        == "Balcalı"
    )

    assert (
        36 <= balcalı["latitude"] <= 38
    )

    assert (
        34 <= balcalı["longitude"] <= 37
    )
def test_missing_neighborhood_is_rejected(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",

            "latitude":
                37.1110882,

            "longitude":
                35.473158069576726,
        },
    )

    assert response.status_code == 422

    detail = response.json()[
        "detail"
    ]

    assert any(
        item.get("loc")
        == [
            "body",
            "neighborhood",
        ]
        for item in detail
    )
def test_missing_neighborhood_is_rejected(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text":
                "Çöp konteyneri boşaltılmadı.",

            "latitude":
                37.1110882,

            "longitude":
                35.473158069576726,
        },
    )

    assert response.status_code == 422

    detail = response.json()[
        "detail"
    ]

    assert any(
        item.get("loc")
        == [
            "body",
            "neighborhood",
        ]
        for item in detail
    )
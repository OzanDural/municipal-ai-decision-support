def test_public_can_get_saricam_streets(
    client,
):
    response = client.get(
        "/locations/streets"
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

    streets = body[
        "streets"
    ]

    assert body["count"] == len(
        streets
    )

    assert body["count"] > 0

    assert (
        len(streets)
        == len(set(streets))
    )

    assert (
        "Atatürk Caddesi"
        in streets
    )

    assert (
        "Mithat Özsan Bulvarı"
        in streets
    )


def test_canonical_street_suggestion_is_accepted(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Çöp konteyneri "
                "günlerdir boşaltılmadı."
            ),
            "neighborhood":
                "Kemalpaşa",
            "street_name":
                "Atatürk Caddesi",
            "latitude":
                37.1110882,
            "longitude":
                35.473158069576726,
        },
    )

    assert response.status_code == 201

    assert (
        response.json()[
            "street_name"
        ]
        == "Atatürk Caddesi"
    )


def test_manual_street_is_accepted(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Yolda büyük bir "
                "çukur bulunuyor."
            ),
            "neighborhood":
                "Kemalpaşa",
            "street_name":
                "4453 Sokak",
            "latitude":
                37.1110882,
            "longitude":
                35.473158069576726,
        },
    )

    assert response.status_code == 201

    assert (
        response.json()[
            "street_name"
        ]
        == "4453 Sokak"
    )


def test_street_suggestions_have_no_case_duplicates(
    client,
):
    response = client.get(
        "/locations/streets"
    )

    assert response.status_code == 200

    streets = response.json()[
        "streets"
    ]

    normalized = [
        street
        .casefold()
        .replace("\u0307", "")
        for street in streets
    ]

    assert (
        len(normalized)
        == len(set(normalized))
    )
def test_complaint_address_fields_are_saved_and_returned(
    client,
    operator_headers,
):
    create_response = client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Sokağımızdaki çöp konteyneri "
                "üç gündür boşaltılmadı."
            ),
            "neighborhood": (
                "Kemalpaşa"
            ),
            "street_name": (
                "Atatürk Caddesi"
            ),
            "address_text": (
                "Marketin karşısındaki "
                "konteynerin yanı."
            ),
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert (
        create_response.status_code
        == 201
    )

    created = (
        create_response.json()
    )

    assert (
        created["neighborhood"]
        == "Kemalpaşa"
    )

    assert (
        created["street_name"]
        == "Atatürk Caddesi"
    )

    assert (
        created["address_text"]
        == (
            "Marketin karşısındaki "
            "konteynerin yanı."
        )
    )

    complaint_id = (
        created["id"]
    )

    detail_response = client.get(
        f"/complaints/{complaint_id}",
        headers=operator_headers,
    )

    assert (
        detail_response.status_code
        == 200
    )

    detail = (
        detail_response.json()
    )

    assert (
        detail["neighborhood"]
        == "Kemalpaşa"
    )

    assert (
        detail["street_name"]
        == "Atatürk Caddesi"
    )

    assert (
        detail["address_text"]
        == (
            "Marketin karşısındaki "
            "konteynerin yanı."
        )
    )


def test_address_text_fields_are_normalized(
    client,
):
    response = client.post(
        "/complaints",
        json={
            "complaint_text": (
                "Parkın girişindeki "
                "çöp kutusu dolmuş."
            ),
            "neighborhood": (
                "   Kemalpaşa   "
            ),
            "street_name": (
                "   Atatürk    Caddesi   "
            ),
            "address_text": (
                "   Marketin   hemen "
                "karşısı   "
            ),
            "latitude": 37.1110882,
            "longitude": 35.473158069576726,
        },
    )

    assert (
        response.status_code
        == 201
    )

    body = (
        response.json()
    )

    assert (
        body["neighborhood"]
        == "Kemalpaşa"
    )

    assert (
        body["street_name"]
        == "Atatürk Caddesi"
    )

    assert (
        body["address_text"]
        == "Marketin hemen karşısı"
    )
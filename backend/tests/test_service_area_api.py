def test_public_can_get_saricam_service_area(
    client,
):
    response = client.get(
        "/service-areas/SARICAM"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["type"] == "Feature"

    assert (
        data["properties"]["code"]
        == "SARICAM"
    )

    assert (
        data["properties"]["name"]
        == "Sarıçam"
    )

    assert (
        data["geometry"]["type"]
        == "MultiPolygon"
    )

    assert (
        len(
            data["geometry"][
                "coordinates"
            ]
        )
        > 0
    )


def test_service_area_code_is_normalized(
    client,
):
    response = client.get(
        "/service-areas/saricam"
    )

    assert response.status_code == 200

    assert (
        response.json()[
            "properties"
        ]["code"]
        == "SARICAM"
    )


def test_unknown_service_area_returns_404(
    client,
):
    response = client.get(
        "/service-areas/UNKNOWN"
    )

    assert (
        response.status_code
        == 404
    )

    assert (
        response.json()["detail"]
        == "Service area not found."
    )
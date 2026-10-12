def test_health(
    client,
):
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert (
        data["database"]["connected"]
        is True
    )

    assert (
        data["database"]["name"]
        == "belediye_ai_test"
    )

    assert (
        data["postgis"]["available"]
        is True
    )

    assert (
        data["model"]["loaded"]
        is True
    )

    assert (
        data["model"]["device"]
        == "cpu"
    )
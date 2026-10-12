def test_operator_login_success(
    client,
    operator_credentials,
):
    response = client.post(
        "/auth/login",
        json=operator_credentials,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["username"] == (
        "test_operator"
    )

    assert data["role"] == "OPERATOR"

    assert (
        isinstance(
            data["access_token"],
            str,
        )
    )

    assert (
        len(data["access_token"])
        > 20
    )


def test_wrong_password_returns_401(
    client,
    operator_credentials,
):
    response = client.post(
        "/auth/login",
        json={
            "username":
                operator_credentials[
                    "username"
                ],
            "password":
                "WrongPassword123!",
        },
    )

    assert response.status_code == 401


def test_anonymous_complaint_list_returns_401(
    client,
):
    response = client.get(
        "/complaints"
    )

    assert response.status_code == 401


def test_operator_can_list_complaints(
    client,
    operator_headers,
):
    response = client.get(
        "/complaints",
        headers=operator_headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_manager_can_list_complaints(
    client,
    manager_headers,
):
    response = client.get(
        "/complaints",
        headers=manager_headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_operator_cannot_access_manager_analytics(
    client,
    operator_headers,
):
    response = client.get(
        "/analytics/summary",
        headers=operator_headers,
    )

    assert response.status_code == 403

    assert (
        response.json()["detail"]
        == "Manager role required."
    )


def test_manager_can_access_analytics(
    client,
    manager_headers,
):
    response = client.get(
        "/analytics/summary",
        headers=manager_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_complaints"] == 0
    assert data["auto_accepted"] == 0
    assert data["review_pending"] == 0
    assert data["reviewed"] == 0


def test_manager_cannot_review_complaint(
    client,
    manager_headers,
):
    response = client.post(
        "/complaints/999/review",
        headers=manager_headers,
        json={
            "final_category":
                "TEMIZLIK_ATIK"
        },
    )

    assert response.status_code == 403

    assert (
        response.json()["detail"]
        == "Operator role required."
    )
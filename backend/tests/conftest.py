import os


# KRİTİK GÜVENLİK KURALI:
# Herhangi bir backend modülü import edilmeden ÖNCE
# test veritabanını seçiyoruz.
os.environ["DB_NAME"] = "belediye_ai_test"


import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import backend.app.main as main_module

from backend.app.core.security import hash_password
from backend.app.db.database import (
    SessionLocal,
    engine,
)
from backend.app.models.enums import UserRole
from backend.app.models.user import User


class FakeDevice:
    type = "cpu"

    def __str__(self) -> str:
        return "cpu"


class FakeClassifier:
    def __init__(self) -> None:
        self.device = FakeDevice()
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def predict(
        self,
        text: str,
    ) -> dict:
        if "review" in text.lower():
            return {
                "predicted_category":
                    "HAYVAN_VETERINER",

                "secondary_category":
                    "KENT_DUZENI_DENETIM",

                "model_score": 0.58,
                "second_model_score": 0.40,
                "score_margin": 0.18,

                "review_required": True,

                "review_reason":
                    "LOW_CONFIDENCE_AND_CLOSE_TOP2",

                "model_version":
                    "test-model",
            }

        return {
            "predicted_category":
                "TEMIZLIK_ATIK",

            "secondary_category":
                "YOL_KALDIRIM",

            "model_score": 0.99,
            "second_model_score": 0.005,
            "score_margin": 0.985,

            "review_required": False,
            "review_reason": None,

            "model_version":
                "test-model",
        }


def assert_test_database(
    connection,
) -> None:
    database_name = connection.execute(
        text(
            "SELECT current_database()"
        )
    ).scalar_one()

    if database_name != "belediye_ai_test":
        raise RuntimeError(
            "SAFETY STOP: Tests attempted to use "
            f"database '{database_name}'. "
            "Expected 'belediye_ai_test'."
        )


def seed_saricam_service_area() -> None:
    geojson_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "data"
        / "service_areas"
        / "saricam_osm_relation_1249247.geojson"
    )

    if not geojson_path.exists():
        raise RuntimeError(
            "Sarıçam service-area GeoJSON file "
            f"was not found: {geojson_path}"
        )

    data = json.loads(
        geojson_path.read_text(
            encoding="utf-8"
        )
    )

    features = data.get(
        "features",
        [],
    )

    if len(features) != 1:
        raise RuntimeError(
            "Expected exactly one Sarıçam "
            "GeoJSON feature for tests."
        )

    geometry = features[0].get(
        "geometry"
    )

    if not geometry:
        raise RuntimeError(
            "Sarıçam GeoJSON feature "
            "has no geometry."
        )

    geometry_json = json.dumps(
        geometry,
        ensure_ascii=False,
    )

    with engine.begin() as connection:
        assert_test_database(
            connection
        )

        connection.execute(
            text(
                """
                INSERT INTO service_areas (
                    code,
                    name,
                    source,
                    geometry
                )
                VALUES (
                    'SARICAM',
                    'Sarıçam',
                    'Automated test fixture',
                    ST_Multi(
                        ST_SetSRID(
                            ST_GeomFromGeoJSON(
                                :geometry_json
                            ),
                            4326
                        )
                    )
                )
                """
            ),
            {
                "geometry_json":
                    geometry_json
            },
        )


@pytest.fixture(autouse=True)
def clean_database():
    with engine.begin() as connection:
        assert_test_database(
            connection
        )

        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    complaints,
                    users,
                    service_areas
                RESTART IDENTITY CASCADE
                """
            )
        )

    seed_saricam_service_area()

    yield

    with engine.begin() as connection:
        assert_test_database(
            connection
        )

        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    complaints,
                    users,
                    service_areas
                RESTART IDENTITY CASCADE
                """
            )
        )


@pytest.fixture
def client(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module,
        "BerturkClassifier",
        FakeClassifier,
    )

    with TestClient(
        main_module.app
    ) as test_client:
        yield test_client


def create_test_user(
    username: str,
    password: str,
    role: UserRole,
) -> User:
    db = SessionLocal()

    try:
        user = User(
            username=username,
            password_hash=(
                hash_password(
                    password
                )
            ),
            role=role,
            is_active=True,
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    finally:
        db.close()


@pytest.fixture
def operator_credentials():
    username = "test_operator"
    password = "Operator123!"

    create_test_user(
        username=username,
        password=password,
        role=UserRole.OPERATOR,
    )

    return {
        "username": username,
        "password": password,
    }


@pytest.fixture
def manager_credentials():
    username = "test_manager"
    password = "Manager123!"

    create_test_user(
        username=username,
        password=password,
        role=UserRole.MANAGER,
    )

    return {
        "username": username,
        "password": password,
    }


@pytest.fixture
def operator_headers(
    client,
    operator_credentials,
):
    response = client.post(
        "/auth/login",
        json=operator_credentials,
    )

    assert response.status_code == 200

    token = response.json()[
        "access_token"
    ]

    return {
        "Authorization":
            f"Bearer {token}"
    }


@pytest.fixture
def manager_headers(
    client,
    manager_credentials,
):
    response = client.post(
        "/auth/login",
        json=manager_credentials,
    )

    assert response.status_code == 200

    token = response.json()[
        "access_token"
    ]

    return {
        "Authorization":
            f"Bearer {token}"
    }


@pytest.fixture
def real_client():
    """
    Gerçek fine-tuned BERTurk modelini kullanır.
    FakeClassifier monkeypatch edilmez.
    """
    with TestClient(
        main_module.app
    ) as test_client:
        yield test_client

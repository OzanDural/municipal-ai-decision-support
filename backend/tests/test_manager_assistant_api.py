import backend.app.api.manager_assistant as manager_assistant_api
import backend.app.services.manager_assistant as manager_assistant_service

from backend.app.services.manager_assistant import (
    ManagerAssistantError,
    ManagerAssistantRateLimitError,
    ManagerAssistantResult,
)


class FakeSuccessfulManagerAssistant:
    def ask(
        self,
        message: str,
        db,
    ) -> str:
        assert message == (
            "Toplam başvuru sayısını özetle."
        )

        assert db is not None

        return (
            "Test analitik yanıtı."
        )

    def ask_preset(
        self,
        **kwargs,
    ) -> ManagerAssistantResult:
        analysis_type = kwargs[
            "analysis_type"
        ]

        if analysis_type == "executive_insight":
            return ManagerAssistantResult(
                answer="Yönetici içgörüsü test yanıtı.",
                source="gemini",
            )

        assert analysis_type == "summary"

        return ManagerAssistantResult(
            answer=(
                "Deterministik preset yanıtı."
            ),
            source=(
                "deterministic_fallback"
            ),
            notice=(
                "Gemini kota sınırında."
            ),
        )


class FakeUnavailableManagerAssistant:
    def ask(
        self,
        message: str,
        db,
    ) -> str:
        raise ManagerAssistantError(
            "Simulated Gemini failure."
        )


class FakeRateLimitedManagerAssistant:
    def ask(
        self,
        message: str,
        db,
    ) -> str:
        raise ManagerAssistantRateLimitError(
            "Simulated rate limit."
        )


def test_manager_can_use_manager_assistant(
    client,
    manager_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        manager_assistant_api,
        "ManagerAssistant",
        FakeSuccessfulManagerAssistant,
    )

    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message":
                "Toplam başvuru sayısını özetle."
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer":
            "Test analitik yanıtı.",
        "source": "gemini",
    }


def test_manager_can_use_preset_analysis(
    client,
    manager_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        manager_assistant_api,
        "ManagerAssistant",
        FakeSuccessfulManagerAssistant,
    )

    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message":
                "Genel özeti ver.",
            "analysis_type":
                "summary",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer":
            "Deterministik preset yanıtı.",
        "source":
            "deterministic_fallback",
        "notice":
            "Gemini kota sınırında.",
    }


def test_manager_can_use_executive_insight(
    client,
    manager_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        manager_assistant_api,
        "ManagerAssistant",
        FakeSuccessfulManagerAssistant,
    )

    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message":
                "Seçili dönem için yönetici içgörüsü üret.",
            "analysis_type":
                "executive_insight",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer":
            "Yönetici içgörüsü test yanıtı.",
        "source": "gemini",
    }


def test_executive_insight_aggregates_analytics(
    monkeypatch,
):
    calls = []

    def fake_summary(**kwargs):
        calls.append("summary")
        return {"total_complaints": 10}

    def fake_categories(**kwargs):
        calls.append("categories")
        return [{"category": "TEMIZLIK_ATIK", "count": 4}]

    def fake_trend(**kwargs):
        calls.append("trend")
        return {"total_complaints": 10}

    def fake_neighborhoods(**kwargs):
        calls.append("neighborhoods")
        return [{"neighborhood": "Test", "count": 3}]

    def fake_model_human(**kwargs):
        calls.append("model_human")
        return {"reviewed": 2, "confirmed": 2, "corrected": 0}

    monkeypatch.setattr(
        manager_assistant_service,
        "get_summary",
        fake_summary,
    )
    monkeypatch.setattr(
        manager_assistant_service,
        "get_category_distribution",
        fake_categories,
    )
    monkeypatch.setattr(
        manager_assistant_service,
        "get_trend_analysis",
        fake_trend,
    )
    monkeypatch.setattr(
        manager_assistant_service,
        "get_neighborhood_stats",
        fake_neighborhoods,
    )
    monkeypatch.setattr(
        manager_assistant_service,
        "get_model_human_comparison",
        fake_model_human,
    )

    assistant = (
        manager_assistant_service
        .ManagerAssistant
        .__new__(
            manager_assistant_service.ManagerAssistant
        )
    )

    result = assistant._run_preset_analysis(
        analysis_type="executive_insight",
        db=object(),
    )

    assert calls == [
        "summary",
        "categories",
        "trend",
        "neighborhoods",
        "model_human",
    ]

    assert set(result) == {
        "summary",
        "category_distribution",
        "trend",
        "neighborhoods",
        "model_human",
    }


def test_operator_cannot_use_manager_assistant(
    client,
    operator_headers,
):
    response = client.post(
        "/manager-assistant/chat",
        headers=operator_headers,
        json={
            "message":
                "Toplam başvuru sayısını özetle."
        },
    )

    assert response.status_code == 403


def test_manager_assistant_rejects_blank_message(
    client,
    manager_headers,
):
    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message": "   "
        },
    )

    assert response.status_code == 422


def test_nearby_preset_requires_coordinates(
    client,
    manager_headers,
):
    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message":
                "Yakın çevreyi özetle.",
            "analysis_type":
                "nearby",
        },
    )

    assert response.status_code == 422


def test_manager_assistant_returns_429_on_rate_limit(
    client,
    manager_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        manager_assistant_api,
        "ManagerAssistant",
        FakeRateLimitedManagerAssistant,
    )

    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message":
                "Toplam başvuru sayısını özetle."
        },
    )

    assert response.status_code == 429

    assert response.json() == {
        "detail": (
            "Gemini geçici istek sınırına "
            "ulaştı. Kısa süre sonra tekrar "
            "deneyin."
        )
    }


def test_manager_assistant_returns_503_when_service_fails(
    client,
    manager_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        manager_assistant_api,
        "ManagerAssistant",
        FakeUnavailableManagerAssistant,
    )

    response = client.post(
        "/manager-assistant/chat",
        headers=manager_headers,
        json={
            "message":
                "Toplam başvuru sayısını özetle."
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Manager assistant is temporarily "
            "unavailable."
        )
    }

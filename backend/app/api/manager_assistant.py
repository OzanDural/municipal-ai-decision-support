from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session

from backend.app.api.dependencies import (
    require_manager,
)
from backend.app.db.database import get_db
from backend.app.models.user import User
from backend.app.schemas.manager_assistant import (
    ManagerAssistantRequest,
    ManagerAssistantResponse,
)
from backend.app.services.manager_assistant import (
    ManagerAssistant,
    ManagerAssistantError,
    ManagerAssistantRateLimitError,
)


router = APIRouter(
    prefix="/manager-assistant",
    tags=["manager-assistant"],
)


@router.post(
    "/chat",
    response_model=ManagerAssistantResponse,
    response_model_exclude_none=True,
)
def manager_assistant_chat(
    payload: ManagerAssistantRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    try:
        assistant = ManagerAssistant()

        if payload.analysis_type:
            result = assistant.ask_preset(
                analysis_type=(
                    payload.analysis_type
                ),
                message=payload.message,
                db=db,
                from_date=payload.from_date,
                to_date=payload.to_date,
                latitude=payload.latitude,
                longitude=payload.longitude,
                radius_meters=(
                    payload.radius_meters
                ),
                category=payload.category,
            )

            return ManagerAssistantResponse(
                answer=result.answer,
                source=result.source,
                notice=result.notice,
            )

        answer = assistant.ask(
            message=payload.message,
            db=db,
        )

    except ManagerAssistantRateLimitError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_429_TOO_MANY_REQUESTS
            ),
            detail=(
                "Gemini geçici istek sınırına "
                "ulaştı. Kısa süre sonra tekrar "
                "deneyin."
            ),
        ) from exc

    except ManagerAssistantError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Manager assistant is temporarily "
                "unavailable."
            ),
        ) from exc

    return ManagerAssistantResponse(
        answer=answer,
        source="gemini",
    )

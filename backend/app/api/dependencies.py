from fastapi import (
    Depends,
    HTTPException,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
import jwt
from sqlalchemy.orm import Session

from backend.app.core.security import (
    decode_access_token,
)
from backend.app.db.database import get_db
from backend.app.models.enums import UserRole
from backend.app.models.user import User


bearer_scheme = HTTPBearer(
    auto_error=False
)


def get_current_user(
    credentials: (
        HTTPAuthorizationCredentials | None
    ) = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    try:
        payload = decode_access_token(
            credentials.credentials
        )

        user_id = int(
            payload["sub"]
        )

    except (
        jwt.ExpiredSignatureError,
        jwt.InvalidTokenError,
        KeyError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid or expired token."
            ),
        ) from exc

    user = db.get(
        User,
        user_id,
    )

    if (
        user is None
        or not user.is_active
    ):
        raise HTTPException(
            status_code=401,
            detail="User is not available.",
        )

    return user


def require_operator(
    current_user: User = Depends(
        get_current_user
    ),
) -> User:
    if current_user.role != UserRole.OPERATOR:
        raise HTTPException(
            status_code=403,
            detail="Operator role required.",
        )

    return current_user
def require_manager(
    current_user: User = Depends(
        get_current_user
    ),
) -> User:
    if current_user.role != UserRole.MANAGER:
        raise HTTPException(
            status_code=403,
            detail="Manager role required.",
        )

    return current_user
def require_staff(
    current_user: User = Depends(
        get_current_user
    ),
) -> User:
    if current_user.role not in (
        UserRole.OPERATOR,
        UserRole.MANAGER,
    ):
        raise HTTPException(
            status_code=403,
            detail="Staff role required.",
        )

    return current_user
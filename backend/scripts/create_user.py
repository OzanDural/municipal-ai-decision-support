import getpass
import sys

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.core.security import (
    hash_password,
)
from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.enums import (
    UserRole,
)
from backend.app.models.user import User


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python -m "
            "backend.scripts.create_user "
            "<username> <OPERATOR|MANAGER>"
        )
        raise SystemExit(1)

    username = (
        sys.argv[1]
        .strip()
        .lower()
    )

    try:
        role = UserRole(
            sys.argv[2].upper()
        )
    except ValueError:
        print(
            "Role must be OPERATOR "
            "or MANAGER."
        )
        raise SystemExit(1)

    password = getpass.getpass(
        "Password: "
    )

    password_again = getpass.getpass(
        "Password again: "
    )

    if password != password_again:
        print(
            "Passwords do not match."
        )
        raise SystemExit(1)

    if len(password) < 8:
        print(
            "Password must contain "
            "at least 8 characters."
        )
        raise SystemExit(1)

    db = SessionLocal()

    try:
        existing = db.execute(
            select(User).where(
                User.username
                == username
            )
        ).scalar_one_or_none()

        if existing:
            print(
                "User already exists."
            )
            raise SystemExit(1)

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

        print(
            f"USER_CREATED "
            f"id={user.id} "
            f"username={user.username} "
            f"role={user.role.value}"
        )

    except IntegrityError:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
import getpass
import sys

from sqlalchemy import select

from backend.app.core.security import hash_password
from backend.app.db.database import SessionLocal
from backend.app.models.user import User


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m "
            "backend.scripts.change_password "
            "<username>"
        )
        raise SystemExit(1)

    username = sys.argv[1].strip().lower()

    new_password = getpass.getpass(
        "New password: "
    )

    new_password_again = getpass.getpass(
        "New password again: "
    )

    if new_password != new_password_again:
        print("Passwords do not match.")
        raise SystemExit(1)

    if len(new_password) < 8:
        print(
            "Password must contain "
            "at least 8 characters."
        )
        raise SystemExit(1)

    db = SessionLocal()

    try:
        user = db.execute(
            select(User).where(
                User.username == username
            )
        ).scalar_one_or_none()

        if user is None:
            print("User not found.")
            raise SystemExit(1)

        user.password_hash = hash_password(
            new_password
        )

        db.commit()

        print(
            f"PASSWORD_CHANGED "
            f"username={user.username}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
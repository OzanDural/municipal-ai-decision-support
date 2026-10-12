from sqlalchemy import text

from backend.app.db.database import engine


def main() -> None:
    with engine.connect() as connection:
        current_user = connection.execute(
            text("SELECT current_user")
        ).scalar_one()

        current_database = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        postgis_version = connection.execute(
            text("SELECT PostGIS_Version()")
        ).scalar_one()

    print("=" * 70)
    print("STAGE 15C - SQLALCHEMY DATABASE CONNECTION TEST")
    print("=" * 70)
    print(f"Current user     : {current_user}")
    print(f"Current database : {current_database}")
    print(f"PostGIS          : {postgis_version}")
    print()
    print("STATUS: DATABASE_CONNECTION_OK")


if __name__ == "__main__":
    main()
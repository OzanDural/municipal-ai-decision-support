from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.db.database import Base, engine
import backend.app.models  # noqa: F401, E402


config = context.config


if config.config_file_name is not None:
    fileConfig(
        config.config_file_name
    )


target_metadata = Base.metadata

def include_object(
    object_,
    name,
    type_,
    reflected,
    compare_to,
):
    # PostGIS extension tarafından yönetilir.
    # Uygulama migration'ları bunu değiştirmemeli.
    if (
        type_ == "table"
        and name == "spatial_ref_sys"
    ):
        return False

    return True

def run_migrations_offline() -> None:
    url = engine.url.render_as_string(
        hide_password=False
    )

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named"
        },
        compare_type=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
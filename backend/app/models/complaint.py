from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base
from backend.app.models.enums import (
    ComplaintCategory,
    ComplaintStatus,
)


complaint_category_enum = SQLEnum(
    ComplaintCategory,
    name="complaint_category",
)

complaint_status_enum = SQLEnum(
    ComplaintStatus,
    name="complaint_status",
)


class Complaint(Base):
    __tablename__ = "complaints"

    __table_args__ = (
        CheckConstraint(
            "model_score >= 0 AND model_score <= 1",
            name="ck_complaints_model_score_range",
        ),
        CheckConstraint(
            "second_model_score >= 0 "
            "AND second_model_score <= 1",
            name=(
                "ck_complaints_"
                "second_model_score_range"
            ),
        ),
        CheckConstraint(
            "score_margin >= 0 AND score_margin <= 1",
            name="ck_complaints_score_margin_range",
        ),
        Index(
            "ix_complaints_location_gist",
            "location",
            postgresql_using="gist",
        ),
        Index(
            "ix_complaints_review_queue",
            "review_required",
            "created_at",
        ),
        Index(
            "ix_complaints_neighborhood_created_at",
            "neighborhood",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    application_no: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
    )

    complaint_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Kullanıcı tarafından seçilen standart mahalle.
    # İleride mahalle + tarih + kategori
    # analizlerinde kullanılacak.
    neighborhood: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Sokak / cadde adı.
    # Şimdilik kullanıcı girdisi olarak tutulacak.
    street_name: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
    )

    # Bina, yakın konum, tarif vb.
    # serbest metin adres açıklaması.
    address_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # CBS açısından esas konum bilgisi.
    # Adres alanları bunun yerine geçmez.
    location: Mapped[object] = mapped_column(
        Geography(
            geometry_type="POINT",
            srid=4326,
            spatial_index=False,
        ),
        nullable=False,
    )

    predicted_category: Mapped[
        ComplaintCategory
    ] = mapped_column(
        complaint_category_enum,
        nullable=False,
    )

    secondary_category: Mapped[
        ComplaintCategory
    ] = mapped_column(
        complaint_category_enum,
        nullable=False,
    )

    model_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    second_model_score: Mapped[
        float
    ] = mapped_column(
        Float,
        nullable=False,
    )

    score_margin: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    review_required: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        nullable=False,
    )

    review_reason: Mapped[
        str | None
    ] = mapped_column(
        String(80),
        nullable=True,
    )

    final_category: Mapped[
        ComplaintCategory | None
    ] = mapped_column(
        complaint_category_enum,
        nullable=True,
    )

    was_corrected: Mapped[
        bool | None
    ] = mapped_column(
        Boolean,
        nullable=True,
    )

    reviewed_by: Mapped[
        int | None
    ] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    reviewed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    status: Mapped[
        ComplaintStatus
    ] = mapped_column(
        complaint_status_enum,
        nullable=False,
    )

    created_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
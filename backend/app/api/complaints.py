from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from geoalchemy2.elements import WKTElement
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.models.complaint import Complaint
from backend.app.models.enums import (
    ComplaintCategory,
    ComplaintStatus,
)
from backend.app.schemas.complaint import (
    ComplaintCreate,
    ComplaintCreateResponse,
    ComplaintDetailResponse,
    ComplaintListItem,
    ComplaintPageResponse,
    ComplaintReviewRequest,
    ComplaintReviewResponse,
)
from geoalchemy2 import Geometry
from sqlalchemy import cast, func, select

router = APIRouter(
    prefix="/complaints",
    tags=["complaints"],
)

from datetime import datetime, timezone

from backend.app.api.dependencies import (
    require_operator,
    require_staff,
)
from backend.app.models.user import User
from backend.app.services.service_area import (
    ServiceAreaNotConfiguredError,
    is_location_within_service_area,
)


def generate_application_no() -> str:
    return (
        "SRM-"
        + uuid4().hex[:12].upper()
    )


@router.post(
    "",
    response_model=ComplaintCreateResponse,
    status_code=201,
)
def create_complaint(
    payload: ComplaintCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        location_is_valid = (
            is_location_within_service_area(
                db,
                latitude=payload.latitude,
                longitude=payload.longitude,
            )
        )

    except ServiceAreaNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Municipal service area "
                "is not configured."
            ),
        ) from exc

    if not location_is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Selected location is outside "
                "the Sarıçam service area."
            ),
        )

    classifier = (
        request.app.state.classifier
    )

    prediction = classifier.predict(
        payload.complaint_text
    )

    if prediction["review_required"]:
        complaint_status = (
            ComplaintStatus.REVIEW_REQUIRED
        )
    else:
        complaint_status = (
            ComplaintStatus.AUTO_ACCEPTED
        )

    location = WKTElement(
        (
            f"POINT("
            f"{payload.longitude} "
            f"{payload.latitude}"
            f")"
        ),
        srid=4326,
    )

    complaint = Complaint(
        application_no=(
            generate_application_no()
        ),

        complaint_text=(
            payload.complaint_text
        ),

        neighborhood=(
            payload.neighborhood
        ),

        street_name=(
            payload.street_name
        ),

        address_text=(
            payload.address_text
        ),

        location=location,

        predicted_category=(
            ComplaintCategory(
                prediction[
                    "predicted_category"
                ]
            )
        ),

        secondary_category=(
            ComplaintCategory(
                prediction[
                    "secondary_category"
                ]
            )
        ),

        model_score=(
            prediction[
                "model_score"
            ]
        ),

        second_model_score=(
            prediction[
                "second_model_score"
            ]
        ),

        score_margin=(
            prediction[
                "score_margin"
            ]
        ),

        model_version=(
            prediction[
                "model_version"
            ]
        ),

        review_required=(
            prediction[
                "review_required"
            ]
        ),

        review_reason=(
            prediction[
                "review_reason"
            ]
        ),

        # İnsan kontrolü yapılmadığı için
        # final_category bilinçli olarak NULL.
        final_category=None,

        was_corrected=None,

        reviewed_by=None,

        reviewed_at=None,

        status=complaint_status,
    )

    try:
        db.add(
            complaint
        )

        db.commit()

        db.refresh(
            complaint
        )

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Complaint could not "
                "be saved."
            ),
        ) from exc

    return ComplaintCreateResponse(
        id=complaint.id,

        application_no=(
            complaint.application_no
        ),

        neighborhood=(
            complaint.neighborhood
        ),

        street_name=(
            complaint.street_name
        ),

        address_text=(
            complaint.address_text
        ),

        predicted_category=(
            complaint
            .predicted_category
            .value
        ),

        secondary_category=(
            complaint
            .secondary_category
            .value
        ),

        model_score=(
            complaint.model_score
        ),

        second_model_score=(
            complaint
            .second_model_score
        ),

        score_margin=(
            complaint.score_margin
        ),

        review_required=(
            complaint.review_required
        ),

        review_reason=(
            complaint.review_reason
        ),

        status=(
            complaint.status.value
        ),

        model_version=(
            complaint.model_version
        ),

        latitude=payload.latitude,
        longitude=payload.longitude,

        created_at=(
            complaint.created_at
        ),
    )


def complaint_coordinates():
    geometry = cast(
        Complaint.location,
        Geometry(
            geometry_type="POINT",
            srid=4326,
        ),
    )

    latitude = func.ST_Y(
        geometry
    ).label("latitude")

    longitude = func.ST_X(
        geometry
    ).label("longitude")

    return latitude, longitude


@router.get(
    "",
    response_model=list[ComplaintListItem],
)
def list_complaints(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_staff
    ),
    review_required: bool | None = None,
    complaint_status: ComplaintStatus | None = Query(
        default=None,
        alias="status",
    ),
    limit: int = 100,
):
    latitude, longitude = (
        complaint_coordinates()
    )

    statement = select(
        Complaint,
        latitude,
        longitude,
    )

    # Eski review_required filtresini
    # koruyoruz.
    if review_required is not None:
        statement = statement.where(
            Complaint.review_required
            == review_required
        )

    # Yeni durum sekmeleri için
    # gerçek status filtresi.
    if complaint_status is not None:
        statement = statement.where(
            Complaint.status
            == complaint_status
        )

    statement = (
        statement
        .order_by(
            Complaint.created_at.desc()
        )
        .limit(
            min(limit, 200)
        )
    )

    rows = db.execute(
        statement
    ).all()

    return [
        ComplaintListItem(
            id=complaint.id,

            application_no=(
                complaint.application_no
            ),

            complaint_text=(
                complaint.complaint_text
            ),

            neighborhood=(
                complaint.neighborhood
            ),

            street_name=(
                complaint.street_name
            ),

            address_text=(
                complaint.address_text
            ),

            predicted_category=(
                complaint
                .predicted_category
                .value
            ),

            secondary_category=(
                complaint
                .secondary_category
                .value
            ),

            model_score=(
                complaint.model_score
            ),

            score_margin=(
                complaint.score_margin
            ),

            review_required=(
                complaint.review_required
            ),

            review_reason=(
                complaint.review_reason
            ),

            final_category=(
                complaint.final_category.value
                if complaint.final_category
                else None
            ),

            was_corrected=(
                complaint.was_corrected
            ),

            status=(
                complaint.status.value
            ),

            latitude=float(
                latitude_value
            ),

            longitude=float(
                longitude_value
            ),

            created_at=(
                complaint.created_at
            ),
        )
        for (
            complaint,
            latitude_value,
            longitude_value,
        )
        in rows
    ]
@router.get(
    "/paged",
    response_model=ComplaintPageResponse,
)
def list_complaints_paged(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_staff
    ),
    complaint_status: ComplaintStatus | None = Query(
        default=None,
        alias="status",
    ),
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
):
    latitude, longitude = (
        complaint_coordinates()
    )

    filters = []

    if complaint_status is not None:
        filters.append(
            Complaint.status
            == complaint_status
        )

    count_statement = select(
        func.count(
            Complaint.id
        )
    )

    if filters:
        count_statement = (
            count_statement.where(
                *filters
            )
        )

    total = int(
        db.execute(
            count_statement
        ).scalar_one()
        or 0
    )

    total_pages = (
        (
            total
            + page_size
            - 1
        )
        // page_size
        if total > 0
        else 0
    )

    offset = (
        page - 1
    ) * page_size

    statement = (
        select(
            Complaint,
            latitude,
            longitude,
        )
        .order_by(
            Complaint.created_at.desc(),
            Complaint.id.desc(),
        )
        .offset(
            offset
        )
        .limit(
            page_size
        )
    )

    if filters:
        statement = (
            statement.where(
                *filters
            )
        )

    rows = db.execute(
        statement
    ).all()

    items = [
        ComplaintListItem(
            id=complaint.id,

            application_no=(
                complaint.application_no
            ),

            complaint_text=(
                complaint.complaint_text
            ),

            neighborhood=(
                complaint.neighborhood
            ),

            street_name=(
                complaint.street_name
            ),

            address_text=(
                complaint.address_text
            ),

            predicted_category=(
                complaint
                .predicted_category
                .value
            ),

            secondary_category=(
                complaint
                .secondary_category
                .value
            ),

            model_score=(
                complaint.model_score
            ),

            score_margin=(
                complaint.score_margin
            ),

            review_required=(
                complaint.review_required
            ),

            review_reason=(
                complaint.review_reason
            ),

            final_category=(
                complaint.final_category.value
                if complaint.final_category
                else None
            ),

            was_corrected=(
                complaint.was_corrected
            ),

            status=(
                complaint.status.value
            ),

            latitude=float(
                latitude_value
            ),

            longitude=float(
                longitude_value
            ),

            created_at=(
                complaint.created_at
            ),
        )
        for (
            complaint,
            latitude_value,
            longitude_value,
        )
        in rows
    ]

    return ComplaintPageResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
@router.get(
    "/by-application-no/{application_no}",
    response_model=ComplaintDetailResponse,
)
def get_complaint_by_application_no(
    application_no: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_staff
    ),
):
    normalized_application_no = (
        application_no
        .strip()
        .upper()
    )

    complaint_id = db.execute(
        select(Complaint.id)
        .where(
            Complaint.application_no
            == normalized_application_no
        )
    ).scalar_one_or_none()

    if complaint_id is None:
        raise HTTPException(
            status_code=404,
            detail="Complaint not found.",
        )

    return get_complaint(
        complaint_id=complaint_id,
        db=db,
        current_user=current_user,
    )


@router.get(
    "/{complaint_id}",
    response_model=ComplaintDetailResponse,
)
def get_complaint(
    complaint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_staff
    ),
):
    latitude, longitude = (
        complaint_coordinates()
    )

    statement = (
        select(
            Complaint,
            latitude,
            longitude,
        )
        .where(
            Complaint.id
            == complaint_id
        )
    )

    row = db.execute(
        statement
    ).one_or_none()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Complaint not found.",
        )

    (
        complaint,
        latitude_value,
        longitude_value,
    ) = row

    return ComplaintDetailResponse(
        id=complaint.id,
        application_no=(
            complaint.application_no
        ),
        complaint_text=(
            complaint.complaint_text
        ),
        neighborhood=(
            complaint.neighborhood
        ),
        street_name=(
            complaint.street_name
        ),
        address_text=(
            complaint.address_text
        ),
        predicted_category=(
            complaint
            .predicted_category
            .value
        ),
        secondary_category=(
            complaint
            .secondary_category
            .value
        ),
        model_score=(
            complaint.model_score
        ),
        second_model_score=(
            complaint.second_model_score
        ),
        score_margin=(
            complaint.score_margin
        ),
        model_version=(
            complaint.model_version
        ),
        review_required=(
            complaint.review_required
        ),
        review_reason=(
            complaint.review_reason
        ),
        final_category=(
            complaint.final_category.value
            if complaint.final_category
            else None
        ),
        was_corrected=(
            complaint.was_corrected
        ),
        reviewed_by=(
            complaint.reviewed_by
        ),
        reviewed_at=(
            complaint.reviewed_at
        ),
        status=(
            complaint.status.value
        ),
        latitude=float(
            latitude_value
        ),
        longitude=float(
            longitude_value
        ),
        created_at=(
            complaint.created_at
        ),
        updated_at=(
            complaint.updated_at
        ),
    )


@router.post(
    "/{complaint_id}/review",
    response_model=ComplaintReviewResponse,
)
def review_complaint(
    complaint_id: int,
    payload: ComplaintReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_operator
    ),
):
    complaint = db.get(
        Complaint,
        complaint_id,
    )

    if complaint is None:
        raise HTTPException(
            status_code=404,
            detail="Complaint not found.",
        )

    if complaint.status not in (
        ComplaintStatus.REVIEW_REQUIRED,
        ComplaintStatus.AUTO_ACCEPTED,
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Complaint is not waiting "
                "for human review."
            ),
        )

    final_category = (
        payload.final_category
    )

    was_corrected = (
        final_category
        != complaint.predicted_category
    )

    complaint.final_category = (
        final_category
    )

    complaint.was_corrected = (
        was_corrected
    )

    complaint.reviewed_by = (
        current_user.id
    )

    complaint.reviewed_at = (
        datetime.now(
            timezone.utc
        )
    )

    complaint.review_required = False

    complaint.status = (
        ComplaintStatus.REVIEWED
    )

    try:
        db.commit()
        db.refresh(
            complaint
        )

    except Exception:
        db.rollback()
        raise

    return ComplaintReviewResponse(
        id=complaint.id,
        application_no=(
            complaint.application_no
        ),
        predicted_category=(
            complaint
            .predicted_category
            .value
        ),
        final_category=(
            complaint
            .final_category
            .value
        ),
        was_corrected=(
            complaint.was_corrected
        ),
        reviewed_by=(
            complaint.reviewed_by
        ),
        reviewed_at=(
            complaint.reviewed_at
        ),
        review_required=(
            complaint.review_required
        ),
        status=(
            complaint.status.value
        ),
    )
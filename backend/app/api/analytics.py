from datetime import date

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from geoalchemy2 import (
    Geography,
    Geometry,
)
from sqlalchemy import (
    Date,
    case,
    cast,
    func,
    select,
)
from sqlalchemy.orm import Session

from backend.app.api.dependencies import (
    require_manager,
)
from backend.app.db.database import get_db
from backend.app.models.complaint import Complaint
from backend.app.models.enums import (
    ComplaintCategory,
    ComplaintStatus,
)
from backend.app.models.user import User
from backend.app.schemas.analytics import (
    AnalyticsSummaryResponse,
    CategoryDistributionItem,
    NearbyComplaintItem,
    NearbyComplaintsResponse,
    TrendItem,
    MapPointItem,
    NeighborhoodAnalyticsItem,
    ModelHumanCategoryItem,
    CorrectionFlowItem,
    ModelHumanComparisonResponse,
)
from backend.app.services.manager_analytics import (
    get_category_distribution,
    get_model_human_comparison,
    get_nearby_complaints,
    get_neighborhood_stats,
    get_summary,
    get_trend,
)

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
)


def validate_date_range(
    from_date: date | None,
    to_date: date | None,
) -> None:
    if (
        from_date is not None
        and to_date is not None
        and from_date > to_date
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "from_date cannot be later "
                "than to_date."
            ),
        )


def apply_created_at_filter(
    statement,
    from_date: date | None,
    to_date: date | None,
):
    validate_date_range(
        from_date,
        to_date,
    )

    created_date = cast(
        Complaint.created_at,
        Date,
    )

    if from_date is not None:
        statement = statement.where(
            created_date >= from_date
        )

    if to_date is not None:
        statement = statement.where(
            created_date <= to_date
        )

    return statement


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
)
def analytics_summary(
    from_date: date | None = Query(
        default=None,
    ),
    to_date: date | None = Query(
        default=None,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    return get_summary(
        db=db,
        from_date=from_date,
        to_date=to_date,
    )

@router.get(
    "/categories",
    response_model=list[
        CategoryDistributionItem
    ],
)
def category_distribution(
    from_date: date | None = Query(
        default=None,
    ),
    to_date: date | None = Query(
        default=None,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    return get_category_distribution(
        db=db,
        from_date=from_date,
        to_date=to_date,
    )

@router.get(
    "/trend",
    response_model=list[TrendItem],
)
def complaint_trend(
    from_date: date | None = Query(
        default=None
    ),
    to_date: date | None = Query(
        default=None
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    return get_trend(
        db=db,
        from_date=from_date,
        to_date=to_date,
    )
@router.get(
    "/map-points",
    response_model=list[MapPointItem],
)
def complaint_map_points(
    from_date: date | None = Query(
        default=None,
    ),
    to_date: date | None = Query(
        default=None,
    ),
    category: ComplaintCategory | None = Query(
        default=None,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    complaint_geometry = cast(
        Complaint.location,
        Geometry(
            geometry_type="POINT",
            srid=4326,
        ),
    )

    complaint_latitude = func.ST_Y(
        complaint_geometry
    ).label("latitude")

    complaint_longitude = func.ST_X(
        complaint_geometry
    ).label("longitude")

    effective_category = func.coalesce(
        Complaint.final_category,
        Complaint.predicted_category,
    ).label(
        "effective_category"
    )

    statement = select(
        Complaint,
        complaint_latitude,
        complaint_longitude,
        effective_category,
    ).order_by(
        Complaint.created_at.desc()
    )

    statement = apply_created_at_filter(
        statement,
        from_date,
        to_date,
    )

    if category is not None:
        statement = statement.where(
            func.coalesce(
                Complaint.final_category,
                Complaint.predicted_category,
            )
            == category
        )

    rows = db.execute(
        statement
    ).all()

    result: list[MapPointItem] = []

    for row in rows:
        category_value = (
            row.effective_category.value
            if hasattr(
                row.effective_category,
                "value",
            )
            else str(
                row.effective_category
            )
        )

        status_value = (
            row.Complaint.status.value
            if hasattr(
                row.Complaint.status,
                "value",
            )
            else str(
                row.Complaint.status
            )
        )

        result.append(
            MapPointItem(
                id=row.Complaint.id,
                application_no=(
                    row.Complaint.application_no
                ),
                complaint_text=(
                    row.Complaint.complaint_text
                ),
                neighborhood=(
                    row.Complaint.neighborhood
                ),
                category=category_value,
                status=status_value,
                latitude=float(
                    row.latitude
                ),
                longitude=float(
                    row.longitude
                ),
                created_at=(
                    row.Complaint.created_at
                ),
            )
        )

    return result
@router.get(
    "/neighborhoods",
    response_model=list[
        NeighborhoodAnalyticsItem
    ],
)
def neighborhood_analytics(
    from_date: date | None = Query(
        default=None,
    ),
    to_date: date | None = Query(
        default=None,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    return get_neighborhood_stats(
        db=db,
        from_date=from_date,
        to_date=to_date,
    )

@router.get(
    "/model-human",
    response_model=(
        ModelHumanComparisonResponse
    ),
)
def model_human_comparison(
    from_date: date | None = Query(
        default=None,
    ),
    to_date: date | None = Query(
        default=None,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    return get_model_human_comparison(
        db=db,
        from_date=from_date,
        to_date=to_date,
    )

@router.get(
    "/nearby",
    response_model=NearbyComplaintsResponse,
)
def nearby_complaints(
    latitude: float = Query(
        ge=-90,
        le=90,
    ),
    longitude: float = Query(
        ge=-180,
        le=180,
    ),
    radius_meters: float = Query(
        default=1000,
        ge=1,
        le=10000,
    ),
    category: ComplaintCategory | None = None,
    from_date: date | None = Query(
        default=None,
    ),
    to_date: date | None = Query(
        default=None,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_manager
    ),
):
    validate_date_range(
        from_date,
        to_date,
    )

    return get_nearby_complaints(
        db=db,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        category=category,
        from_date=from_date,
        to_date=to_date,
    )
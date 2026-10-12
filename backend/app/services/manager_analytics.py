from datetime import date,timedelta

from sqlalchemy import (
    Date,
    case,
    cast,
    func,
    select,
)
from sqlalchemy.orm import Session

from backend.app.models.complaint import Complaint
from backend.app.models.enums import (
    ComplaintCategory,
    ComplaintStatus,
)
from geoalchemy2 import (
    Geography,
    Geometry,
)

def _validate_date_range(
    from_date: date | None,
    to_date: date | None,
) -> None:
    if (
        from_date is not None
        and to_date is not None
        and from_date > to_date
    ):
        raise ValueError(
            "from_date cannot be later than to_date."
        )


def _apply_created_at_filter(
    statement,
    from_date: date | None,
    to_date: date | None,
):
    _validate_date_range(
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


def get_summary(
    db: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> dict[str, object]:
    statement = select(
        func.count(
            Complaint.id
        ).label("total"),

        func.sum(
            case(
                (
                    Complaint.status
                    == ComplaintStatus.AUTO_ACCEPTED,
                    1,
                ),
                else_=0,
            )
        ).label("auto_accepted"),

        func.sum(
            case(
                (
                    Complaint.status
                    == ComplaintStatus.REVIEW_REQUIRED,
                    1,
                ),
                else_=0,
            )
        ).label("review_pending"),

        func.sum(
            case(
                (
                    Complaint.status
                    == ComplaintStatus.REVIEWED,
                    1,
                ),
                else_=0,
            )
        ).label("reviewed"),

        func.sum(
            case(
                (
                    Complaint.was_corrected
                    == True,
                    1,
                ),
                else_=0,
            )
        ).label("corrected"),
    )

    statement = _apply_created_at_filter(
        statement,
        from_date,
        to_date,
    )

    row = db.execute(
        statement
    ).one()

    total = int(
        row.total or 0
    )

    auto_accepted = int(
        row.auto_accepted or 0
    )

    review_pending = int(
        row.review_pending or 0
    )

    reviewed = int(
        row.reviewed or 0
    )

    corrected = int(
        row.corrected or 0
    )

    human_review_total = (
        review_pending
        + reviewed
    )

    auto_acceptance_rate = (
        auto_accepted / total
        if total > 0
        else 0.0
    )

    human_review_rate = (
        human_review_total / total
        if total > 0
        else 0.0
    )

    correction_rate = (
        corrected / reviewed
        if reviewed > 0
        else 0.0
    )

    return {
        "from_date": (
            from_date.isoformat()
            if from_date is not None
            else None
        ),
        "to_date": (
            to_date.isoformat()
            if to_date is not None
            else None
        ),
        "total_complaints":
            total,
        "auto_accepted":
            auto_accepted,
        "review_pending":
            review_pending,
        "reviewed":
            reviewed,
        "human_review_total":
            human_review_total,
        "corrected_by_human":
            corrected,
        "auto_acceptance_rate":
            auto_acceptance_rate,
        "human_review_rate":
            human_review_rate,
        "correction_rate":
            correction_rate,
    }


def get_category_distribution(
    db: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[dict[str, object]]:
    effective_category = func.coalesce(
        Complaint.final_category,
        Complaint.predicted_category,
    ).label(
        "effective_category"
    )

    statement = (
        select(
            effective_category,
            func.count(
                Complaint.id
            ).label("count"),
        )
        .group_by(
            effective_category
        )
        .order_by(
            func.count(
                Complaint.id
            ).desc()
        )
    )

    statement = _apply_created_at_filter(
        statement,
        from_date,
        to_date,
    )

    rows = db.execute(
        statement
    ).all()

    total = sum(
        int(row.count)
        for row in rows
    )

    result: list[
        dict[str, object]
    ] = []

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

        count = int(
            row.count
        )

        percentage = (
            count / total
            if total > 0
            else 0.0
        )

        result.append(
            {
                "category":
                    category_value,
                "count":
                    count,
                "percentage":
                    percentage,
            }
        )

    return result
def get_trend(
    db: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[dict[str, object]]:
    created_date = cast(
        Complaint.created_at,
        Date,
    ).label("date")

    statement = (
        select(
            created_date,
            func.count(
                Complaint.id
            ).label("count"),
        )
        .group_by(
            created_date
        )
        .order_by(
            created_date.asc()
        )
    )

    statement = _apply_created_at_filter(
        statement,
        from_date,
        to_date,
    )

    rows = db.execute(
        statement
    ).all()

    return [
        {
            "date": row.date.isoformat(),
            "count": int(
                row.count
            ),
        }
        for row in rows
    ]
def get_trend_analysis(
    db: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> dict[str, object]:
    """
    Gemini yönetici asistanı için günlük trend
    verisini deterministik biçimde özetler.

    Eksik takvim günleri 0 başvuru ile doldurulur.
    Gemini'nin ortalama, toplam veya en yoğun gün
    gibi metrikleri kendisinin hesaplamasına gerek
    kalmaz.
    """

    _validate_date_range(
        from_date,
        to_date,
    )

    raw_trend = get_trend(
        db=db,
        from_date=from_date,
        to_date=to_date,
    )

    counts_by_date: dict[
        date,
        int,
    ] = {
        date.fromisoformat(
            str(item["date"])
        ): int(
            item["count"]
        )
        for item in raw_trend
    }

    if from_date is not None:
        period_start = from_date
    elif counts_by_date:
        period_start = min(
            counts_by_date
        )
    else:
        period_start = to_date

    if to_date is not None:
        period_end = to_date
    elif counts_by_date:
        period_end = max(
            counts_by_date
        )
    else:
        period_end = from_date

    if (
        period_start is None
        or period_end is None
    ):
        return {
            "from_date": None,
            "to_date": None,
            "total_complaints": 0,
            "calendar_days": 0,
            "active_days": 0,
            "daily_average": 0.0,
            "busiest_day": None,
            "busiest_day_count": 0,
            "daily_data": [],
        }

    daily_data: list[
        dict[str, object]
    ] = []

    current_date = period_start

    while current_date <= period_end:
        daily_data.append(
            {
                "date":
                    current_date.isoformat(),
                "count":
                    counts_by_date.get(
                        current_date,
                        0,
                    ),
            }
        )

        current_date += timedelta(
            days=1
        )

    total_complaints = sum(
        int(item["count"])
        for item in daily_data
    )

    calendar_days = len(
        daily_data
    )

    active_days = sum(
        1
        for item in daily_data
        if int(item["count"]) > 0
    )

    daily_average = (
        total_complaints
        / calendar_days
        if calendar_days > 0
        else 0.0
    )

    if total_complaints > 0:
        busiest_item = max(
            daily_data,
            key=lambda item:
                int(item["count"]),
        )

        busiest_day = str(
            busiest_item["date"]
        )

        busiest_day_count = int(
            busiest_item["count"]
        )
    else:
        busiest_day = None
        busiest_day_count = 0

    return {
        "from_date":
            period_start.isoformat(),

        "to_date":
            period_end.isoformat(),

        "total_complaints":
            total_complaints,

        "calendar_days":
            calendar_days,

        "active_days":
            active_days,

        "daily_average":
            daily_average,

        "busiest_day":
            busiest_day,

        "busiest_day_count":
            busiest_day_count,

        "daily_data":
            daily_data,
    }
def get_neighborhood_stats(
    db: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[dict[str, object]]:
    effective_category = func.coalesce(
        Complaint.final_category,
        Complaint.predicted_category,
    ).label(
        "effective_category"
    )

    statement = (
        select(
            Complaint.neighborhood,
            effective_category,
            func.count(
                Complaint.id
            ).label("count"),
        )
        .where(
            Complaint.neighborhood.is_not(
                None
            )
        )
        .where(
            func.length(
                func.trim(
                    Complaint.neighborhood
                )
            ) > 0
        )
        .group_by(
            Complaint.neighborhood,
            effective_category,
        )
    )

    statement = _apply_created_at_filter(
        statement,
        from_date,
        to_date,
    )

    rows = db.execute(
        statement
    ).all()

    grouped: dict[
        str,
        dict[str, int],
    ] = {}

    for row in rows:
        neighborhood = str(
            row.neighborhood
        )

        category = (
            row.effective_category.value
            if hasattr(
                row.effective_category,
                "value",
            )
            else str(
                row.effective_category
            )
        )

        if neighborhood not in grouped:
            grouped[
                neighborhood
            ] = {}

        grouped[
            neighborhood
        ][
            category
        ] = int(
            row.count
        )

    total = sum(
        sum(
            category_counts.values()
        )
        for category_counts
        in grouped.values()
    )

    result: list[
        dict[str, object]
    ] = []

    for (
        neighborhood,
        category_counts,
    ) in grouped.items():
        count = sum(
            category_counts.values()
        )

        (
            dominant_category,
            dominant_category_count,
        ) = max(
            category_counts.items(),
            key=lambda item:
                item[1],
        )

        percentage = (
            count / total
            if total > 0
            else 0.0
        )

        result.append(
            {
                "neighborhood":
                    neighborhood,

                "count":
                    count,

                "percentage":
                    percentage,

                "dominant_category":
                    dominant_category,

                "dominant_category_count":
                    dominant_category_count,

                "category_counts":
                    category_counts,
            }
        )

    result.sort(
        key=lambda item: (
            -int(
                item["count"]
            ),
            str(
                item["neighborhood"]
            ),
        )
    )

    return result
def get_model_human_comparison(
    db: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> dict[str, object]:
    statement = (
        select(
            Complaint.predicted_category,
            Complaint.final_category,
            Complaint.was_corrected,
        )
        .where(
            Complaint.status
            == ComplaintStatus.REVIEWED
        )
        .where(
            Complaint.final_category.is_not(
                None
            )
        )
    )

    statement = _apply_created_at_filter(
        statement,
        from_date,
        to_date,
    )

    rows = db.execute(
        statement
    ).all()

    reviewed = len(rows)

    confirmed = 0
    corrected = 0

    category_stats: dict[
        str,
        dict[str, int],
    ] = {}

    flow_counts: dict[
        tuple[str, str],
        int,
    ] = {}

    for row in rows:
        predicted = (
            row.predicted_category.value
            if hasattr(
                row.predicted_category,
                "value",
            )
            else str(
                row.predicted_category
            )
        )

        final = (
            row.final_category.value
            if hasattr(
                row.final_category,
                "value",
            )
            else str(
                row.final_category
            )
        )

        if predicted not in category_stats:
            category_stats[predicted] = {
                "reviewed": 0,
                "confirmed": 0,
                "corrected": 0,
            }

        category_stats[
            predicted
        ]["reviewed"] += 1

        if predicted == final:
            confirmed += 1

            category_stats[
                predicted
            ]["confirmed"] += 1

        else:
            corrected += 1

            category_stats[
                predicted
            ]["corrected"] += 1

            flow_key = (
                predicted,
                final,
            )

            flow_counts[
                flow_key
            ] = (
                flow_counts.get(
                    flow_key,
                    0,
                )
                + 1
            )

    agreement_rate = (
        confirmed / reviewed
        if reviewed > 0
        else 0.0
    )

    correction_rate = (
        corrected / reviewed
        if reviewed > 0
        else 0.0
    )

    by_predicted_category = []

    for (
        category,
        values,
    ) in category_stats.items():
        category_reviewed = (
            values["reviewed"]
        )

        category_correction_rate = (
            values["corrected"]
            / category_reviewed
            if category_reviewed > 0
            else 0.0
        )

        by_predicted_category.append(
            {
                "category":
                    category,

                "reviewed":
                    category_reviewed,

                "confirmed":
                    values["confirmed"],

                "corrected":
                    values["corrected"],

                "correction_rate":
                    category_correction_rate,
            }
        )

    by_predicted_category.sort(
        key=lambda item: (
            -int(
                item["reviewed"]
            ),
            str(
                item["category"]
            ),
        )
    )

    correction_flows = [
        {
            "predicted_category":
                predicted,

            "final_category":
                final,

            "count":
                count,
        }
        for (
            predicted,
            final,
        ),
        count
        in flow_counts.items()
    ]

    correction_flows.sort(
        key=lambda item: (
            -int(
                item["count"]
            ),
            str(
                item["predicted_category"]
            ),
            str(
                item["final_category"]
            ),
        )
    )

    return {
        "reviewed":
            reviewed,

        "confirmed":
            confirmed,

        "corrected":
            corrected,

        "agreement_rate":
            agreement_rate,

        "correction_rate":
            correction_rate,

        "by_predicted_category":
            by_predicted_category,

        "correction_flows":
            correction_flows,
    }
def get_nearby_complaints(
    db: Session,
    latitude: float,
    longitude: float,
    radius_meters: float = 1000.0,
    category: ComplaintCategory | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
) -> dict[str, object]:
    if not -90 <= latitude <= 90:
        raise ValueError(
            "latitude must be between -90 and 90."
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            "longitude must be between -180 and 180."
        )

    if not 1 <= radius_meters <= 10000:
        raise ValueError(
            (
                "radius_meters must be between "
                "1 and 10000."
            )
        )

    _validate_date_range(
        from_date,
        to_date,
    )

    search_point = cast(
        func.ST_SetSRID(
            func.ST_MakePoint(
                longitude,
                latitude,
            ),
            4326,
        ),
        Geography(
            geometry_type="POINT",
            srid=4326,
        ),
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
    ).label(
        "latitude"
    )

    complaint_longitude = func.ST_X(
        complaint_geometry
    ).label(
        "longitude"
    )

    effective_category = func.coalesce(
        Complaint.final_category,
        Complaint.predicted_category,
    ).label(
        "effective_category"
    )

    distance_meters = func.ST_Distance(
        Complaint.location,
        search_point,
    ).label(
        "distance_meters"
    )

    statement = (
        select(
            Complaint,
            effective_category,
            complaint_latitude,
            complaint_longitude,
            distance_meters,
        )
        .where(
            func.ST_DWithin(
                Complaint.location,
                search_point,
                radius_meters,
            )
        )
        .order_by(
            distance_meters.asc()
        )
    )

    statement = _apply_created_at_filter(
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

    results: list[
        dict[str, object]
    ] = []

    for (
        complaint,
        category_value,
        latitude_value,
        longitude_value,
        distance_value,
    ) in rows:
        category_name = (
            category_value.value
            if hasattr(
                category_value,
                "value",
            )
            else str(
                category_value
            )
        )

        status_name = (
            complaint.status.value
            if hasattr(
                complaint.status,
                "value",
            )
            else str(
                complaint.status
            )
        )

        results.append(
            {
                "id":
                    complaint.id,

                "application_no":
                    complaint.application_no,

                "complaint_text":
                    complaint.complaint_text,

                "category":
                    category_name,

                "status":
                    status_name,

                "latitude":
                    float(
                        latitude_value
                    ),

                "longitude":
                    float(
                        longitude_value
                    ),

                "distance_meters":
                    float(
                        distance_value
                    ),
            }
        )

    return {
        "center_latitude":
            latitude,

        "center_longitude":
            longitude,

        "radius_meters":
            radius_meters,

        "category_filter": (
            category.value
            if category is not None
            else None
        ),

        "count":
            len(results),

        "results":
            results,
    }
def get_nearby_summary(
    db: Session,
    latitude: float,
    longitude: float,
    radius_meters: float = 1000.0,
    category: ComplaintCategory | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
) -> dict[str, object]:
    nearby = get_nearby_complaints(
        db=db,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        category=category,
        from_date=from_date,
        to_date=to_date,
    )

    results = nearby["results"]

    category_counts: dict[str, int] = {}

    for item in results:
        category_name = str(
            item["category"]
        )

        category_counts[
            category_name
        ] = (
            category_counts.get(
                category_name,
                0,
            )
            + 1
        )

    total = int(
        nearby["count"]
    )

    category_distribution = [
        {
            "category":
                category_name,

            "count":
                count,

            "percentage": (
                count / total
                if total > 0
                else 0.0
            ),
        }
        for (
            category_name,
            count,
        )
        in category_counts.items()
    ]

    category_distribution.sort(
        key=lambda item: (
            -int(
                item["count"]
            ),
            str(
                item["category"]
            ),
        )
    )

    if results:
        nearest_item = results[0]

        nearest_distance_meters = float(
            nearest_item[
                "distance_meters"
            ]
        )

        nearest_category = str(
            nearest_item[
                "category"
            ]
        )

        nearest_status = str(
            nearest_item[
                "status"
            ]
        )
    else:
        nearest_distance_meters = None
        nearest_category = None
        nearest_status = None

    dominant_category = None
    dominant_category_count = 0
    dominant_category_percentage = 0.0

    if category_distribution:
        dominant = (
            category_distribution[0]
        )

        dominant_category = str(
            dominant["category"]
        )

        dominant_category_count = int(
            dominant["count"]
        )

        dominant_category_percentage = float(
            dominant["percentage"]
        )

    return {
        "center_latitude":
            nearby["center_latitude"],

        "center_longitude":
            nearby["center_longitude"],

        "radius_meters":
            nearby["radius_meters"],

        "category_filter":
            nearby["category_filter"],

        "count":
            total,

        "nearest_distance_meters":
            nearest_distance_meters,

        "nearest_category":
            nearest_category,

        "nearest_status":
            nearest_status,

        "dominant_category":
            dominant_category,

        "dominant_category_count":
            dominant_category_count,

        "dominant_category_percentage":
            dominant_category_percentage,

        "category_distribution":
            category_distribution,
    }
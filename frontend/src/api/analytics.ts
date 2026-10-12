import type {
  AnalyticsDateRange,
  AnalyticsSummary,
  CategoryDistributionItem,
  TrendItem,
  MapPointItem,
  NearbyComplaintsResponse,
  NeighborhoodAnalyticsItem,
  ModelHumanComparisonResponse,
} from "../types/analytics";


const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL;


function getAccessToken(): string {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  if (!accessToken) {
    throw new Error(
      "Oturum bulunamadı.",
    );
  }

  return accessToken;
}


function getApiBaseUrl(): string {
  if (!API_BASE_URL) {
    throw new Error(
      "VITE_API_BASE_URL yapılandırılmamış.",
    );
  }

  return API_BASE_URL;
}


function buildDateQuery(
  range: AnalyticsDateRange,
): string {
  const params =
    new URLSearchParams();

  if (range.fromDate) {
    params.set(
      "from_date",
      range.fromDate,
    );
  }

  if (range.toDate) {
    params.set(
      "to_date",
      range.toDate,
    );
  }

  const query =
    params.toString();

  return query
    ? `?${query}`
    : "";
}


async function getErrorMessage(
  response: Response,
): Promise<string> {
  try {
    const body =
      await response.json();

    if (
      body
      && typeof body.detail
      === "string"
    ) {
      return body.detail;
    }
  } catch {
    // JSON olmayan hata cevabında
    // genel mesaj kullanılır.
  }

  return (
    `İstek başarısız: ${response.status}`
  );
}


async function managerGet<T>(
  path: string,
): Promise<T> {
  const accessToken =
    getAccessToken();

  const response =
    await fetch(
      `${getApiBaseUrl()}${path}`,
      {
        headers: {
          Authorization:
            `Bearer ${accessToken}`,
        },
      },
    );

  if (response.status === 401) {
    sessionStorage.clear();

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (response.status === 403) {
    throw new Error(
      "Bu işlem için yönetici yetkisi gerekli.",
    );
  }

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
      ),
    );
  }

  return response.json();
}


export async function getAnalyticsSummary(
  range: AnalyticsDateRange = {},
): Promise<AnalyticsSummary> {
  return managerGet<AnalyticsSummary>(
    (
      "/analytics/summary"
      + buildDateQuery(range)
    ),
  );
}


export async function getCategoryDistribution(
  range: AnalyticsDateRange = {},
): Promise<
  CategoryDistributionItem[]
> {
  return managerGet<
    CategoryDistributionItem[]
  >(
    (
      "/analytics/categories"
      + buildDateQuery(range)
    ),
  );
}


export async function getComplaintTrend(
  range: AnalyticsDateRange = {},
): Promise<TrendItem[]> {
  return managerGet<TrendItem[]>(
    (
      "/analytics/trend"
      + buildDateQuery(range)
    ),
  );
}


export async function getComplaintMapPoints(
  range: AnalyticsDateRange = {},
): Promise<MapPointItem[]> {
  return managerGet<MapPointItem[]>(
    (
      "/analytics/map-points"
      + buildDateQuery(range)
    ),
  );
}


export interface NearbyComplaintQuery {
  latitude: number;
  longitude: number;
  radiusMeters: number;

  category?: string;
  range?: AnalyticsDateRange;
}


export async function getNearbyComplaints(
  query: NearbyComplaintQuery,
): Promise<NearbyComplaintsResponse> {
  const params =
    new URLSearchParams({
      latitude: String(
        query.latitude,
      ),
      longitude: String(
        query.longitude,
      ),
      radius_meters: String(
        query.radiusMeters,
      ),
    });

  if (query.category) {
    params.set(
      "category",
      query.category,
    );
  }

  if (
    query.range?.fromDate
  ) {
    params.set(
      "from_date",
      query.range.fromDate,
    );
  }

  if (
    query.range?.toDate
  ) {
    params.set(
      "to_date",
      query.range.toDate,
    );
  }

  return managerGet<
    NearbyComplaintsResponse
  >(
    "/analytics/nearby?"
    + params.toString(),
  );
}


export async function getNeighborhoodAnalytics(
  range: AnalyticsDateRange = {},
): Promise<NeighborhoodAnalyticsItem[]> {
  return managerGet<
    NeighborhoodAnalyticsItem[]
  >(
    (
      "/analytics/neighborhoods"
      + buildDateQuery(range)
    ),
  );
}

export async function getModelHumanComparison(
  range: AnalyticsDateRange = {},
): Promise<ModelHumanComparisonResponse> {
  return managerGet<
    ModelHumanComparisonResponse
  >(
    (
      "/analytics/model-human"
      + buildDateQuery(range)
    ),
  );
}

